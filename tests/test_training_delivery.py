"""Physical signal alignment and final delivery regression coverage."""
import json
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from colosseum_client.training_delivery import physical_vectors, validate_delivery
from colosseum_client.roboeval_compact import digest


def test_distinct_arms_use_latest_past_feedback_and_actual_commands():
    states, actions = [], []
    for side, sign in (('left', -1), ('right', 1)):
        for stamp in (1., 1.04):
            states.append(dict(arm_id=side, feedback_received_monotonic_s=stamp,
                joint_velocity_rad_s=[sign*stamp]*6, joint_torque_nm=[sign*2*stamp]*6,
                eef_position_world_m=[sign*stamp]*3, eef_orientation_world_wxyz=[1.,0.,0.,0.],
                gripper_width_m=stamp*.01, gripper_reading_normalized_unclipped=1.1))
            actions.append(dict(kind='sent_command', arm_id=side, command_monotonic_s=stamp,
                values=[0.]*6+[stamp*.1], gripper_command_width_m=stamp*.02))
    result = physical_vectors([{'captured_monotonic_s': t} for t in (1.,1.033333,1.066667)], states, actions)
    np.testing.assert_allclose(result['observation.joint_velocity'][:,0], [-1.,-1.,-1.04])
    np.testing.assert_allclose(result['observation.joint_velocity'][:,6], [1.,1.,1.04])
    np.testing.assert_allclose(result['action.gripper_command_width'][:,0], [.02,.02,.0208])
    assert 'observation.gripper_effort_nm' not in result  # no inferred effort
    assert not any('holding' in key for key in result)
    states[0]['joint_torque_nm'] = None
    with pytest.raises(ValueError, match='physical feedback'):
        physical_vectors([{'captured_monotonic_s': 1.}], states, actions)


def test_complete_unused_action_tail_and_events_preserved(tmp_path):
    from test_continuous_export import synthetic_run, CAMERAS
    from test_roboeval import setup_file
    from test_roboeval_shared import assignment, result
    from colosseum_client.roboeval import ACTION_SCHEMA
    from colosseum_client.roboeval_shared import record_event
    from colosseum_client.roboeval_feedback import build_comparison, outcome_for_run
    from colosseum_client.lerobot_export import export_trial
    a=assignment(); run=a['runs'][0]; source=tmp_path/'source'
    synthetic_run(source,setup_file(tmp_path/'cal'),run['id'],resolution=(640,480),with_effort=True)
    actions_path=source/'roboeval/meta/roboeval/actions.parquet'
    actions=pq.read_table(actions_path).to_pylist()
    for side in ('left','right'):
        tail=dict(actions[0],arm_id=side,kind='raw_policy',chunk_index=99,control_step=None,
                  values=[.9]*7,executable_prefix_length=2,source_block_id='complete-model-block')
        actions.append(tail)
    pq.write_table(pa.Table.from_pylist(actions,schema=ACTION_SCHEMA),actions_path)
    # Fixture setup, before export: reseal the deliberately constructed raw source.
    manifest=source/'roboeval/manifest.json';doc=json.loads(manifest.read_text())
    doc['files']['meta/roboeval/actions.parquet']=digest(actions_path)
    manifest.write_text(json.dumps(doc))
    record_event(source,a['id'],run['id'],'inference',stamp=10.2,
                 observation_monotonic_s=10.19,action_ready_monotonic_s=10.21,status='ok',plan_id=1)
    record_event(source,a['id'],run['id'],'operator_stop',stamp=10.6,reason='test')
    kwargs=dict(fps=30,robot_type='yam',task='physical signals',continuous_training_fps=30,
        omit_auxiliary_columns=True,training_delivery=True,export_destination=tmp_path/'delivery',
        metadata=dict(assignment_id=a['id'],run_id=run['id'],action_space='joint_position',
                      roboeval_outcome=outcome_for_run(build_comparison(a,result()),run['id'])))
    dataset=export_trial(source,CAMERAS,**kwargs)
    validate_delivery(dataset)
    ep=pq.read_table(dataset/'meta/episodes/chunk-000/file-000.parquet').to_pylist()[0]
    assert ep['roboeval/actions']==actions
    assert {'inference','operator_stop'} <= {e['kind'] for e in ep['roboeval/shared']}
    assert sum(p.is_file() for p in dataset.rglob('*'))==9
    marker=json.loads((dataset/'meta/colosseum.json').read_text())
    assert marker['training_layout']['outcome']['comparison']['raw_feedback']==result()['feedback']
    assert 'policy_targets' not in json.dumps(marker)
    corrupt=tmp_path/'corrupt';shutil.copytree(dataset,corrupt)
    path=corrupt/'data/chunk-000/file-000.parquet';table=pq.read_table(path)
    values=table.to_pylist();values[0]['observation.joint_velocity'][0]=float('nan')
    pq.write_table(pa.Table.from_pylist(values,schema=table.schema),path)
    marker['files']['data/chunk-000/file-000.parquet']=digest(path)
    (corrupt/'meta/colosseum.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError,match='physical feature'):
        validate_delivery(corrupt)
