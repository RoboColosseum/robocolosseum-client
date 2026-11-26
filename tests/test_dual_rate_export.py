import hashlib,json,shutil
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from colosseum_client.roboeval_export import validate_uploaded_extension
from colosseum_client.continuous_recording import align,command_coverage
from test_continuous_export import synthetic_run,CAMERAS
from test_roboeval import setup_file
from test_roboeval_shared import assignment,result
from colosseum_client.roboeval_feedback import build_comparison,outcome_for_run
from test_continuous_recording import fixture


def hashes(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.name!='.dataset-operation.lock'}










def test_missing_command_evidence_rejected():
    targets=[dict(frame=0,action_executed_monotonic_s=1.)]
    with pytest.raises(ValueError,match='paired command'):command_coverage(targets,{'action_source_frame_index':[0]},[])
    assert command_coverage(targets,{'action_source_frame_index':[]},[])['unreferenced_targets_without_paired_commands']==[0]


@pytest.mark.parametrize('hz',[30,100])
def test_wait_and_nonmatching_grid_boundary_are_causal(hz):
    states,frames,targets=fixture(16.)
    targets[1]['action_executed_monotonic_s']=20.015
    rows,columns,_=align(states,frames,targets,end=26.,cameras=CAMERAS,grid_hz=hz)
    for row in rows:
        assert row['action']==([1.]*14 if row['captured_monotonic_s']<20.015 else [2.]*14)
    assert np.all(np.asarray([r['captured_monotonic_s'] for r in rows])>=columns['action_source_executed_monotonic_s'])


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('with_effort', [False, True])
def test_page_export_entry_produces_30hz_only_and_preserves_legacy(tmp_path, legacy, with_effort):
    from types import SimpleNamespace
    from colosseum_client.evaluation import export_recorded_run
    from colosseum_client.lerobot_export import export_trial
    cal=setup_file(tmp_path/'cal'); a=assignment(); run=a['runs'][0]
    a['task']={'instruction':'dual rate','cameras':CAMERAS}
    source=tmp_path/run['id']; synthetic_run(source,cal,run['id'],resolution=(640,480),with_effort=with_effort)
    cp=source/'recording-context.json'; ctx=json.loads(cp.read_text())
    ctx.update(fps=30,robot_type='yam',test=False,action_space='joint_position')
    cp.write_text(json.dumps(ctx))
    (tmp_path/'manifest.json').write_text(json.dumps({'task_difficulty':3}))
    annotation=outcome_for_run(build_comparison(a,result()),run['id'])
    if legacy:
        export_trial(source,CAMERAS,fps=30,robot_type='yam',task='legacy',metadata={
            'assignment_id':a['id'],'run_id':run['id'],'action_space':'joint_position',
            'action_recording':'policy_targets','roboeval_outcome':annotation})
    before=hashes(source)
    cfg=SimpleNamespace(cameras=CAMERAS,institution='lab',scene='lab',evaluator='test')
    args=(cfg,a,run,tmp_path,'open',result()['outcomes'][run['side']],annotation)
    dataset=export_recorded_run(*args)
    assert dataset==source/('export_physical_30hz/lerobot' if legacy else 'lerobot')
    assert json.loads((dataset/'meta/info.json').read_text())['fps']==30
    assert not (dataset.parent/'raw_100hz').exists()
    assert len(hashes(dataset)) == 9
    validate_uploaded_extension(dataset)
    table = pq.read_table(dataset/'data/chunk-000/file-000.parquet').to_pydict()
    assert not any(k.startswith(('observation.source_', 'observation.capture_', 'action_source_')) for k in table)
    if with_effort:
        np.testing.assert_array_equal(table['observation.gripper_effort_nm'], [[-.25,.75]]*15)
    else:
        assert 'observation.gripper_effort_nm' not in table
    info=json.loads((dataset/'meta/info.json').read_text())
    stats=json.loads((dataset/'meta/stats.json').read_text())
    for keys in (info['features'],stats):
        assert not any(k.startswith(('observation.source_', 'observation.capture_', 'action_source_')) for k in keys)
    assert len(table['action']) == 15 and np.isfinite(table['action']).all()
    np.testing.assert_array_equal(np.asarray(table['action'])[:9], np.zeros((9,14)))
    np.testing.assert_allclose(np.asarray(table['action'])[9:], .1)
    episode=pq.read_table(dataset/'meta/episodes/chunk-000/file-000.parquet').to_pylist()[0]
    assert 'roboeval/states' not in episode and 'roboeval/frames' not in episode
    assert episode['roboeval/actions'] == pq.read_table(source/'roboeval/meta/roboeval/actions.parquet').to_pylist()
    assert episode['roboeval/shared']
    np.testing.assert_allclose(table['observation.joint_velocity'], .1)
    np.testing.assert_allclose(table['observation.joint_torque'], .2)
    np.testing.assert_allclose(table['observation.gripper_width'], .05)
    assert len(table['observation.eef_orientation'][0]) == 8
    from colosseum_client.roboeval_compact import probe
    assert all(len(probe(video)['streams']) == 1 for video in dataset.rglob('*.mp4'))
    after=hashes(source)
    assert all(after[k]==v for k,v in before.items())
    assert export_recorded_run(*args)==dataset
    assert hashes(source)==after
