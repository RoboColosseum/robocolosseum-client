"""Synthetic 720p source, real exporters, real encoder and integrity checks."""
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from colosseum_client.roboeval import RoboEvalRecorder, STATE_SCHEMA, ACTION_SCHEMA, FRAME_SCHEMA
from colosseum_client.roboeval_shared import record_event
from colosseum_client.roboeval_feedback import build_comparison, outcome_for_run
from colosseum_client.lerobot_export import export_trial
from colosseum_client.roboeval_export import validate_uploaded_extension
from test_roboeval import setup_file, FakeKinematics
from test_roboeval_shared import assignment, result

CAMERAS = ['head_image', 'left_image', 'right_image']


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def synthetic_run(path, cal, rid, source_fps=30, resolution=(1280,720), with_effort=False):
    recorder=RoboEvalRecorder(path,cal,hz=100,continuous=True,
        identity={'assignment_id':'ev_shared','run_id':rid},kinematics_factory=FakeKinematics)
    root=path/'roboeval';meta=root/'meta/roboeval'
    states=[]
    for side in ('left','right'):
        for i in range(61):
            states.append(dict(arm_id=side,sample_index=i,sample_monotonic_s=10+i/100,
                feedback_sequence=i,feedback_received_monotonic_s=10+i/100,
                joint_position_rad=[i*.001]*6,joint_velocity_rad_s=[.1]*6,joint_torque_nm=[.2]*6,
                eef_position_world_m=[0.,0.,0.],eef_orientation_world_wxyz=[1.,0.,0.,0.],
                gripper_reading_normalized_unclipped=.5,gripper_width_m=.05,
                gripper_effort_nm=(-.25 if side=='left' else .75) if with_effort else None))
    pq.write_table(pa.Table.from_pylist(states,schema=STATE_SCHEMA),meta/'states.parquet')
    # Distinct raw model blocks and physically issued command evidence remain separate.
    actions=[dict(kind=kind,arm_id=side,plan_id=1,chunk_index=i,control_step=i,
                  command_monotonic_s=10.1+i*.3,recorded_monotonic_s=10.1+i*.3,
                  action_space='joint_position',values=[i*.1]*7,
                  gripper_command_width_m=recorder.setup.width(side,i*.1) if kind=='sent_command' else None)
             for kind in ('raw_policy','sent_command') for side in ('left','right') for i in range(2)]
    pq.write_table(pa.Table.from_pylist(actions,schema=ACTION_SCHEMA),meta/'actions.parquet')
    frames=[]
    frame_count=int(.6*source_fps)+1
    for ci,camera in enumerate(CAMERAS):
        rel=f'videos/roboeval/{camera}/chunk-000/file-000.mp4'
        video=root/rel;video.parent.mkdir(parents=True)
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
            f'color=c={"red" if ci==0 else "blue"}:s={resolution[0]}x{resolution[1]}:r=30','-frames:v',str(frame_count),
            '-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True)
        for i in range(frame_count):
            frames.append(dict(arm_id=None if camera=='head_image' else camera.split('_')[0],
                camera_id=camera,frame_index=i,source_frame_number=i,source_timestamp_ms=i*1000/source_fps,
                source_timestamp_domain='synthetic',received_monotonic_s=10+i/source_fps,
                video_path=rel,video_frame_index=i,video_timestamp_s=i/30))
    pq.write_table(pa.Table.from_pylist(frames,schema=FRAME_SCHEMA),meta/'frames.parquet')
    write(meta/'status.json',dict(complete=True,continuous=True,target_state_hz=100,
        synthetic_test=True,counts={'states':len(states),'actions':len(actions),'frames':len(frames)}))
    write(root/'manifest.json',dict(schema_version=1,files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob('*') if p.is_file()}))
    rows=[dict(frame=i,seconds=i*.3,captured_monotonic_s=10+i*.3,
               action_executed_monotonic_s=10.1+i*.3,joints=[0.]*12,gripper=[.5,.5],
               cartesian_position=[],action=[i*.1]*14) for i in range(2)]
    (path/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(path/'recording-status.json',dict(complete=True,frames=2,image_storage='continuous_source'))
    write(path/'recording-context.json',dict(roboeval_required=True,roboeval_shared_required=True,
        roboeval_state_hz=100,execution_enabled=True,continuous_end_monotonic_s=10.6))
    record_event(path,'ev_shared',rid,'episode_start',stamp=10.)
    record_event(path,'ev_shared',rid,'episode_end',stamp=10.6)


@pytest.fixture(scope='module', params=[(15,(1280,720)),(25,(1280,720)),(30,(1280,720)),(30,(640,480))])
def continuous_pair(tmp_path_factory, request):
    root=tmp_path_factory.mktemp(f'continuous-synthetic-{request.param}fps')
    cal=setup_file(root/'cal');a=assignment();res=result()
    res['b6']={r['id']:dict(drop='unknown',damage='no',spill='not_applicable') for r in a['runs']}
    comparison=build_comparison(a,res);outputs=[]
    for r in a['runs']:
        path=root/r['id'];synthetic_run(path,cal,r['id'],source_fps=request.param[0],resolution=request.param[1])
        metadata=dict(assignment_id=a['id'],run_id=r['id'],action_space='joint_position',
            action_recording='policy_targets',task_difficulty=3,task_difficulty_scale=10,
            roboeval_outcome=outcome_for_run(comparison,r['id']))
        output=export_trial(path,CAMERAS,fps=30,robot_type='yam',task='synthetic test',metadata=metadata)
        outputs.append((path,output,metadata))
    return outputs


def test_full_pair_18_files_real_encoder_alignment_and_retry(continuous_pair):
    for path,output,metadata in continuous_pair:
        validate_uploaded_extension(output)
        files={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
               for p in output.rglob('*') if p.is_file()}
        assert len(files)==9
        data=pq.read_table(output/'data/chunk-000/file-000.parquet').to_pydict()
        assert len(data['timestamp'])==50
        assert np.allclose(data['timestamp'],np.arange(50)/100)
        assert data['action'][10]==[0.]*14  # no action during this wait; target persists
        marker=json.loads((output/'meta/colosseum.json').read_text())
        assert marker['roboeval_layout']['version']==2
        assert marker['task_difficulty']==3
        export_trial(path,CAMERAS,fps=30,robot_type='yam',task='synthetic test',metadata=metadata)
        assert files=={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in output.rglob('*') if p.is_file()}
        # Decode every stream, including archived pre-action frames.
        for video in output.rglob('*.mp4'):
            subprocess.run(['ffmpeg','-v','error','-i',str(video),'-map','0:v','-f','null','-'],check=True)


def test_alignment_tampering_rejected_even_with_new_hash(continuous_pair,tmp_path):
    _,output,_=continuous_pair[0];dest=tmp_path/'tampered';shutil.copytree(output,dest)
    p=dest/'data/chunk-000/file-000.parquet';table=pq.read_table(p);rows=table.to_pylist()
    rows[10]['action']=[.9]*14;pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),p)
    m=dest/'meta/colosseum.json';doc=json.loads(m.read_text())
    doc['files'][p.relative_to(dest).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest();write(m,doc)
    with pytest.raises(ValueError,match='Continuous alignment differs'):validate_uploaded_extension(dest)


def test_resolution_metadata_corruption_rejected(continuous_pair):
    from colosseum_client.continuous_recording import validate_continuous
    from colosseum_client.roboeval_compact import load_stream
    _, output, _ = continuous_pair[0]
    marker = json.loads((output/'meta/colosseum.json').read_text())
    current = marker['continuous_recording']['source_resolution']
    marker['continuous_recording']['source_resolution'] = [640,480] if current == [1280,720] else [1280,720]
    with pytest.raises(ValueError,match='dimensions'):
        validate_continuous(output, marker, states=load_stream(output,'states',original=True).to_pylist(),frames=load_stream(output,'frames',original=True).to_pylist())
