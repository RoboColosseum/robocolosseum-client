"""Synthetic data only. No robot, camera pipeline, service or network access."""
import ast
import copy
import json
from pathlib import Path
import numpy as np
import pytest

from colosseum_client.continuous_recording import align

CAMERAS = ['head_image', 'left_image', 'right_image']


def fixture(end=3.):
    states = [dict(arm_id=side, sample_index=i, sample_monotonic_s=10+i/100,
                   feedback_received_monotonic_s=10+i/100, feedback_sequence=i,
                   joint_position_rad=[float(i)]*6, gripper_reading_normalized_unclipped=.5)
              for side in ('left', 'right') for i in range(round(end*100)+1)]
    frames = [dict(camera_id=c, received_monotonic_s=10+i/30, frame_index=i,
                   source_frame_number=i, video_frame_index=i, video_path=c+'.mp4')
              for c in CAMERAS for i in range(round(end*30)+1)]
    targets = [dict(frame=0, action=[1.]*14, action_executed_monotonic_s=10.1),
               dict(frame=1, action=[2.]*14, action_executed_monotonic_s=10+end-.1)]
    return states, frames, targets


@pytest.mark.parametrize('delay', [.5, 2.5, 5., 16.])
def test_waits_keep_real_time_and_original_target_without_extra_execution(delay):
    states, frames, targets = fixture(delay+.3)
    original = copy.deepcopy(targets)
    rows, columns, mapping = align(states, frames, targets, end=10+delay+.3, cameras=CAMERAS)
    assert len(rows) == round((delay+.2)*100)
    assert targets == original  # Alignment never mutates or executes commands.
    for row, executed in zip(rows, columns['action_source_executed_monotonic_s']):
        assert executed <= row['captured_monotonic_s']
        assert row['action'][0] == (1. if row['captured_monotonic_s'] < targets[1]['action_executed_monotonic_s'] else 2.)
    assert len({r['source_frame_number'] for r in mapping['head_image']}) < len(rows)/2
    assert rows[0]['captured_monotonic_s'] == 10.1  # Prelude excluded only from main table.
    for side in ('left', 'right'):
        age = np.asarray([r['captured_monotonic_s'] for r in rows])-columns[f'observation.source_{side}_sample_monotonic_s']
        assert np.all((age>=0)&(age<=.020000001))


@pytest.mark.parametrize('problem', ['repeated_feedback', 'old_feedback', 'missing_states', 'missing_frames', 'nan_action', 'missing_execution'])
def test_incomplete_or_fabricated_sources_rejected(problem):
    states, frames, targets = fixture()
    if problem == 'repeated_feedback': states[20]['feedback_sequence'] = states[19]['feedback_sequence']
    if problem == 'old_feedback':
        for s in states: s['feedback_received_monotonic_s'] -= .05
    if problem == 'missing_states': states = [s for s in states if not 10.5<s['sample_monotonic_s']<10.6]
    if problem == 'missing_frames': frames = [f for f in frames if not 10.5<f['received_monotonic_s']<10.8]
    if problem == 'nan_action': targets[1]['action'][0] = float('nan')
    if problem == 'missing_execution': targets[0]['action_executed_monotonic_s'] = None
    with pytest.raises(ValueError): align(states,frames,targets,end=13,cameras=CAMERAS,version=2)




def test_encoder_uses_real_source_frames_at_100fps(tmp_path):
    import subprocess
    from colosseum_client.continuous_recording import encode_aligned_video
    from colosseum_client.roboeval_compact import probe
    root=tmp_path/'roboeval';root.mkdir()
    src=root/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=red:s=1280x720:r=30',
                    '-frames:v','6','-c:v','libx264','-pix_fmt','yuv420p',str(src)],check=True)
    rows=[dict(video_path='source.mp4',video_frame_index=i*30//100) for i in range(20)]
    out=tmp_path/'training.mp4'
    stats=encode_aligned_video(tmp_path,rows,out,100)
    v=probe(out)['streams'][0]
    assert (v['width'],v['height'],v['r_frame_rate'],int(v['nb_frames']))==(1280,720,'100/1',20)
    assert stats['count']==[20]
    assert not list(tmp_path.rglob('*.png'))


def test_config_has_opt_in_and_does_not_change_control_rate():
    from colosseum_client.robot_config import RobotClientConfig
    assert RobotClientConfig.__dataclass_fields__['roboeval_state_hz'].default==0
    assert RobotClientConfig.__dataclass_fields__['control_hz'].default==15


def test_gripper_state_preserves_legacy_clipping_and_raw_source():
    states,frames,targets=fixture()
    for row in states: row['gripper_reading_normalized_unclipped']=1.02
    rows,_,_=align(states,frames,targets,end=13,cameras=CAMERAS)
    assert rows[0]['gripper']==[1.,1.]
    assert states[0]['gripper_reading_normalized_unclipped']==1.02


def test_decoder_closes_pipe_before_termination_and_bounds_shutdown():
    import subprocess
    from colosseum_client.continuous_recording import _stop_process
    events=[]
    class Stream:
        closed=False
        def close(self):self.closed=True;events.append('close')
    class Process:
        stdout=Stream()
        stdin=None
        waits=0
        def poll(self):return None
        def terminate(self):
            assert self.stdout.closed
            events.append('terminate')
        def wait(self,timeout):
            assert timeout==3
            self.waits+=1
            if self.waits==1:raise subprocess.TimeoutExpired('fake-ffmpeg',timeout)
            events.append('wait')
        def kill(self):events.append('kill')
    _stop_process(Process())
    assert events==['close','terminate','kill','wait']


@pytest.mark.parametrize('actual', [(1280,720,30),(640,480,30),(1280,720,15)])
def test_camera_requires_native_profile_without_starting_a_real_device(monkeypatch,actual):
    import sys
    from types import SimpleNamespace
    from colosseum_client.robots.yam import _RealSenseCamera
    calls=[]
    profile=SimpleNamespace(width=lambda:actual[0],height=lambda:actual[1],fps=lambda:actual[2])
    stream=SimpleNamespace(as_video_stream_profile=lambda:profile)
    pipe=SimpleNamespace(start=lambda _:SimpleNamespace(get_stream=lambda _:stream),stop=lambda:calls.append('stop'))
    config=SimpleNamespace(enable_device=lambda _:None,enable_stream=lambda *a:calls.append(a))
    rs=SimpleNamespace(pipeline=lambda:pipe,config=lambda:config,stream=SimpleNamespace(color='color'),format=SimpleNamespace(rgb8='rgb8'))
    monkeypatch.setitem(sys.modules,'pyrealsense2',rs)
    if actual==(640,480,30):
        _RealSenseCamera('fake',1000,continuous=True)
    else:
        with pytest.raises(RuntimeError,match='native'):_RealSenseCamera('fake',1000,continuous=True)
        assert 'stop' in calls
    assert ('color',640,480,'rgb8',30) in calls


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf')])
def test_incomplete_gripper_effort_is_not_fabricated(bad):
    states,frames,targets=fixture()
    for row in states:row['gripper_effort_nm']=.3 if row['arm_id']=='left' else bad
    with pytest.raises(ValueError,match='gripper effort'):
        align(states,frames,targets,end=13.,cameras=CAMERAS,grid_hz=30)
