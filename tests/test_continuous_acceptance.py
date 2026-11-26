"""Acceptance boundaries using synthetic summaries/sources; no device access."""
from types import SimpleNamespace

import pytest

from colosseum_client.roboeval import RoboEvalRecorder
from colosseum_client.continuous_recording import align
from colosseum_client.continuity_quality import assess_timing, require_timing
from test_continuous_recording import CAMERAS, fixture


def capture_summary(fps):
    # Bypass construction: this check only reads online capture summaries.
    recorder = RoboEvalRecorder.__new__(RoboEvalRecorder)
    recorder.continuous = True
    recorder.hz = 100
    recorder.started = 0.
    recorder.setup = SimpleNamespace(arms=['left', 'right'])
    recorder.required_camera_keys = CAMERAS
    recorder.counts = dict(states=2002, actions=4)
    recorder.action_kinds = {side: {'raw_policy', 'sent_command'} for side in recorder.setup.arms}
    recorder.state_stats = {side: dict(samples=1001, first=0., last=10., max_gap_s=.01)
                            for side in recorder.setup.arms}
    recorder.frame_stats = {camera: dict(frames=round(fps*10)+1, first=0., last=10., max_gap_s=1/fps)
                            for camera in CAMERAS}
    return recorder


@pytest.mark.parametrize('fps', [15, 25, 30])
def test_real_frame_rate_at_or_above_customer_minimum_is_accepted(fps):
    recorder = capture_summary(fps)
    recorder._validate_capture(10.)
    assert all(s['actual_fps'] == fps for s in recorder.frame_stats.values())
    assert all(s['actual_hz'] == 100 for s in recorder.state_stats.values())


@pytest.mark.parametrize('camera', CAMERAS)
def test_each_camera_must_reach_15_fps(camera):
    states, frames, targets = fixture()
    frames = [r for r in frames if r['camera_id']!=camera] + [
        dict(camera_id=camera, received_monotonic_s=10+i/14.9, source_frame_number=i)
        for i in range(45)]
    with pytest.raises(ValueError, match='source rate'):
        require_timing(assess_timing(states,frames,targets,end=13.,cameras=CAMERAS))


@pytest.mark.parametrize('camera', CAMERAS)
def test_raw_integrity_does_not_misreport_camera_gap_as_failed_write(camera):
    recorder = capture_summary(25)
    recorder.frame_stats[camera]['max_gap_s'] = .1
    recorder._validate_capture(10.)
    recorder.frame_stats[camera]['max_gap_s'] = .100001
    recorder._validate_capture(10.)  # Grid gap rejection is tested below.


@pytest.mark.parametrize('side', ['left', 'right'])
def test_recorder_interval_is_not_the_physical_source_age(side):
    recorder = capture_summary(25)
    recorder.state_stats[side]['max_gap_s'] = .02
    recorder._validate_capture(10.)
    recorder.state_stats[side]['max_gap_s'] = .020001
    recorder._validate_capture(10.)
    states,frames,targets=fixture()
    states=[r for r in states if r['arm_id']!=side or not 10.5<r['feedback_received_monotonic_s']<10.55]
    with pytest.raises(ValueError, match='exceed 20 ms'):
        require_timing(assess_timing(states,frames,targets,end=13.,cameras=CAMERAS,version=2))


@pytest.mark.parametrize('camera', CAMERAS)
def test_export_rejects_stale_camera_source_despite_25_fps_average(camera):
    states, _, targets = fixture()
    frames = [dict(camera_id=c, received_monotonic_s=10+i/25, frame_index=i,
                   source_frame_number=i, video_frame_index=i, video_path=c+'.mp4')
              for c in CAMERAS for i in range(76)]
    rows, columns, _ = align(states, frames, targets, end=13., cameras=CAMERAS)
    assert len(rows) == 290
    # Preserve the same frame count and total duration; create a 120 ms hole.
    for i, stamp in [(26, 11.001), (27, 11.002)]:
        next(f for f in frames if f['camera_id'] == camera and f['frame_index'] == i)['received_monotonic_s'] = stamp
    with pytest.raises(ValueError, match='No causal source within 0.1s'):
        align(states, frames, targets, end=13., cameras=CAMERAS)
