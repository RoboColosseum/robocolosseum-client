import pytest
from colosseum_client.continuity_quality import assess_timing, require_timing
from colosseum_client.continuous_recording import align
from test_continuous_recording import fixture, CAMERAS


def sparse_states():
    states,frames,targets=fixture(5.)
    states=[r for r in states if r['arm_id']!='left' or r['sample_index'] not in (51,52)]
    row=next(r for r in states if r['arm_id']=='left' and r['sample_index']==53)
    row['sample_monotonic_s']+=.005;row['feedback_received_monotonic_s']+=.005
    return states,frames,targets


def test_state_age_is_recorded_without_rejecting_or_fabricating_feedback():
    states,frames,targets=sparse_states()
    report=assess_timing(states,frames,targets,end=15.,cameras=CAMERAS)
    require_timing(report)
    assert report['complete'] and report['version']==3
    assert report['streams']['left']['violations']>0
    assert not report['streams']['left']['age_is_blocking']
    rows,columns,_=align(states,frames,targets,end=15.,cameras=CAMERAS)
    found=False
    for row,receipt,seq in zip(rows,columns['observation.source_left_feedback_received_monotonic_s'],columns['observation.source_left_feedback_sequence']):
        assert receipt<=row['captured_monotonic_s']
        source=next(r for r in states if r['arm_id']=='left' and r['feedback_sequence']==seq)
        assert row['joints'][:6]==source['joint_position_rad']
        found|=row['captured_monotonic_s']-receipt>.02
    assert found
    with pytest.raises(ValueError):align(states,frames,targets,end=15.,cameras=CAMERAS,version=2)
    with pytest.raises(ValueError):require_timing(assess_timing(states,frames,targets,end=15.,cameras=CAMERAS,version=2))


def test_camera_over_100ms_still_blocks():
    states,frames,targets=sparse_states()
    frames=[r for r in frames if r['camera_id']!='head_image' or not 11<r['received_monotonic_s']<11.2]
    report=assess_timing(states,frames,targets,end=15.,cameras=CAMERAS)
    assert not report['complete']
    with pytest.raises(ValueError,match='head_image'):require_timing(report)
    with pytest.raises(ValueError,match='0.1s'):align(states,frames,targets,end=15.,cameras=CAMERAS)


def test_no_causal_state_is_still_rejected():
    states,frames,targets=fixture()
    states=[r for r in states if r['feedback_received_monotonic_s']>10.2]
    report=assess_timing(states,frames,targets,end=13.,cameras=CAMERAS)
    with pytest.raises(ValueError,match='no causal source'):require_timing(report)
    with pytest.raises(ValueError):align(states,frames,targets,end=13.,cameras=CAMERAS)
