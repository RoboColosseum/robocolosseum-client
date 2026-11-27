import copy
import pytest
from colosseum_client.continuous_recording import align
from colosseum_client.continuity_quality import assess_timing, require_timing
from test_continuous_recording import fixture, CAMERAS


def test_late_copy_does_not_turn_already_received_feedback_into_future_feedback():
    states, frames, targets = fixture()
    for r in states: r['sample_monotonic_s'] += .025
    rows, cols, _ = align(states, frames, targets, end=13., cameras=CAMERAS)
    assert all(t<=r['captured_monotonic_s'] for t,r in zip(cols['observation.source_left_feedback_received_monotonic_s'],rows))
    assert cols['observation.source_left_sample_monotonic_s'][0]>rows[0]['captured_monotonic_s']
    with pytest.raises(ValueError): align(states, frames, targets, end=13., cameras=CAMERAS, version=1)


def test_future_receipt_action_and_frame_are_never_selected():
    states, frames, targets = fixture()
    rows, cols, mapped = align(states, frames, targets, end=13., cameras=CAMERAS)
    for i, row in enumerate(rows):
        stamp = row['captured_monotonic_s']
        for side in ('left', 'right'):
            assert cols[f'observation.source_{side}_feedback_received_monotonic_s'][i]<=stamp
        assert cols['action_source_executed_monotonic_s'][i]<=stamp
        for c in CAMERAS: assert mapped[c][i]['received_monotonic_s']<=stamp


def test_pre_action_gap_is_retained_but_not_a_primary_interval_failure():
    states, frames, targets = fixture()
    states = [r for r in states if not 10.01<=r['feedback_received_monotonic_s']<=10.07]
    report = assess_timing(states, frames, targets, end=13., cameras=CAMERAS)
    assert report['complete']
    assert all(v['violations']==0 for v in report['streams'].values())


def test_real_grid_gaps_are_reported_and_not_fixed_by_alignment():
    states, frames, targets = fixture()
    states = [r for r in states if r['arm_id']!='left' or not 10.5<r['feedback_received_monotonic_s']<10.56]
    report = assess_timing(states, frames, targets, end=13., cameras=CAMERAS)
    assert report['streams']['left']['violations']>0
    assert report['streams']['right']['violations']==0
    with pytest.raises(ValueError, match='grid sources'):
        require_timing(assess_timing(states,frames,targets,end=13.,cameras=CAMERAS,version=2))
    with pytest.raises(ValueError): align(states, frames, targets, end=13., cameras=CAMERAS,version=2)


def test_batched_copy_timestamps_allowed_only_in_new_version():
    states, frames, targets = fixture()
    for r in states:
        r['sample_monotonic_s'] = 14.  # Imported batch, no synthetic physical receipts.
    align(states, frames, targets, end=13., cameras=CAMERAS)
    with pytest.raises(ValueError, match='Repeated'): align(states, frames, targets, end=13., cameras=CAMERAS, version=1)
