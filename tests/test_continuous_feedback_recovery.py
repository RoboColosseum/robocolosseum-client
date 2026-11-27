from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from colosseum_client import roboeval
from test_roboeval import setup_file, FakeKinematics


def sample(monkeypatch, sequences):
    clock = [0.]
    index = [0]
    class Stop:
        stopped = False
        def is_set(self): return self.stopped
        def set(self): self.stopped = True
        def wait(self, delay):
            clock[0] += delay
            index[0] += 1
            if index[0] == len(sequences): self.set()
    def feedback():
        seq = sequences[index[0]]
        return {s: dict(sequence=seq, received_monotonic_s=0. if seq==1 else clock[0],
                       running=True, position=[0.]*6, velocity=[0.]*6,
                       torque_nm=[0.]*6, gripper_reading=.5) for s in ('left','right')}
    monkeypatch.setattr(roboeval.time, 'monotonic', lambda: clock[0])
    obj = roboeval.RoboEvalRecorder.__new__(roboeval.RoboEvalRecorder)
    obj.continuous=True; obj.hz=100; obj.stop=Stop(); obj.error=None; obj.missed_sampling_ticks=0
    obj.setup=SimpleNamespace(arms=('left','right'))
    obj.robot=SimpleNamespace(get_roboeval_feedback=feedback)
    rows=[]; obj._put=lambda kind,row: rows.append(row)
    obj._sample_loop()
    return obj,rows


def test_short_duplicate_is_not_a_sample_and_collection_resumes(monkeypatch):
    recorder,rows=sample(monkeypatch,[1,1,2])
    assert recorder.error is None
    assert recorder.duplicate_feedback_reads==dict(left=1,right=1)
    for side in ('left','right'):
        selected=[r for r in rows if r['arm_id']==side]
        assert [r['feedback_sequence'] for r in selected]==[1,2]
        assert [r['sample_monotonic_s'] for r in selected]==[0.,.02]


def test_stale_feedback_is_marked_without_stopping_other_streams(monkeypatch):
    recorder,rows=sample(monkeypatch,[1,1,1,1])
    assert recorder.error is None
    assert set(recorder.quality_issues)=={'left','right'}
    # Raw quality events no longer certify/reject an unrelated action interval.
    recorder.counts = dict(states=0, actions=0)
    with pytest.raises(RuntimeError,match='Missing arm'):recorder._validate_capture(.03)
    assert len(rows)==2  # No copies manufactured while feedback is frozen.
    assert recorder.feedback_read_stats['count'] == 4
    event = recorder.quality_issues['left']['events'][0]
    assert event['feedback_received_monotonic_s'] == 0.
    assert event['read_started_monotonic_s'] == .03
    assert event['read_completed_monotonic_s'] == .03
    assert event['sequence'] == event['previous'] == 1


def test_feedback_issue_evidence_is_bounded_without_losing_total_count():
    recorder = roboeval.RoboEvalRecorder.__new__(roboeval.RoboEvalRecorder)
    recorder.quality_issues = {}
    for i in range(100):
        recorder._feedback_issue('left',float(i),.03,i,i,
            details=dict(feedback_received_monotonic_s=float(i)-.03))
    issues=recorder.quality_issues['left']
    assert issues['count']==100 and len(issues['events'])==32 and issues['events_omitted']==68
    assert issues['last_monotonic_s']==99.


def test_reordered_feedback_is_not_accepted(monkeypatch):
    recorder,rows=sample(monkeypatch,[2,1])
    assert recorder.error is None and recorder.quality_issues
    assert len(rows)==2


def test_fresh_feedback_resumes_after_gap_without_copies(monkeypatch):
    recorder,rows=sample(monkeypatch,[1,1,1,1,2,3])
    assert recorder.error is None and recorder.quality_issues
    for side in ('left','right'):
        assert [r['feedback_sequence'] for r in rows if r['arm_id']==side]==[1,2,3]


def test_feedback_gap_does_not_reject_subsequent_camera_frames(tmp_path):
    recorder=roboeval.RoboEvalRecorder(tmp_path/'run',setup_file(tmp_path/'setup'),
        hz=100,continuous=True,identity={},kinematics_factory=FakeKinematics)
    recorder._feedback_issue('left',1.,.026,10,10)
    recorder.check()  # Quality is not an unrecoverable writer error.
    recorder.frame('left',{'source_frame_number':50})
    recorder.frame('right',{'source_frame_number':51})
    assert not recorder.stop.is_set()
    assert recorder.queue.get_nowait()==('frames',('left',{'source_frame_number':50}))
    assert recorder.queue.get_nowait()==('frames',('right',{'source_frame_number':51}))
    with pytest.raises(RuntimeError,match='Missing arm'):recorder._validate_capture(2.)


def test_sampler_failure_preserves_already_queued_states(tmp_path):
    recorder=roboeval.RoboEvalRecorder(tmp_path/'run',setup_file(tmp_path/'setup'),
        hz=100,continuous=True,identity={},kinematics_factory=FakeKinematics)
    for side in ('left','right'):
        recorder._put('states',dict(arm_id=side,sample_index=0,sample_monotonic_s=10.,
            feedback_sequence=1,feedback_received_monotonic_s=10.,joint_position_rad=[0.]*6,
            joint_velocity_rad_s=[0.]*6,joint_torque_nm=[0.]*6,gripper_reading_normalized_unclipped=.5))
    recorder.fail(RuntimeError('feedback unavailable'))
    recorder.queue.put(None)
    recorder._write_loop()
    assert len(pq.read_table(recorder.meta/'states.parquet'))==2
    assert recorder.counts['states']==2
    with pytest.raises(RuntimeError,match='incomplete'):recorder.check()
