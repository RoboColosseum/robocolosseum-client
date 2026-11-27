from types import SimpleNamespace
import pytest
from colosseum_client import roboeval


def run_sampler(monkeypatch, *, gap=False, stopped=False):
    clock=[10.]
    calls=[0]
    class Stop:
        stopped=False
        def is_set(self): return self.stopped
        def set(self): self.stopped=True
        def wait(self, delay):
            clock[0]+=.05  # Consumer scheduling pause; producer has real history.
            if calls[0]>=2: self.set()
    def history():
        calls[0]+=1
        sequences=[1] if calls[0]==1 else list(range(2,14))
        if gap and calls[0]==2: sequences.remove(4)
        return {s:[dict(sequence=i,received_monotonic_s=10+(i-1)*.004,
            running=not stopped,position=[i]*6,velocity=[0.]*6,torque_nm=[0.]*6,
            gripper_reading=.5,gripper_effort_nm=(-i*.01 if s=='left' else i*.01)) for i in sequences] for s in ('left','right')}
    monkeypatch.setattr(roboeval.time,'monotonic',lambda:clock[0])
    obj=roboeval.RoboEvalRecorder.__new__(roboeval.RoboEvalRecorder)
    obj.robot=SimpleNamespace(get_roboeval_feedback_history=history)
    obj.setup=SimpleNamespace(arms=('left','right'))
    obj.stop=Stop();obj.hz=100;obj.error=None;obj.feedback_read_stats={};obj.quality_issues={};obj.missed_sampling_ticks=0
    batches=[];obj._put=lambda kind,rows:batches.append((kind,rows))
    obj._history_loop()
    return obj,batches


def test_late_consumer_preserves_actual_sources_in_one_queue_item(monkeypatch):
    obj,batches=run_sampler(monkeypatch)
    assert obj.error is None and len(batches)==2
    assert all(kind=='state_batch' for kind,_ in batches)
    rows=[r for _,batch in batches for r in batch]
    for side in ('left','right'):
        sources=[r for r in rows if r['arm_id']==side]
        assert [r['feedback_sequence'] for r in sources]==list(range(1,14))
        assert [r['gripper_effort_nm'] for r in sources]==[(-i*.01 if side=='left' else i*.01) for i in range(1,14)]
        assert len(set(r['sample_monotonic_s'] for r in sources))==2
        assert len(set(r['feedback_received_monotonic_s'] for r in sources))==13


@pytest.mark.parametrize('option',['gap','stopped'])
def test_missing_or_stopped_feedback_is_a_capture_error(monkeypatch,option):
    obj,_=run_sampler(monkeypatch,**{option:True})
    assert obj.error is not None and obj.stop.is_set()


def test_normal_close_drains_last_received_sweeps_before_writer_exit(tmp_path):
    import threading
    import time
    import pyarrow.parquet as pq
    from test_roboeval import setup_file, FakeKinematics
    obj=roboeval.RoboEvalRecorder(tmp_path/'run',setup_file(tmp_path/'cal'),hz=100,
        continuous=True,identity={},kinematics_factory=FakeKinematics)
    obj.started=time.monotonic()
    stamp=obj.started
    events=[]
    feedback={s:[dict(sequence=7,received_monotonic_s=stamp,running=True,
        position=[0.]*6,velocity=[0.]*6,torque_nm=[0.]*6,gripper_reading=.5)] for s in ('left','right')}
    obj.robot=SimpleNamespace(stop_roboeval=lambda:events.append('cameras detached'),
        get_roboeval_feedback_history=lambda:feedback,
        stop_roboeval_feedback_history=lambda:events.append('history stopped'))
    obj.feedback_storage='all_received_sweeps'
    obj._history_previous={};obj._history_received={};obj._history_index=0
    obj.worker=threading.Thread(target=obj._write_loop);obj.worker.start()
    obj.close(complete=False)
    rows=pq.read_table(obj.meta/'states.parquet').to_pylist()
    assert len(rows)==2 and all(r['feedback_sequence']==7 for r in rows)
    assert not obj.worker.is_alive() and obj.error is None
    assert events==['cameras detached','history stopped']


def test_fatal_stop_never_accepts_a_late_history_batch(tmp_path):
    from test_roboeval import setup_file, FakeKinematics
    obj=roboeval.RoboEvalRecorder(tmp_path/'run',setup_file(tmp_path/'cal'),hz=100,
        continuous=True,identity={},kinematics_factory=FakeKinematics)
    obj.fail(RuntimeError('write failure'))
    obj._put('state_batch',[{'not':'accepted'}])
    assert obj.queue.empty()
