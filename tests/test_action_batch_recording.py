"""Model bursts must not exhaust the queue used by continuous camera/state data."""
import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from colosseum_client import colosseum_pb2 as pb
from colosseum_client.roboeval import RoboEvalRecorder, RAW_SOURCE
from colosseum_client.tensors import tensor_from_numpy
from test_roboeval import setup_file, FakeKinematics


def make_recorder(tmp_path, **kwargs):
    return RoboEvalRecorder(tmp_path/'run', setup_file(tmp_path/'setup'), hz=100,
        identity={}, continuous=True, kinematics_factory=FakeKinematics, **kwargs)


def make_plan(length, evidence):
    values=np.arange(length*14,dtype=float).reshape(length,14)/1000
    if evidence:
        doc=dict(version=1,source='dreamlabs',blocks=[
            dict(id='a',values=values.tolist()),dict(id='b',values=values.tolist())],
            execution=dict(kind='prefix',source_block='b',rows=[0,1,2]))
        return pb.ActionPlan(plan_id=1,start_step=0,action_evidence_json=json.dumps(doc).encode()),values
    return pb.ActionPlan(plan_id=1,start_step=0,raw_actions=tensor_from_numpy(values),
        raw_action_source=RAW_SOURCE,raw_action_space='joint_position'),values


@pytest.mark.parametrize('length',[48,96,4096])
@pytest.mark.parametrize('evidence',[False,True])
def test_burst_survives_stalled_writer_and_preserves_every_row(tmp_path,length,evidence):
    rec=make_recorder(tmp_path)
    plan,values=make_plan(length,evidence)
    # Writer intentionally has not consumed anything, as during one slow encode.
    rec.plan(plan,action_space='joint_position',executable_length=3)
    rec.select(1,0,0)
    rec.sent('left',values[0,:7]);rec.sent('right',values[0,7:])
    rec.frame('left',dict(source_frame_number=73))
    assert rec.error is None and not rec.stop.is_set()
    rec._write_video=lambda side,frame:dict(camera_id=side,source_frame_number=frame['source_frame_number'])
    rec.queue.put(None);rec._write_loop()
    rows=pq.read_table(rec.meta/'actions.parquet').to_pylist()
    raw=[r for r in rows if r['kind']=='raw_policy']
    assert len(raw)==length*2*(2 if evidence else 1)
    blocks=['a','b'] if evidence else [None]
    for block in blocks:
        for arm,offset in [('left',0),('right',7)]:
            selected=[r for r in raw if r['arm_id']==arm and r['source_block_id']==block]
            assert [r['chunk_index'] for r in selected]==list(range(length))
            np.testing.assert_allclose([r['values'] for r in selected],values[:,offset:offset+7])
    assert len([r for r in rows if r['kind']=='execution_mapping'])==int(evidence)
    assert len([r for r in rows if r['kind']=='sent_command'])==2
    assert pq.read_table(rec.meta/'frames.parquet')['source_frame_number'].to_pylist()==[73]


def test_invalid_mapping_does_not_enqueue_partial_raw_evidence(tmp_path):
    rec=make_recorder(tmp_path);plan,_=make_plan(48,True)
    doc=json.loads(plan.action_evidence_json);doc['execution']['source_block']='missing'
    plan.action_evidence_json=json.dumps(doc).encode()
    with pytest.raises(ValueError):rec.plan(plan,action_space='joint_position',executable_length=3)
    assert rec.queue.empty()
    assert not getattr(rec,'_source_blocks',{})


def test_full_queue_rejects_whole_plan_without_blocking_or_partial_rows(tmp_path):
    rec=make_recorder(tmp_path,queue_size=1)
    rec.frame('left',dict(source_frame_number=1))
    plan,_=make_plan(96,True)
    with pytest.raises(RuntimeError):rec.plan(plan,action_space='joint_position',executable_length=3)
    assert 'queue full' in str(rec.error)
    assert rec.queue.get_nowait()[0]=='frames' and rec.queue.empty()
    assert not getattr(rec,'_source_blocks',{})


def test_disk_failure_is_not_certified_as_complete(tmp_path,monkeypatch):
    rec=make_recorder(tmp_path);plan,_=make_plan(96,True)
    rec.plan(plan,action_space='joint_position',executable_length=3)
    class BrokenWriter:
        def __init__(self,*args,**kwargs):pass
        def write_table(self,table):raise OSError('simulated disk full')
        def close(self):pass
    monkeypatch.setattr(pq,'ParquetWriter',BrokenWriter)
    rec._write_loop()
    assert rec.stop.is_set() and 'disk full' in str(rec.error)
    with pytest.raises(RuntimeError):rec.check()
