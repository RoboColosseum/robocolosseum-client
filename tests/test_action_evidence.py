import json
from types import SimpleNamespace
import numpy as np
import pyarrow as pa
import pytest
from colosseum_client import colosseum_pb2 as pb
from colosseum_client.roboeval import RoboEvalRecorder, ACTION_SCHEMA

def recorder():
    obj=RoboEvalRecorder.__new__(RoboEvalRecorder)
    obj.setup=SimpleNamespace(arms={'left':{},'right':{}}); obj.check=lambda:None
    obj.rows=[]; obj._put=lambda stream,row:obj.rows.extend(row) if stream=='action_batch' else obj.rows.append(row)
    return obj

def send(obj,evidence,step=0,length=2):
    obj.plan(pb.ActionPlan(plan_id=step+1,start_step=step,action_evidence_json=json.dumps(evidence).encode()),
             action_space='joint_position',executable_length=length)

def evidence():
    return dict(version=1,source='dreamlabs',blocks=[dict(id='a',source='dream_reply',values=np.arange(70).reshape(5,14).tolist())],
        execution=dict(kind='linear_interpolation_after_gripper_clip',source_block='a',lower_rows=[0,1],upper_rows=[1,2],weights=[.25,.75]))

def test_raw_tail_and_reused_block_associations_survive_arrow():
    obj=recorder(); doc=evidence(); send(obj,doc)
    doc['blocks']=[]; send(obj,doc,step=2)
    rows=pa.Table.from_pylist(obj.rows,schema=ACTION_SCHEMA).to_pylist()
    raw=[r for r in rows if r['kind']=='raw_policy']; mappings=[r for r in rows if r['kind']=='execution_mapping']
    assert len(raw)==10 and len(mappings)==2
    assert raw[-1]['values']==list(range(63,70))
    assert raw[-1]['control_step'] is None
    assert json.loads(mappings[1]['evidence_json'])['execution']['weights']==[.25,.75]

@pytest.mark.parametrize('mutation',[lambda d:d['execution'].update(source_block='missing'),
    lambda d:d['execution'].update(upper_rows=[1,99]),lambda d:d['execution'].update(weights=[.2]),
    lambda d:d['execution'].update(weights=[float('nan'),.5])])
def test_bad_source_association_rejected(mutation):
    doc=evidence(); mutation(doc)
    with pytest.raises(ValueError):send(recorder(),doc)

def test_terminal_unused_block_is_preserved_without_execution():
    obj=recorder(); doc=evidence(); doc['execution']={'kind':'unused_terminal'}
    send(obj,doc,length=0)
    assert len([r for r in obj.rows if r['kind']=='raw_policy'])==10
    assert not [r for r in obj.rows if r['kind']=='sent_command']


def test_required_recording_cannot_silently_upload_a_legacy_package(tmp_path):
    from colosseum_client.roboeval_export import validate_uploaded_extension
    root=tmp_path/'lerobot';(root/'meta').mkdir(parents=True)
    (root/'meta/colosseum.json').write_text('{}')
    (tmp_path/'recording-context.json').write_text('{"roboeval_required":true}')
    with pytest.raises(ValueError,match='absent'):validate_uploaded_extension(root)

async def test_client_terminal_receive_is_separate_from_action_queue():
    import asyncio
    from colosseum_client.local_policy import LocalPolicyClient
    # Use an actual receiver with a fake wire, never a policy service or robot.
    client=LocalPolicyClient.__new__(LocalPolicyClient)
    client.run_id='run';client.incoming=asyncio.Queue();client.failure=None
    client._recording_drain=asyncio.get_running_loop().create_future()
    evidence={'version':1,'source':'dreamlabs','blocks':[],'execution':{'kind':'unused_terminal'}}
    async def frames():
        yield pb.RelayFrame(protocol_version=1,type=pb.LOCAL_CONTROL,session_id='run',
            payload=json.dumps({'recording_drain':evidence}).encode()).SerializeToString()
        await asyncio.Future()
    client.connection=frames();task=asyncio.create_task(client._receive())
    try:
        assert await asyncio.wait_for(client._recording_drain,1)==evidence
        assert client.incoming.empty() and client.failure is None
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
