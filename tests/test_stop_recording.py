"""Real evaluation loop/JSON writer with extracted GUI guard and fake I/O only."""
import ast
import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from colosseum_client import evaluation as e
from colosseum_client.robot_config import RobotClientConfig
from colosseum_client.robot_interface import RobotObservation


def guard(relative, prompts):
    path = Path(__file__).parents[1] / relative
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=='execute_unless_stopped')
    scope = dict(PROMPTS=prompts, ev=e, execute=e.execute_robot_action_async,
                 S=SimpleNamespace(last_obs=None), so101=lambda: False)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['execute_unless_stopped']


@pytest.mark.parametrize('entry',['scripts/eval_gui.py'])
@pytest.mark.parametrize('stop_at',['before_first','during_inference','after_command','never'])
def test_stop_does_not_invent_executed_target(tmp_path,monkeypatch,entry,stop_at):
    prompts=SimpleNamespace(finish_now=threading.Event())
    calls=[];finished=[];inferences=[]
    class Robot:
        joint_count=12;has_gripper=True;action_dim=14;action_space_name='joint_position'
        def __init__(self,config):pass
        def get_observation(self):
            return RobotObservation(images={},joints=np.zeros(12),gripper=np.zeros(2),cartesian_position=np.zeros(0))
        def execute(self,action):
            calls.append(np.asarray(action).tolist())
            if stop_at=='after_command':prompts.finish_now.set()
            return None  # Normal successful legacy execution must still be recorded.
        def close(self):finished.append('robot_closed')
    class Client:
        def __init__(self,*args,**kwargs):pass
        async def connect(self,**kwargs):return SimpleNamespace(action_spaces=['joint_position'])
        async def infer(self,*args,**kwargs):
            inferences.append(len(inferences))
            if stop_at=='before_first' or stop_at=='during_inference' and len(inferences)==2:
                prompts.finish_now.set()
            return SimpleNamespace(done=False)
        async def close(self):finished.append('client_closed')
    monkeypatch.setattr(e,'ColosseumClient',Client)
    monkeypatch.setattr(e,'protobuf_observation',lambda *a,**kw:object())
    monkeypatch.setattr(e,'action_chunk',lambda *a,**kw:np.full((1,14),len(inferences),dtype=np.float32))
    monkeypatch.setattr(e,'execute_robot_action_async',guard(entry,prompts))
    monkeypatch.setattr(e,'sys',SimpleNamespace(stdin=SimpleNamespace(isatty=lambda:False)))
    e.INTERRUPT.clear()
    cfg=RobotClientConfig(url='ws://offline.invalid',token='test',cameras={},control_hz=1000,roboeval_state_hz=100,recording=True,test=True)
    assignment={'task':{'instruction':'fake stop test','cameras':[],'max_steps':3}}
    api=SimpleNamespace(request=lambda *args:finished.append('finished'))
    asyncio.run(e.run_trial(cfg,assignment,{'id':'fake'},tmp_path,api,robot_factory=Robot))
    rows=[json.loads(x) for x in (tmp_path/'frames.jsonl').read_text().splitlines()]
    targets=[x for x in rows if x.get('action_executed_monotonic_s') is not None]
    expected={'before_first':0,'during_inference':1,'after_command':1,'never':3}[stop_at]
    assert len(calls)==len(targets)==expected
    assert [r['action'] for r in targets]==calls
    assert [r['frame'] for r in targets]==list(range(expected))
    assert len(rows)==expected+1 and rows[-1]['action'] is None
    assert 'action_executed_monotonic_s' not in rows[-1]
    assert json.loads((tmp_path/'recording-status.json').read_text())['complete']
    assert set(finished)=={'finished','robot_closed','client_closed'}
