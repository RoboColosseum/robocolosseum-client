"""Actual SDK cache methods, without constructing a CAN interface."""
import ast
from collections import deque
from pathlib import Path
import threading
from types import SimpleNamespace, MethodType

import pytest


def chain_fixture():
    source = Path(__file__).parent/'fixtures/i2rt/dm_driver.py'
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name=='DMChainCanInterface')
    names = {'_feedback_snapshot_locked', '_retain_feedback_locked', 'start_feedback_history',
             'stop_feedback_history', 'drain_feedback_history', 'get_feedback_snapshot', '_joint_position_real_to_sim_idx'}
    scope = {'deque': deque}
    exec(compile(ast.Module(body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names],type_ignores=[]),str(source),'exec'),scope)
    chain = SimpleNamespace(state_lock=threading.Lock(), running=True, _feedback_sequence=1,
        _feedback_monotonic_s=10., _feedback_history=None, _feedback_history_dropped=0,
        state=[SimpleNamespace(id=i+1, velocity=0., torque=0.) for i in range(7)],
        motor_direction=[1.]*7, motor_offset=[0.]*7, absolute_positions=[0.]*7)
    for name in names:
        setattr(chain, name, MethodType(scope[name], chain))
    return chain


def publish(chain, seq):
    with chain.state_lock:
        chain._feedback_sequence = seq
        chain._feedback_monotonic_s = 10.+(seq-1)*.004
        chain.absolute_positions[0] = float(seq)
        chain._retain_feedback_locked()


def test_delayed_consumer_keeps_each_real_sweep_once_without_overwrite():
    chain = chain_fixture()
    chain.start_feedback_history()
    for seq in range(2, 102): publish(chain, seq)  # 400 ms consumer delay, no CAN operations.
    batch = chain.drain_feedback_history()
    assert batch['dropped']==0 and len(batch['rows'])==101
    assert [r['sequence'] for r in batch['rows']]==list(range(1,102))
    assert batch['rows'][1]['position'][0]==2.
    assert chain.drain_feedback_history()['rows']==[]
    assert chain.get_feedback_snapshot()['sequence']==101


def test_overflow_is_bounded_and_latched_not_silently_recovered():
    chain = chain_fixture(); chain.start_feedback_history(capacity=4)
    for seq in range(2, 8): publish(chain, seq)
    batch = chain.drain_feedback_history()
    assert len(batch['rows'])==4 and batch['dropped']==3
    assert chain.drain_feedback_history()['dropped']==3
    chain.stop_feedback_history()
    publish(chain, 8)
    assert chain._feedback_history is None
    with pytest.raises(RuntimeError, match='not active'): chain.drain_feedback_history()


def test_stop_and_duplicate_start_are_detected():
    chain = chain_fixture(); chain.start_feedback_history()
    with pytest.raises(RuntimeError, match='already active'): chain.start_feedback_history()
    chain.running = False
    assert not chain.drain_feedback_history()['running']
