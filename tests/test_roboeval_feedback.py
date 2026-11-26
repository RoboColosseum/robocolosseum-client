"""Exercise the actual passive SDK method without importing CAN/GPU dependencies."""
import ast
from pathlib import Path
import threading
from types import SimpleNamespace
import numpy as np
import pytest


def yam_feedback_method(name='get_roboeval_feedback'):
    source = Path(__file__).parents[1] / 'src/colosseum_client/robots/yam.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    method = next(n for cls in tree.body if isinstance(cls, ast.ClassDef)
                  for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == name)
    namespace = {'np': np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[method.name]


def test_passive_yam_feedback_does_not_wait_for_control_lock():
    def remap(q):
        value = np.array(q, copy=True)
        value[6] -= 1.0
        return value
    mapper = SimpleNamespace(to_command_joint_pos_space=remap)

    class ControlLock:
        def __enter__(self):
            raise AssertionError('Recording must not acquire the control loop lock')
        def __exit__(self, *args): pass

    snapshots = [dict(sequence=17+i, received_monotonic_s=100.+i/1000,
                      running=True, motor_ids=list(range(1, 8)),
                      position=[.1, -.2, .3, -.4, .5, -.6, 1.25],
                      velocity=list(range(7)), torque_nm=list(range(10, 17))) for i in range(2)]
    calls = []
    def snapshot(i):
        calls.append(i)
        return snapshots[i]
    robot = SimpleNamespace(roboeval_arm_ids=('left','right'), arms=[
        SimpleNamespace(_state_lock=ControlLock(), remapper=mapper,
                        motor_chain=SimpleNamespace(get_feedback_snapshot=lambda i=i: snapshot(i)))
        for i in range(2)])
    feedback = yam_feedback_method()(robot)
    assert calls == [0, 1]
    for i, side in enumerate(('left', 'right')):
        result = feedback[side]
        assert result['sequence'] == 17+i
        assert result['received_monotonic_s'] == 100.+i/1000
        assert result['running'] is True
        np.testing.assert_allclose(result['position'], snapshots[i]['position'][:6])
        assert result['velocity'] == list(range(6))
        assert result['torque_nm'] == list(range(10, 16))
        assert result['gripper_reading'] == .25
        assert result['gripper_effort_nm'] == 16
        assert snapshots[i]['position'][6] == 1.25
    snapshots[0]['motor_ids'].reverse()
    with pytest.raises(ValueError, match='motor ordering'):
        yam_feedback_method()(robot)


def test_snapshot_uses_real_receipt_metadata_and_coherent_signed_feedback():
    source = Path(__file__).parent / 'fixtures/i2rt/dm_driver.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DMChainCanInterface')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ('get_feedback_snapshot', '_feedback_snapshot_locked', '_joint_position_real_to_sim_idx')]
    namespace = {}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), namespace)
    chain = SimpleNamespace(state_lock=threading.Lock(), running=True,
        _feedback_sequence=17, _feedback_monotonic_s=100.,
        state=[SimpleNamespace(id=1, velocity=2., torque=3.), SimpleNamespace(id=2, velocity=4., torque=5.)],
        motor_direction=[1, -1], motor_offset=[.1, .2], absolute_positions=[.5, .8])
    chain._joint_position_real_to_sim_idx = lambda q, i: namespace['_joint_position_real_to_sim_idx'](chain, q, i)
    chain._feedback_snapshot_locked = lambda: namespace['_feedback_snapshot_locked'](chain)
    first = namespace['get_feedback_snapshot'](chain)
    second = namespace['get_feedback_snapshot'](chain)
    assert first['sequence'] == second['sequence'] == 17
    assert first['received_monotonic_s'] == second['received_monotonic_s'] == 100.
    assert first['velocity'] == [2, -4] and first['torque_nm'] == [3, -5]
    np.testing.assert_allclose(first['position'], [.4, -.6])
    assert first['motor_ids'] == [1, 2]
    first['position'][0] = 999
    assert second['position'][0] != 999 and chain.absolute_positions == [.5, .8]
    chain._feedback_monotonic_s = None
    with pytest.raises(RuntimeError, match='No complete'):
        namespace['get_feedback_snapshot'](chain)


def test_history_retains_signed_gripper_effort_separately():
    snapshot=dict(sequence=1, received_monotonic_s=1.,running=True,motor_ids=list(range(1,8)),
        position=[0.]*7,velocity=[0.]*7,torque_nm=[.1]*6+[-.7])
    chain=SimpleNamespace(drain_feedback_history=lambda:dict(rows=[snapshot],dropped=0,running=True))
    robot=SimpleNamespace(roboeval_arm_ids=('left',),arms=[SimpleNamespace(motor_chain=chain,
        remapper=SimpleNamespace(to_command_joint_pos_space=lambda q:q))])
    row=yam_feedback_method('get_roboeval_feedback_history')(robot)['left'][0]
    assert row['gripper_effort_nm']==-.7 and row['torque_nm']==[.1]*6
    assert snapshot['torque_nm']==[.1]*6+[-.7]
