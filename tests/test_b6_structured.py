import importlib.util
import json
from pathlib import Path
import pytest
from colosseum_client.roboeval_b6 import validate_b6
from colosseum_client.roboeval_feedback import build_comparison, outcome_for_run

A = {'id':'ev','runs':[{'id':'right-id','side':'B'},{'id':'left-id','side':'A'}]}
B = {'left-id':dict(drop='yes',damage='no',spill='unknown'),
     'right-id':dict(drop='no',damage='not_applicable',spill='yes')}

def test_required_and_draft():
    assert validate_b6(A,None)=={'right-id':{},'left-id':{}}
    for rid in B:
        for field in B[rid]:
            missing={k:dict(v) for k,v in B.items()};del missing[rid][field]
            with pytest.raises(ValueError):validate_b6(A,missing,required=True)
    for bad in [{'alien':{}},{'left-id':{'drop':False}},{'left-id':{'other':'yes'}}]:
        with pytest.raises(ValueError):validate_b6(A,bad)

def test_export_identity_and_raw_score():
    result=dict(feedback='free text',preference='A',outcomes={s:dict(success=False,partial_success=.4) for s in 'AB'},b6=B)
    c=build_comparison(A,result)
    assert outcome_for_run(c,'left-id')['episode_labels']['drop']=='yes'
    assert outcome_for_run(c,'right-id')['episode_labels']['spill']=='yes'
    assert outcome_for_run(c,'left-id')['score_0_100']==40
    assert c['policies']['A']['completed_stages']==[]
    assert build_comparison(A,{k:v for k,v in result.items() if k!='b6'})['format_version']=='roboeval-original-feedback-v1'

def test_save_and_locked_record(tmp_path):
    spec=importlib.util.spec_from_file_location('b6_review',Path(__file__).parents[1]/'scripts/review_evals.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ROOT=tmp_path
    folder=tmp_path/'ev';folder.mkdir();p=folder/'manifest.json'
    p.write_text(json.dumps({'assignment':A}))
    form={'score_A':['40'],'score_B':['50'],'preference':['A'],'feedback':['original']}
    with pytest.raises(ValueError):m.save('ev',form,b6={},require_b6=True)
    assert 'b6' not in json.loads(p.read_text())
    m.save('ev',form,b6=B,require_b6=True)
    d=json.loads(p.read_text());assert d['b6']==B;assert set(d['result'])=={'outcomes','preference','feedback'}
    d['submitted']=True;p.write_text(json.dumps(d));before=p.read_bytes()
    m.save('ev',form,b6={});assert p.read_bytes()==before


def test_runtime_limits_are_observed_not_hardcoded(monkeypatch):
    import ast, sys, types
    import numpy as np
    source=Path(__file__).parents[1]/'src/colosseum_client/robots/yam.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='roboeval_limits')
    constants=types.SimpleNamespace(TORQUE_MIN=-28,TORQUE_MAX=28,VELOCITY_MIN=-10,VELOCITY_MAX=10)
    monkeypatch.setitem(sys.modules,'i2rt.motor_drivers.utils',types.SimpleNamespace(MotorType=types.SimpleNamespace(get_motor_constants=lambda _:constants)))
    ns={'np':np};exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),ns)
    arm=types.SimpleNamespace(_clip_motor_torque=float('inf'),_joint_limits=None,_limit_gripper_force=47,
        _gripper_limits=[0,1],motor_chain=types.SimpleNamespace(motor_list=[(1,'DM4340')]))
    robot=types.SimpleNamespace(low=np.zeros(12),high=np.ones(12),max_step=np.full(12,.05),joint_limit_margin=0,
        joint_step_mode='clip_commanded',joint_limit_mode='clip',roboeval_arm_ids=['left','right'],arms=[arm,arm])
    limits=ns['roboeval_limits'](robot)['left']
    assert limits['gripper_force_limit_n']==47
    assert limits['sdk_limits']['feedforward_torque_clip_abs_nm'] is None
    assert limits['independent_firmware_limits_status']=='not_read'
    assert limits['nominal_control_hz']==30


def test_structured_b6_ninefile_roundtrip(tmp_path):
    from test_roboeval_shared import assignment, result, make_recording
    from test_roboeval import setup_file
    from colosseum_client.lerobot_export import export_trial
    from colosseum_client.roboeval_export import validate_uploaded_extension
    a=assignment();res=result()
    res['b6']={r['id']:dict(drop='unknown',damage='no',spill='not_applicable') for r in a['runs']}
    comp=build_comparison(a,res)
    calibration=setup_file(tmp_path/'cal')
    for r in reversed(a['runs']):
        run=tmp_path/r['id'];cameras=make_recording(run,calibration,r['id'])
        own=outcome_for_run(comp,r['id'])
        out=export_trial(run,cameras,fps=30,robot_type='yam',task='orange',metadata={
            'assignment_id':a['id'],'run_id':r['id'],'roboeval_outcome':own})
        validate_uploaded_extension(out)
        marker=json.loads((out/'meta/colosseum.json').read_text())
        assert marker['roboeval_layout']['documents']['outcome']==own
        assert own['episode_labels']['drop']=='unknown'
        assert len([p for p in out.rglob('*') if p.is_file()])==9
