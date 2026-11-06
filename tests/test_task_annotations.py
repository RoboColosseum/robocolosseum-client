import builtins
import importlib
import json
from pathlib import Path

import pytest

from colosseum_client import evaluation as ev
from colosseum_client.robot_config import RobotClientConfig
from colosseum_client.task_annotations import difficulty, set_difficulty


@pytest.fixture
def review(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    module = importlib.import_module('review_evals')
    monkeypatch.setattr(module, 'ROOT', tmp_path, raising=False)
    monkeypatch.setattr(module, 'JOBS', {})
    folder = tmp_path / 'ev_test'
    folder.mkdir()
    document = {'assignment': {'id': folder.name, 'task': {'instruction': 'pick'},
                              'runs': [{'id': 'run_A', 'side': 'A'}, {'id': 'run_B', 'side': 'B'}]},
                'outcomes': {}, 'review': 'pending'}
    (folder / 'manifest.json').write_text(json.dumps(document))
    return module, folder


@pytest.mark.parametrize('value', [True, False, 0, 11, -1, 1.5, '1.5', '1/10', 'NaN', '３', [], {}])
def test_invalid_difficulty_does_not_modify_manifest(review, value):
    module, folder = review
    original = (folder / 'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='integer'):
        module.save(folder.name, {'task_difficulty': [value]})
    assert (folder / 'manifest.json').read_bytes() == original


def test_draft_and_submission_require_one_shared_rating(review, monkeypatch):
    module, folder = review
    form = {'score_A': ['25'], 'score_B': ['75'], 'preference': ['B'], 'feedback': ['observed']}
    module.save(folder.name, form)
    with pytest.raises(ValueError, match='required'):
        module.save(folder.name, form, require_difficulty=True)
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: pytest.fail('No submit without difficulty'))
    with pytest.raises(ValueError, match='required'):
        module.router(folder.name, 'submit')
    module.save(folder.name, {**form, 'task_difficulty': ['3']}, require_difficulty=True)
    saved = module.load(folder.name)
    assert saved['task_difficulty'] == 3 and saved['task_difficulty_scale'] == 10
    assert 'task_difficulty' not in saved['result']  # Dataset metadata, not a Router result-schema change.
    module.save(folder.name, form)  # An older caller omitting the field preserves the saved value.
    assert module.load(folder.name)['task_difficulty'] == 3
    assert module.eval_page(folder.name).count(b'name=task_difficulty') == 1


@pytest.mark.parametrize('existing', [None, 3])
def test_exported_rating_is_frozen_including_legacy_missing_value(review, existing):
    module, folder = review
    marker = folder / 'run_A/lerobot/meta/colosseum.json'
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({} if existing is None else {'task_difficulty': existing}))
    document = module.load(folder.name)
    set_difficulty(folder, document, existing, required=True)
    assert document.get('task_difficulty') == existing
    with pytest.raises(ValueError, match='cannot change'):
        set_difficulty(folder, document, 4)
    if existing is not None:
        with pytest.raises(ValueError, match='cannot change'):
            set_difficulty(folder, document, None)
    assert json.loads(marker.read_text()).get('task_difficulty') == existing


def test_cli_reprompts_until_integer_rating(monkeypatch):
    answers = iter(['', '1/10', '0', '11', '1.5', '10'])
    monkeypatch.setattr(ev, 'ask', lambda prompt: next(answers))
    assert ev.prompt_task_difficulty() == 10
    assert difficulty('1') == 1


def test_gui_api_saves_and_restores_rating_and_rejects_boolean(review, monkeypatch):
    module, folder = review
    gui = importlib.import_module('eval_gui')
    monkeypatch.setattr(gui, 'rv', module)
    body = {'score_A': '25', 'score_B': '75', 'preference': 'B', 'feedback': 'observed',
            'task_difficulty': 3}
    gui.act_eval(folder.name, body)
    assert gui.eval_detail(folder.name)['task_difficulty'] == 3
    with pytest.raises(gui.HTTPException) as error:
        gui.act_eval(folder.name, {**body, 'task_difficulty': True})
    assert error.value.status_code == 400
    sent = []
    monkeypatch.setattr(module, 'router', lambda eid, action: sent.append(module.load(eid)) or 'submitted')
    gui.act_eval(folder.name, {**body, 'action': 'submit'})
    assert len(sent) == 1 and sent[0]['task_difficulty'] == 3


def test_record_only_defers_difficulty_until_review(review, monkeypatch):
    module, folder = review
    recording = importlib.import_module('record_evals')
    from colosseum_client import lerobot_export
    assignment = module.load(folder.name)['assignment']
    assignment.update(state='pending', track='open')
    assignment['task']['cameras'] = []
    for run in assignment['runs']:
        run['state'] = 'assigned'
    class API:
        def __init__(self, config): pass
        def close(self): pass
        def request(self, method, path, body=None):
            assert method == 'GET' and path == '/assignments/ev_test'
            return assignment
    async def trial(config, assignment, run, path, api, **kwargs):
        path.mkdir()
        (path / 'frames.jsonl').write_text('{}\n')
        (path / 'recording-context.json').write_text(json.dumps({
            'fps': 15, 'robot_type': 'franka', 'test': False,
            'action_space': 'joint_position', 'execution_enabled': False}))
        run['state'] = 'finished'
    monkeypatch.setattr(ev, 'EvalAPI', API)
    monkeypatch.setattr(ev, 'run_trial', trial)
    for name, default in [('progress', None), ('prompt_task_difficulty', None),
                          ('input', builtins.input), ('print', builtins.print)]:
        monkeypatch.setattr(ev, name, getattr(ev, name, default), raising=False)
    monkeypatch.setattr(lerobot_export, 'export_trial', lerobot_export.export_trial)
    prompts = []
    config = RobotClientConfig(url='ws://localhost', token='test', cameras={},
                               instruction='pick', scene='desk', evaluation_dir=str(folder.parent))
    assert recording.record(config, resume=folder.name, ask=lambda p: prompts.append(p) or '') == folder.name
    assert len(prompts) == 2 and all('restore the scene' in p for p in prompts)
    assert module.load(folder.name).get('task_difficulty') is None
    assert not list(folder.glob('*/lerobot'))


@pytest.mark.parametrize('rating', [None, 7])
def test_resume_existing_exports_preserves_metadata_without_reprompt(review, monkeypatch, rating):
    module, folder = review
    import numpy as np
    from colosseum_client.droid_robot import RobotObservation
    from colosseum_client.lerobot_export import export_trial
    document = module.load(folder.name)
    assignment = document['assignment']
    assignment.update(state='pending', track='open')
    assignment['task']['cameras'] = ['head_image']
    config = RobotClientConfig(url='ws://localhost', token='test', cameras={'head_image': 'test'},
                               instruction='pick', scene='desk', evaluation_dir=str(folder.parent),
                               skip_upload=True)
    before = {}
    for run in assignment['runs']:
        run['state'] = 'finished'
        path = folder / run['id']
        recorder = ev.TrialRecorder(path, ['head_image'])
        observation = RobotObservation({'head_image': np.zeros((16, 16, 3), dtype=np.uint8)},
                                       np.zeros(7), np.zeros(1), np.zeros(6))
        recorder.add(observation, np.zeros(8))
        recorder.close()
        context = {'robot_type': 'franka', 'fps': 15, 'test': False,
                   'action_space': 'joint_position', 'execution_enabled': False}
        (path / 'recording-context.json').write_text(json.dumps(context))
        outcome = {'success': True, 'partial_success': 1.0}
        document['outcomes'][run['side']] = outcome
        metadata = {'assignment_id': assignment['id'], 'run_id': run['id'], 'side': run['side'],
                    'track': 'open', 'test': False, 'action_space': 'joint_position',
                    'native_control_mode': 'joint_pos', 'action_recording': 'policy_targets',
                    'execution_enabled': False, 'institution': config.institution,
                    'scene': config.scene, 'evaluator': config.evaluator, 'outcome': outcome}
        if rating is not None:
            metadata.update(task_difficulty=rating, task_difficulty_scale=10)
        exported = export_trial(path, ['head_image'], fps=15, robot_type='franka', task='pick', metadata=metadata)
        marker = exported / 'meta/colosseum.json'
        before[marker] = marker.read_bytes()
    if rating is not None:
        document.update(task_difficulty=rating, task_difficulty_scale=10)
    document['result'] = {'outcomes': document['outcomes'], 'preference': 'tie', 'feedback': 'saved'}
    (folder / 'manifest.json').write_text(json.dumps(document))
    submitted = []
    class API:
        def __init__(self, config): pass
        def close(self): pass
        def request(self, method, path, body=None):
            if method == 'GET':
                assert path == '/assignments/ev_test'
                return assignment
            assert method == 'POST' and path == '/assignments/ev_test/result'
            submitted.append(body)
            return {'state': 'completed'}
        def upload_dataset(self, *args):
            pytest.fail('skip_upload must remain effective')
    monkeypatch.setattr(ev, 'EvalAPI', API)
    monkeypatch.setattr(ev, 'run_trial', lambda *a, **k: pytest.fail('Do not rerun finished trials'))
    def answer(prompt):
        assert prompt.startswith('Continue with the next evaluation')
        return 'n'
    monkeypatch.setattr(ev, 'ask', lambda prompt, **kwargs: answer(prompt))
    ev.run_evaluation(config, resume=folder.name)
    assert submitted == [document['result']]
    assert all(marker.read_bytes() == raw for marker, raw in before.items())
    assert module.load(folder.name).get('task_difficulty') == rating
