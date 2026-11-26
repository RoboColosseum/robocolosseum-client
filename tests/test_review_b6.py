import importlib.util
import json
from pathlib import Path
import pytest

@pytest.fixture
def review(tmp_path):
    path = Path(__file__).parents[1] / 'scripts/review_evals.py'
    spec = importlib.util.spec_from_file_location('review_b6_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    folder = tmp_path / 'ev_test'
    folder.mkdir()
    document = {'assignment': {'id': 'ev_test', 'task': {'instruction': 'test'}, 'runs': [
        {'id': 'run_b', 'side': 'B'}, {'id': 'run_a', 'side': 'A'}]}, 'outcomes': {}}
    (folder / 'manifest.json').write_text(json.dumps(document))
    return module


def form():
    values = {'score_A': ['100'], 'score_B': ['0'], 'preference': ['A'], 'feedback': ['test']}
    for run in ('run_a', 'run_b'):
        for field in ('drop', 'damage', 'spill'):
            values[f'b6:{run}:{field}'] = ['no' if run == 'run_a' else 'unknown']
    return values


def test_save_uses_run_identity_and_renders_saved_selection(review):
    review.save('ev_test', form(), require_b6=True)
    doc = review.load('ev_test')
    assert doc['b6']['run_a']['drop'] == 'no'
    assert doc['b6']['run_b']['drop'] == 'unknown'
    page = review.eval_page('ev_test').decode('utf-8')
    assert 'name="b6:run_a:drop"' in page
    assert 'value="unknown" selected' in page
    assert 'b6' not in doc['result']  # Router result contract is unchanged.


def test_invalid_or_missing_required_event_does_not_write(review):
    before = (review.ROOT / 'ev_test/manifest.json').read_bytes()
    data = form()
    data['b6:run_a:drop'] = ['']
    with pytest.raises(ValueError, match='Select Drop'):
        review.save('ev_test', data, require_b6=True)
    data = form(); data['b6:unknown:drop'] = ['yes']
    with pytest.raises(ValueError, match='unknown run'):
        review.save('ev_test', data)
    assert (review.ROOT / 'ev_test/manifest.json').read_bytes() == before


def test_requirement_is_based_on_recording_context(review):
    assert not review.requires_b6('ev_test')
    p = review.ROOT / 'ev_test/run_a/recording-context.json'
    p.parent.mkdir()
    p.write_text(json.dumps({'roboeval_shared_required': True}))
    assert review.requires_b6('ev_test')


def test_closed_evaluation_is_unchanged(review):
    p = review.ROOT / 'ev_test/manifest.json'
    doc = review.load('ev_test'); doc['submitted'] = True
    p.write_text(json.dumps(doc)); before = p.read_bytes()
    review.save('ev_test', form(), require_b6=True)
    assert p.read_bytes() == before


def test_main_gui_uses_existing_review_save(review, monkeypatch):
    import ast
    from fastapi import HTTPException
    path = Path(__file__).parents[1] / 'scripts/eval_gui.py'
    node = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n, ast.FunctionDef) and n.name == 'act_eval')
    node.decorator_list = []
    scope = {'rv': review, 'HTTPException': HTTPException, 'known': lambda _: None}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    labels = {'run_a': {'drop': 'yes', 'damage': 'no', 'spill': 'unknown'},
              'run_b': {'drop': 'no', 'damage': 'no', 'spill': 'not_applicable'}}
    response = scope['act_eval']('ev_test', {'action': 'save', 'score_A': 80,
        'score_B': 20, 'preference': 'A', 'feedback': 'Observed', 'b6': labels})
    assert response['msg'] == 'Saved on this computer.'
    assert review.load('ev_test')['b6'] == labels
    before = (review.ROOT / 'ev_test/manifest.json').read_bytes()
    with pytest.raises(HTTPException) as exc:
        scope['act_eval']('ev_test', {'action': 'save', 'b6': {'unknown': {}}})
    assert exc.value.status_code == 400
    assert (review.ROOT / 'ev_test/manifest.json').read_bytes() == before
