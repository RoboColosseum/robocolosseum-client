import copy
import json

import pytest

from colosseum_client import evaluation
from colosseum_client.roboeval_feedback import build_comparison


def fixture(tmp_path):
    assignment = {'id': 'ev_models', 'track': 'open', 'runs': [
        {'id': 'run_b', 'side': 'B'}, {'id': 'run_a', 'side': 'A'}]}
    result = {'outcomes': {'A': {'success': False, 'partial_success': .4},
                           'B': {'success': False, 'partial_success': .2}},
              'preference': 'A', 'feedback': 'Original optional free text'}
    for side in ['a', 'b']:
        path = tmp_path / ('run_' + side)
        path.mkdir()
        (path / 'policy.json').write_text(json.dumps({'run_id': 'run_' + side,
            'model': {'url': 'local/' + side, 'revision': side * 40,
                      'subfolder': 'checkpoint', 'action_dim': 14}}))
    return assignment, result


def test_existing_export_uses_both_receipts_and_preserves_original_result(tmp_path, monkeypatch):
    assignment, result = fixture(tmp_path)
    before = copy.deepcopy((assignment, result))
    outputs = []
    monkeypatch.setattr(evaluation, 'export_recorded_run', lambda *args: outputs.append(args[-1]))
    for run in assignment['runs']:
        evaluation.export_expanded_run(None, assignment, run, tmp_path, result)
    assert (assignment, result) == before
    for output in outputs:
        comparison = output['comparison']
        for side in ['A', 'B']:
            identity = comparison['runs'][side]
            assert identity['run_id'] == 'run_' + side.lower()
            assert identity['model']['model']['revision'] == side.lower() * 40
        assert comparison['raw_feedback'] == result['feedback']
        assert output['score_0_100'] == result['outcomes'][output['self_side']]['partial_success'] * 100
        # Exercise the same identity reconstruction used by validate_shared.
        reconstructed = {**assignment, 'runs': [dict(id=item['run_id'], side=side,
            **item['model']) for side, item in comparison['runs'].items()]}
        assert build_comparison(reconstructed, result) == comparison


@pytest.mark.parametrize('bad_policy', [
    {'run_id': 'run_b', 'model': {'revision': 'wrong-run'}},
    {'run_id': 'run_a', 'model': {}},
    {'run_id': 'run_a', 'model': None},
    [],
])
def test_mismatched_or_invalid_receipt_cannot_silently_label_export(tmp_path, bad_policy):
    assignment, result = fixture(tmp_path)
    (tmp_path / 'run_a/policy.json').write_text(json.dumps(bad_policy))
    with pytest.raises(ValueError, match='Recorded policy identity'):
        evaluation.build_recorded_comparison(assignment, result, tmp_path)


def test_absent_receipt_preserves_assignment_identity_for_remote_policy(tmp_path):
    assignment, result = fixture(tmp_path)
    (tmp_path / 'run_a/policy.json').unlink()
    assignment['runs'][1]['model_id'] = 'remote-a'
    comparison = evaluation.build_recorded_comparison(assignment, result, tmp_path)
    assert comparison['runs']['A']['model'] == {'model_id': 'remote-a'}
    assert comparison['runs']['B']['model']['model']['url'] == 'local/b'
