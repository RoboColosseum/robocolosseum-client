"""Preserve submitted results; optionally parse explicit English labels."""
from __future__ import annotations

import hashlib
import math
import re

FORMAT_VERSION = 'roboeval-english-feedback-v1'
LABELS = ('Task success', 'Completed stages', 'Main issue', 'Drop', 'Damage', 'Spill')


def parse_feedback(raw):
    if not isinstance(raw, str) or not raw.strip() or '\n' in raw or '\r' in raw:
        raise ValueError('Feedback must be one nonempty English template line')
    if re.search(r'[\u3400-\u9fff]', raw):
        raise ValueError('Use the fixed English feedback template')
    pattern = (r'\s*Task\s*=\s*(?P<task>.+?)\.\s*Policy\s+A\s*:\s*(?P<A>.+?)'
               r'\.\s*Policy\s+B\s*:\s*(?P<B>.+?)\.\s*Final preference\s*=\s*'
               r'(?P<preference>A|B|tie)\s*;\s*Comparison reason\s*=\s*(?P<reason>.+?)\s*')
    match = re.fullmatch(pattern, raw, re.IGNORECASE)
    if not match:
        raise ValueError('Feedback is missing Task, Policy A/B, Final preference or Comparison reason')
    for label in LABELS:
        if len(re.findall(r'\b' + re.escape(label) + r'\s*=', raw, re.I)) != 2:
            raise ValueError(f'Feedback requires exactly one {label} per policy')
    for label in ('Task', 'Final preference', 'Comparison reason'):
        if len(re.findall(r'\b' + re.escape(label) + r'\s*=', raw, re.I)) != 1:
            raise ValueError(f'Duplicate or missing {label}')
    if len(re.findall(r'\bPolicy\s+[AB]\s*:', raw, re.I)) != 2:
        raise ValueError('Duplicate policy sections')
    policies = {}
    for side in ('A', 'B'):
        parts = match[side].split(';')
        if len(parts) != len(LABELS):
            raise ValueError(f'Policy {side} requires the six fixed fields separated by semicolons')
        values = {}
        for label, part in zip(LABELS, parts):
            field = re.fullmatch(r'\s*' + re.escape(label) + r'\s*=\s*(.+?)\s*', part, re.I)
            if not field:
                raise ValueError(f'Policy {side}: missing, reordered or empty {label}')
            values[label] = field[1]
        success = values['Task success'].lower()
        if success not in ('yes', 'no', 'unknown'):
            raise ValueError(f'Policy {side}: Task success must be yes/no/unknown')
        events = {}
        for label in ('Drop', 'Damage', 'Spill'):
            value = values[label].lower()
            if value not in ('yes', 'no', 'unknown', 'na', 'not_applicable'):
                raise ValueError(f'Policy {side}: invalid {label} value')
            events[label.lower()] = 'not_applicable' if value == 'na' else value
        stages = values['Completed stages']
        status = stages.lower() if stages.lower() in ('none', 'unknown') else 'observed'
        steps = [step.strip() for step in stages.split('->')] if status == 'observed' else []
        if any(not step for step in steps):
            raise ValueError(f'Policy {side}: empty completed stage')
        policies[side] = dict(success={'yes': True, 'no': False, 'unknown': None}[success],
            success_status=success, completed_stages=steps, completed_stages_status=status,
            main_issue=values['Main issue'], **events)
    preference = match['preference'].upper() if match['preference'].lower() != 'tie' else 'tie'
    return dict(format_version=FORMAT_VERSION, raw_feedback=raw,
        raw_feedback_sha256=hashlib.sha256(raw.encode('utf-8')).hexdigest(),
        task_text=match['task'].strip(), policies=policies, preference=preference,
        comparison_reason=match['reason'].strip(), annotation_source='operator_english_feedback',
        validation_status='confirmed')


def validate_annotations(raw, preference, outcomes):
    parsed = parse_feedback(raw)
    if parsed['preference'] != preference:
        raise ValueError('Final preference conflicts with the selected page preference')
    if set(outcomes) != {'A', 'B'}:
        raise ValueError('Both page scores are required')
    for side in ('A', 'B'):
        outcome = outcomes[side]
        score = outcome.get('partial_success')
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError(f'Policy {side}: invalid original score')
        derived = score == 1.0
        if type(outcome.get('success')) is not bool or outcome['success'] != derived:
            raise ValueError(f'Policy {side}: original score/success rule differs')
        label = parsed['policies'][side]['success']
        if label is not None and label != derived:
            raise ValueError(f'Policy {side}: Task success conflicts with the original 100-point success rule')
    return parsed


def build_comparison(assignment, result):
    try:
        if 'b6' in result:
            raise ValueError('Structured B6 uses original scores without feedback inference')
        parsed = validate_annotations(result.get('feedback'), result.get('preference'), result.get('outcomes', {}))
    except ValueError as error:
        # Feedback was optional in the original workflow. Do not demand a template
        # or infer unreported stages/events from free text or numeric progress.
        raw = result.get('feedback', '')
        if not isinstance(raw, str):
            raise ValueError('Original feedback must be text')
        labels = {}
        for side in ('A', 'B'):
            original = result['outcomes'][side]
            success, score = original.get('success'), original.get('partial_success')
            if (type(success) is not bool or isinstance(score, bool)
                    or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError('Invalid original outcome')
            labels[side] = dict(success=success, success_status='yes' if success else 'no',
                completed_stages=[], completed_stages_status='unknown', main_issue='unknown',
                drop='unknown', damage='unknown', spill='unknown')
        parsed = dict(format_version='roboeval-original-feedback-v1', raw_feedback=raw,
            raw_feedback_sha256=hashlib.sha256(raw.encode('utf-8')).hexdigest(),
            task_text=assignment.get('task', {}).get('instruction', ''), policies=labels,
            preference=result.get('preference'), comparison_reason=None,
            annotation_source='original_page_result', validation_status='unstructured_or_conflicting',
            label_parse_error=str(error))
    runs = assignment.get('runs', [])
    if (len(runs) != 2 or {run.get('side') for run in runs} != {'A', 'B'}
            or len({run.get('id') for run in runs}) != 2 or any(not run.get('id') for run in runs)
            or not assignment.get('id')):
        raise ValueError('Actual assignment A/B run identities are missing or ambiguous')
    if 'b6' in result:
        from .roboeval_b6 import validate_b6
        labels = validate_b6(assignment, result['b6'], required=True)
        for run in runs:
            parsed['policies'][run['side']].update(labels[run['id']])
        parsed.update(b6=labels, annotation_source='operator_structured_b6',
                      validation_status='confirmed', format_version='roboeval-structured-b6-v1')
        parsed.pop('label_parse_error', None)
    mapping = {run['side']: dict(run_id=run['id'], model={key: run[key] for key in
               ('model', 'model_id', 'policy_id', 'model_url', 'revision') if key in run}) for run in runs}
    return dict(parsed, assignment_id=assignment['id'], comparison_id=assignment['id'], runs=mapping,
                original_outcomes=result['outcomes'], score_success_source=(
                    'original_open_track_score_equals_100' if parsed['format_version'] == FORMAT_VERSION
                    else 'original_submitted_outcomes'))


def outcome_for_run(comparison, run_id):
    sides = [side for side, run in comparison['runs'].items() if run['run_id'] == run_id]
    if len(sides) != 1:
        raise ValueError('Self run does not belong to this comparison')
    side = sides[0]
    return dict(schema_version=1, assignment_id=comparison['assignment_id'], self_run_id=run_id,
        self_side=side, comparison=comparison, episode_labels=comparison['policies'][side],
        original_outcome=comparison['original_outcomes'][side],
        score_0_100=comparison['original_outcomes'][side]['partial_success'] * 100)


def single_run_outcome(assignment, result, run_id):
    """Preserve the existing non-comparison scoring flow, without inventing events."""
    runs = [run for run in assignment['runs'] if run['id'] == run_id]
    if len(runs) != 1:
        raise ValueError('Single-run outcome identity is ambiguous')
    side = runs[0]['side']
    original = result['outcomes'][side]
    if type(original.get('success')) is not bool:
        raise ValueError('Missing original success annotation')
    score = original['partial_success']
    if isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Invalid original progress score')
    labels = dict(success=original['success'], success_status='yes' if original['success'] else 'no',
        completed_stages=[], completed_stages_status='unknown', main_issue='unknown',
        drop='unknown', damage='unknown', spill='unknown')
    return dict(schema_version=1, assignment_id=assignment['id'], self_run_id=run_id,
        self_side=side, comparison=None, episode_labels=labels, original_outcome=original,
        score_0_100=score*100, raw_feedback=result.get('feedback', ''),
        annotation_source='original_noncomparison_page_score')
