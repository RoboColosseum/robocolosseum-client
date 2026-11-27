"""Shared host-clock events and final per-run metadata, outside the control loop."""
from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import time
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

EVENT_SCHEMA = pa.schema([
    ('event_index', pa.int64()), ('event_id', pa.string()), ('kind', pa.string()), ('source', pa.string()),
    ('assignment_id', pa.string()), ('run_id', pa.string()),
    ('sample_monotonic_s', pa.float64()), ('observation_monotonic_s', pa.float64()),
    ('action_ready_monotonic_s', pa.float64()), ('call_id', pa.string()),
    ('plan_id', pa.uint64()), ('status', pa.string()), ('details_json', pa.string())])


@lru_cache(maxsize=1)
def host_clock_id():
    try:
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        return None  # Offline/non-Linux fixtures cannot certify a reboot identity.


def record_event(run_path, assignment_id, run_id, kind, *, source='client', stamp=None, **details):
    """One append per event; a partial trailing record makes finalization fail visibly."""
    root = Path(run_path)
    root.mkdir(parents=True, exist_ok=True)
    row = dict(event_id=uuid.uuid4().hex, kind=kind, source=source,
        assignment_id=assignment_id, run_id=run_id,
        clock_id=host_clock_id(),
        sample_monotonic_s=time.monotonic() if stamp is None else stamp, details=details)
    data = (json.dumps(row, allow_nan=False, sort_keys=True) + '\n').encode('utf-8')
    fd = os.open(root / 'shared-events.jsonl', os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        if os.write(fd, data) != len(data):
            raise OSError('Incomplete shared-event append')
    finally:
        os.close(fd)
    return row


def shared_payload(run_path, outcome):
    if (Path(run_path) / 'shared-events-error.json').exists():
        raise ValueError('Shared event recording previously failed; inspect preserved partial data')
    path = Path(run_path) / 'shared-events.jsonl'
    raw = path.read_bytes()
    if not raw.endswith(b'\n'):
        raise ValueError('Incomplete shared event log')
    events = [json.loads(line) for line in raw.splitlines()]
    clocks = {event.get('clock_id') for event in events}
    if len(clocks) != 1:
        raise ValueError('Shared event clock changed across hosts or reboots')
    ids, rows = set(), []
    for event_index, event in enumerate(events):
        if (event['assignment_id'] != outcome['assignment_id']
                or event['run_id'] != outcome['self_run_id'] or event['event_id'] in ids):
            raise ValueError('Shared event identity is duplicated or belongs to another run')
        ids.add(event['event_id'])
        stamp = event['sample_monotonic_s']
        if not isinstance(stamp, (float, int)) or not math.isfinite(stamp) or stamp < 0:
            raise ValueError('Invalid shared host timestamp')
        details = event['details']
        row = {key: event[key] for key in ('event_id', 'kind', 'source', 'assignment_id', 'run_id', 'sample_monotonic_s')}
        row['event_index'] = event_index
        row.update({key: details.get(key) for key in ('observation_monotonic_s',
            'action_ready_monotonic_s', 'call_id', 'plan_id', 'status')})
        row['details_json'] = json.dumps(details, sort_keys=True, allow_nan=False)
        if event['kind'] == 'inference':
            obs, ready = row['observation_monotonic_s'], row['action_ready_monotonic_s']
            if (obs is None or not math.isfinite(obs) or obs > stamp
                    or ready is not None and (not math.isfinite(ready) or ready < stamp)):
                raise ValueError('Invalid inference timing order')
        rows.append(row)
    # Stable append order resolves equal-resolution timestamps; random UUID order
    # can invert reset_start/reset_end on hosts with a coarse monotonic clock.
    rows.sort(key=lambda row: row['sample_monotonic_s'])
    kinds = {row['kind'] for row in rows}
    if not {'episode_start', 'episode_end'} <= kinds:
        raise ValueError('Missing episode lifecycle events')
    starts = [row['sample_monotonic_s'] for row in rows if row['kind'] == 'episode_start']
    ends = [row['sample_monotonic_s'] for row in rows if row['kind'] == 'episode_end']
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise ValueError('Ambiguous episode boundaries')
    clock = dict(schema_version=1, clock='python.time.monotonic', unit='seconds',
        host_boot_id=next(iter(clocks)),
        scope='recording client host; absolute values cannot be compared across hosts or host reboots',
        episode_start_monotonic_s=starts[0], episode_end_monotonic_s=ends[0],
        stream_timestamps='states/actions/frames/shared use this host monotonic clock; source camera clocks are separate',
        camera_time_semantics='received_monotonic_s is host receipt, not exposure; source timestamp/domain/frame number retained',
        inference_time_semantics='observation ready to client action ready includes transport/queue/model; not pure model compute',
        operator_semantics='Web actions and explicit operator annotations have separate source; unobserved physical stops are not inferred',
        events_source_sha256=hashlib.sha256(raw).hexdigest(), event_count=len(rows))
    signature = hashlib.sha256(json.dumps(dict(clock=clock, rows=rows, outcome=outcome),
                                           sort_keys=True, allow_nan=False).encode()).hexdigest()
    return dict(clock=clock, rows=rows, outcome=outcome, signature=signature)


def stage_shared(payload, staging):
    root = Path(staging) / 'meta/roboeval'
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(payload['rows'], schema=EVENT_SCHEMA), root / 'shared.parquet')
    for name, document in (('clock.json', payload['clock']), ('outcome.json', payload['outcome'])):
        (root / name).write_text(json.dumps(document, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def episode_annotation(outcome):
    labels = outcome['episode_labels']
    return {'roboeval/self_run_id': outcome['self_run_id'],
        'roboeval/comparison_id': outcome['assignment_id'], 'roboeval/self_side': outcome['self_side'],
        'roboeval/score_0_100': outcome['score_0_100'],
        'roboeval/success': labels['success'], 'roboeval/success_status': labels['success_status'],
        'roboeval/completed_stages': labels['completed_stages'],
        'roboeval/completed_stages_status': labels['completed_stages_status'],
        'roboeval/main_issue': labels['main_issue'],
        **{'roboeval/' + key: labels[key] for key in ('drop', 'damage', 'spill')},
        'roboeval/preference': outcome['comparison']['preference'] if outcome['comparison'] else None,
        'roboeval/annotation_source': (outcome['comparison']['annotation_source']
            if outcome['comparison'] else outcome['annotation_source'])}


def validate_shared(root, marker, *, payload=None):
    """Called after file integrity verification, including during upload retries."""
    root = Path(root)
    meta = root / 'meta/roboeval'
    declared = marker.get('roboeval_shared_source_sha256')
    if not declared:
        if payload is not None or any((meta / name).exists() for name in ('shared.parquet', 'clock.json', 'outcome.json')):
            raise ValueError('Undeclared shared metadata')
        return
    outcome = (payload['documents']['outcome'] if payload is not None else
               json.loads((meta / 'outcome.json').read_text(encoding='utf-8')))
    if (outcome != marker.get('roboeval_outcome') or outcome['self_run_id'] != marker['run_id']
            or outcome['assignment_id'] != marker['assignment_id']):
        raise ValueError('Outcome comparison does not match this dataset run')
    from .roboeval_feedback import build_comparison, outcome_for_run, single_run_outcome
    comparison = outcome['comparison']
    if comparison is not None:
        rebuilt = build_comparison({'id': outcome['assignment_id'],
            'task': {'instruction': comparison['task_text']}, 'runs': [
            dict(id=run['run_id'], side=side, **run['model']) for side, run in comparison['runs'].items()]},
            dict(feedback=comparison['raw_feedback'], preference=comparison['preference'],
                 outcomes=comparison['original_outcomes'],
                 **({'b6': comparison['b6']} if 'b6' in comparison else {})))
        expected = outcome_for_run(rebuilt, outcome['self_run_id'])
    else:
        expected = single_run_outcome({'id': outcome['assignment_id'], 'runs': [
            dict(id=outcome['self_run_id'], side=outcome['self_side'])]},
            {'outcomes': {outcome['self_side']: outcome['original_outcome']},
             'feedback': outcome['raw_feedback']}, outcome['self_run_id'])
    if expected != outcome:
        raise ValueError('Outcome labels differ from their submitted feedback')
    clock = (payload['documents']['clock'] if payload is not None else
             json.loads((meta / 'clock.json').read_text(encoding='utf-8')))
    rows = payload['rows'] if payload is not None else pq.read_table(meta / 'shared.parquet').to_pylist()
    actual = hashlib.sha256(json.dumps(dict(clock=clock, rows=rows, outcome=outcome),
                                      sort_keys=True, allow_nan=False).encode()).hexdigest()
    if actual != declared:
        raise ValueError('Shared event provenance differs from export context')
    ep = pq.read_table(root / 'meta/episodes/chunk-000/file-000.parquet').to_pylist()[0]
    if any(ep.get(key) != value for key, value in episode_annotation(outcome).items()):
        raise ValueError('Episode labels do not belong to this run')
