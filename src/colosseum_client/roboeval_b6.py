"""Explicit operator event labels keyed by assignment run ID (never display order)."""
FIELDS = ('drop', 'damage', 'spill')
VALUES = ('yes', 'no', 'unknown', 'not_applicable')


def validate_b6(assignment, raw, *, required=False):
    runs = assignment.get('runs', [])
    ids = [run.get('id') for run in runs]
    if (len(runs) != 2 or {r.get('side') for r in runs} != {'A', 'B'}
            or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != 2):
        raise ValueError('B6 requires unambiguous A/B run identities')
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or set(raw) - set(ids):
        raise ValueError('B6 contains an unknown run identity')
    result = {}
    for run_id in ids:
        labels = raw.get(run_id, {})
        if not isinstance(labels, dict) or set(labels) - set(FIELDS):
            raise ValueError('B6 contains unknown event fields')
        result[run_id] = {}
        for field in FIELDS:
            value = labels.get(field)
            if value in (None, ''):
                if required:
                    raise ValueError('Select Drop, Damage and Spill for both runs before submitting')
                continue
            if not isinstance(value, str) or value not in VALUES:
                raise ValueError('B6 values must be yes/no/unknown/not_applicable')
            result[run_id][field] = value
    return result
