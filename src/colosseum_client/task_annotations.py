"""One human difficulty rating per task, shared by every run in its dataset."""
import json
from pathlib import Path


def difficulty(value, *, required=False):
    if value is None or value == '':
        if required:
            raise ValueError('Task difficulty is required before submitting')
        return None
    if isinstance(value, str) and value.isascii() and value.isdigit():
        value = int(value)
    if type(value) is not int or not 1 <= value <= 10:
        raise ValueError('Task difficulty must be an integer from 1 to 10')
    return value


def exported_difficulties(folder, document):
    values = []
    for run in document['assignment']['runs']:
        marker = Path(folder) / run['id'] / 'lerobot/meta/colosseum.json'
        if marker.exists():
            values.append(json.loads(marker.read_text()).get('task_difficulty'))
    return values


def set_difficulty(folder, document, value, *, required=False):
    """Update an in-memory manifest; existing exports must remain unchanged."""
    exported = exported_difficulties(folder, document)
    # Historical exports without a rating can still resume without changing their signature.
    value = difficulty(value, required=required and not exported)
    if any(existing != value for existing in exported):
        raise ValueError('Task difficulty cannot change after a dataset has been exported')
    if value is None:
        document.pop('task_difficulty', None)
        document.pop('task_difficulty_scale', None)
    else:
        document.update(task_difficulty=value, task_difficulty_scale=10)
    return value
