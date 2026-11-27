"""Locate a completed export without silently selecting a superseded package."""
import json
from pathlib import Path


def exported_dataset(run_path, status=None):
    run_path = Path(run_path)
    if status is not None and status.get('complete') is False:
        raise ValueError('Latest dataset export is incomplete; refusing an older package')
    physical = run_path / 'export_physical_30hz/lerobot'
    if physical.exists():
        marker = physical / 'meta/colosseum.json'
        if not marker.is_file() or not (physical / 'meta/info.json').is_file():
            raise ValueError('Versioned physical export is incomplete')
        if json.loads(marker.read_text()).get('delivery_profile') != 'yam-30hz-physical-v1':
            raise ValueError('Unexpected versioned physical export profile')
        return physical
    return run_path / 'lerobot'
