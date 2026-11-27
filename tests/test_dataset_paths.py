import json
import pytest
from colosseum_client.dataset_paths import exported_dataset


def test_legacy_path_is_preserved(tmp_path):
    assert exported_dataset(tmp_path) == tmp_path / 'lerobot'


def test_upload_guard_uses_export_returned_path(tmp_path, monkeypatch):
    from colosseum_client import dataset_guard
    rebuilt = tmp_path / 'export_physical_30hz/lerobot'
    calls = []
    monkeypatch.setattr(dataset_guard, 'ensure_expanded_export', lambda *args: rebuilt)

    class API:
        @dataset_guard.locked_upload
        def upload_dataset(self, run_id, root):
            calls.append((run_id, root))
            return 'uploaded'

    assert API().upload_dataset('run_a', tmp_path / 'lerobot') == 'uploaded'
    assert calls == [('run_a', rebuilt)]


def test_versioned_export_takes_precedence(tmp_path):
    (tmp_path / 'lerobot').mkdir()
    output = tmp_path / 'export_physical_30hz/lerobot'
    (output / 'meta').mkdir(parents=True)
    (output / 'meta/info.json').write_text('{}')
    (output / 'meta/colosseum.json').write_text(json.dumps({'delivery_profile': 'yam-30hz-physical-v1'}))
    assert exported_dataset(tmp_path) == output


def test_failed_latest_export_never_falls_back(tmp_path):
    (tmp_path / 'lerobot').mkdir()
    with pytest.raises(ValueError, match='refusing an older'):
        exported_dataset(tmp_path, {'complete': False})
    (tmp_path / 'export_physical_30hz/lerobot').mkdir(parents=True)
    with pytest.raises(ValueError, match='incomplete'):
        exported_dataset(tmp_path)
