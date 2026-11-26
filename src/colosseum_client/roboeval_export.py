"""Stage immutable RoboEval extension files before LeRobot export is committed."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _files(root, *, ignore_encoder_logs=False):
    files = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('RoboEval dataset cannot contain symbolic links')
        if path.is_file() and not (ignore_encoder_logs and path.name.endswith('.encoder.log')):
            files[path.relative_to(root).as_posix()] = path
    return files


def source_extension(run):
    if (Path(run) / 'roboeval-error.json').exists():
        raise ValueError('Cannot export incomplete RoboEval recording; inspect preserved error report')
    root = Path(run) / 'roboeval'
    context = Path(run) / 'recording-context.json'
    required = context.exists() and _read(context).get('roboeval_required') is True
    if not root.exists():
        if required:
            raise ValueError('Required RoboEval recording is missing')
        return None
    if root.is_symlink():
        raise ValueError('RoboEval recording cannot be a symbolic link')
    manifest = root / 'manifest.json'
    if not manifest.is_file() or not _read(root / 'meta/roboeval/status.json').get('complete'):
        raise ValueError('Cannot export incomplete RoboEval recording')
    document = _read(manifest)
    files = _files(root, ignore_encoder_logs=True)
    inventory = document.get('files')
    if not isinstance(inventory, dict) or set(inventory) != set(files) - {'manifest.json'}:
        raise ValueError('RoboEval source inventory is missing or modified')
    for name, checksum in inventory.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or '..' in path.parts or '\\' in name
                or not name.startswith(('meta/roboeval/', 'videos/roboeval/'))
                or digest(files[name]) != checksum):
            raise ValueError('RoboEval source path/checksum is invalid or modified')
    return dict(root=root, files=files, signature=digest(manifest),
                identity=_read(root / 'meta/roboeval/setup.json').get('identity', {}))


def stage_extension(source, staging):
    if source is None:
        return
    for name, path in source['files'].items():
        destination = staging / ('meta/roboeval/manifest.json' if name == 'manifest.json' else name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    # Verify staged bytes too; copying a changing source must not produce a valid export.
    manifest = _read(staging / 'meta/roboeval/manifest.json')
    if digest(staging / 'meta/roboeval/manifest.json') != source['signature']:
        raise ValueError('RoboEval source changed while staging')
    for name, checksum in manifest['files'].items():
        if digest(staging / name) != checksum:
            raise ValueError('RoboEval source changed while staging')


def validate_uploaded_extension(root):
    """Validate before sending; also used by the separate Web backfill uploader."""
    root = Path(root)
    marker_path = root / 'meta/colosseum.json'
    extension = root / 'meta/roboeval'
    marker = _read(marker_path) if marker_path.is_file() else {}
    if 'training_layout' in marker:
        from .training_delivery import validate_delivery
        validate_delivery(root)
        return
    if 'roboeval_layout' in marker:
        from .roboeval_compact import validate_compact
        from .roboeval_shared import validate_shared
        validate_compact(root)
        from .roboeval_compact import load_stream
        validate_shared(root, marker, payload={
            'documents': marker['roboeval_layout']['documents'],
            'rows': load_stream(root, 'shared', original=True).to_pylist()})
        if marker.get('continuous_recording'):
            from .continuous_recording import validate_continuous
            validate_continuous(root, marker,
                states=load_stream(root, 'states', original=True).to_pylist(),
                frames=load_stream(root, 'frames', original=True).to_pylist())
        return
    declared = marker.get('roboeval_source_sha256')
    if not declared and not extension.exists():
        context = root.parent / 'recording-context.json'
        if context.exists() and _read(context).get('roboeval_required'):
            raise ValueError('Required RoboEval extension is absent from this dataset')
        return  # Legacy datasets retain their upload behavior.
    if not declared or not extension.is_dir():
        raise ValueError('RoboEval extension missing or added after export hashing')
    files = _files(root)
    inventory = marker.get('files', {})
    if set(inventory) != set(files) - {'meta/colosseum.json'}:
        raise ValueError('Extended dataset file inventory differs from committed export')
    for name, checksum in inventory.items():
        if digest(files[name]) != checksum:
            raise ValueError('Extended dataset contains a missing or modified file')
    if not _read(extension / 'status.json').get('complete'):
        raise ValueError('RoboEval capture is incomplete')
    if digest(extension / 'manifest.json') != declared:
        raise ValueError('RoboEval extension source signature differs from export')
    identity = _read(extension / 'setup.json').get('identity', {})
    for key in ('assignment_id', 'run_id', 'policy'):
        if key in marker and identity.get(key) != marker[key]:
            raise ValueError('RoboEval extension identity differs from the LeRobot dataset')
    from .roboeval_shared import validate_shared
    validate_shared(root, marker)
    if marker.get('continuous_recording'):
        import pyarrow.parquet as pq
        from .continuous_recording import validate_continuous
        validate_continuous(root, marker,
            states=pq.read_table(extension / 'states.parquet').to_pylist(),
            frames=pq.read_table(extension / 'frames.parquet').to_pylist())
