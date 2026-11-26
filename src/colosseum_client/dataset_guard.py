"""Serialize dataset export and transfer without acquiring the UI review lock."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import os
import threading

_local = threading.local()
_threads = {}
_guard = threading.Lock()

@contextmanager
def dataset_lock(dataset):
    root = Path(dataset).resolve()
    path = root.parent / '.dataset-operation.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with _guard:
        lock = _threads.setdefault(str(path), threading.RLock())
    with lock:
        held = getattr(_local, 'held', set())
        if str(path) in held:
            yield
            return
        with path.open('a+b') as stream:
            stream.seek(0, 2)
            if not stream.tell():
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX)
            _local.held = held | {str(path)}
            try:
                yield
            finally:
                _local.held = held
                stream.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)

def locked_export(function):
    @wraps(function)
    def wrapped(path, *args, **kwargs):
        with dataset_lock(kwargs.get('export_destination') or Path(path) / 'lerobot'):
            return function(path, *args, **kwargs)
    return wrapped

def locked_upload(function):
    @wraps(function)
    def wrapped(self, run_id, root, *args, **kwargs):
        with dataset_lock(root):
            destination = ensure_expanded_export(getattr(self, 'config', None), run_id, root)
            with dataset_lock(destination):
                return function(self, run_id, destination, *args, **kwargs)
    return wrapped

def ensure_expanded_export(config, run_id, root):
    import json
    from .evaluation import export_expanded_run
    root = Path(root)
    context = root.parent / 'recording-context.json'
    if not context.exists() or not json.loads(context.read_text(encoding='utf-8')).get('roboeval_required'):
        return root
    if (root / 'meta/colosseum.json').exists():
        from .roboeval_export import validate_uploaded_extension
        validate_uploaded_extension(root)
        return root  # A committed package is immutable, including after upload.
    folder = root.parent.parent
    local = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if not local.get('submitted') or not local.get('result'):
        raise ValueError('Expanded datasets require the confirmed final result')
    run = next(r for r in local['assignment']['runs'] if r['id'] == run_id)
    return export_expanded_run(config, local['assignment'], run, folder, local['result'])
