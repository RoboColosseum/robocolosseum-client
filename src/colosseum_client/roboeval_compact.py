"""Offline prototype: preserve an expanded single-episode export in its nine files.

No robot, camera, network or installed Client modules are imported. Existing exports
are read-only. The destination must be new. This is not a deployed export hook.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

EPISODES = 'meta/episodes/chunk-000/file-000.parquet'
MARKER = 'meta/colosseum.json'
STREAMS = ('states', 'actions', 'frames', 'shared')
DOCUMENTS = ('setup', 'status', 'manifest', 'clock', 'outcome')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')


def stream_digest(table):
    encoded = json.dumps(table.to_pylist(), sort_keys=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(table.schema.serialize().to_pybytes() + encoded).hexdigest()


def safe_relative(name):
    value = PurePosixPath(name)
    if not name or value.is_absolute() or '..' in value.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe relative path: {name}')
    return value


def inventory(root):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlinks are not supported')
        if path.is_file():
            name = path.relative_to(root).as_posix()
            result[name] = dict(size=path.stat().st_size, sha256=digest(path))
    return result


def check_inventory(root):
    actual = inventory(root)
    marker = read_json(Path(root) / MARKER)
    if set(marker['files']) != set(actual) - {MARKER}:
        raise ValueError('Committed inventory has missing or extra files')
    for name, checksum in marker['files'].items():
        safe_relative(name)
        if actual[name]['sha256'] != checksum:
            raise ValueError(f'Committed file modified: {name}')
    return actual, marker


def probe(path, *, packets=False):
    command = ['ffprobe', '-v', 'error', '-show_streams', '-of', 'json']
    if packets:
        command += ['-show_packets', '-show_data_hash', 'sha256']
    return json.loads(subprocess.check_output(command + [str(path)], timeout=60))


def packet_signature(path, stream_index=0):
    info = probe(path, packets=True)
    stream = next(item for item in info['streams'] if item['index'] == stream_index)
    packets = [item for item in info['packets'] if item['stream_index'] == stream_index]
    return dict(codec=stream['codec_name'], width=stream['width'], height=stream['height'],
                packets=[dict(pts=item.get('pts_time'), dts=item.get('dts_time'),
                              duration=item.get('duration_time'), sha256=item['data_hash'])
                         for item in packets])


def remux_tracks(inputs, output):
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-copyts']
    for path in inputs:
        command += ['-i', str(path)]
    for index in range(len(inputs)):
        command += ['-map', f'{index}:v:0']
    command += ['-c', 'copy', '-disposition:v:0', 'default']
    for index in range(1, len(inputs)):
        command += [f'-disposition:v:{index}', '0', f'-metadata:s:v:{index}',
                    f'title=continuous_source_segment_{index - 1}']
    command += ['-avoid_negative_ts', 'disabled', '-movflags', '+faststart', str(output)]
    subprocess.run(command, check=True, timeout=60)


def compact_dataset(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be separate directories')
    before, marker = check_inventory(source)
    info = read_json(source / 'meta/info.json')
    episode_table = pq.read_table(source / EPISODES)
    if info['total_episodes'] != 1 or episode_table.num_rows != 1:
        raise ValueError('Prototype supports one episode per dataset')
    episode = episode_table.to_pylist()[0]
    tables = {name: pq.read_table(source / f'meta/roboeval/{name}.parquet') for name in STREAMS}
    documents = {name: read_json(source / f'meta/roboeval/{name}.json') for name in DOCUMENTS}
    if not documents['status'].get('complete'):
        raise ValueError('Incomplete source capture')
    models = sorted(name for name in before if name.startswith('meta/roboeval/models/'))
    resources = [dict(path=name.removeprefix('meta/roboeval/'), sha256=before[name]['sha256'],
                      content=(source / name).read_bytes()) for name in models]
    frames = tables['frames'].to_pylist()
    camera_paths = {}
    for row in frames:
        safe_relative(row['video_path'])
        camera_paths.setdefault(row['camera_id'], set()).add(row['video_path'])
    video_keys = sorted(key for key, feature in info['features'].items() if feature['dtype'] == 'video')
    if len(video_keys) != 3 or set(camera_paths) != {key.removeprefix('observation.images.') for key in video_keys}:
        raise ValueError('Prototype requires the three declared cameras and source frame mappings')
    raw_videos = {name for name in before if name.startswith('videos/roboeval/')}
    if raw_videos != set().union(*camera_paths.values()):
        raise ValueError('Missing or unreferenced source video segment')
    native_files = set(before) - set(models)
    native_files -= {f'meta/roboeval/{name}.parquet' for name in STREAMS}
    native_files -= {f'meta/roboeval/{name}.json' for name in DOCUMENTS}
    native_files -= raw_videos
    if len(native_files) != 9:
        raise ValueError('Expected the original nine single-shard project files')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.ninefile-', dir=destination.parent))
    try:
        for name in sorted(native_files - {MARKER, EPISODES}):
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not name.endswith('.mp4'):
                shutil.copyfile(source / name, target)
        track_mapping, video_checks = {}, []
        for key in video_keys:
            camera_id = key.removeprefix('observation.images.')
            primary = info['video_path'].format(video_key=key,
                chunk_index=episode[f'videos/{key}/chunk_index'],
                file_index=episode[f'videos/{key}/file_index'])
            raw_paths = sorted(camera_paths[camera_id])
            inputs = [primary, *raw_paths]
            remux_tracks([source / name for name in inputs], staging / primary)
            for index, name in enumerate(inputs):
                old, new = packet_signature(source / name), packet_signature(staging / primary, index)
                if old != new:
                    raise ValueError(f'Remux changed encoded packets or source timing: {name}')
                video_checks.append(dict(source_path=name, destination_path=primary,
                                         stream_index=index, packet_count=len(old['packets'])))
                if index:
                    track_mapping[name] = dict(video_path=primary, video_stream_index=index)
        for row in frames:
            old_path = row['video_path']
            row['source_video_path'] = old_path
            row.update(track_mapping[old_path])
        frame_schema = tables['frames'].schema.append(pa.field('source_video_path', pa.string()))
        frame_schema = frame_schema.append(pa.field('video_stream_index', pa.int32()))
        packed_tables = {**tables, 'frames': pa.Table.from_pylist(frames, schema=frame_schema)}
        for name, table in packed_tables.items():
            episode_table = episode_table.append_column(f'roboeval/{name}',
                pa.array([table.to_pylist()], type=pa.list_(pa.struct(list(table.schema)))))
        resource_type = pa.list_(pa.struct([
            ('path', pa.string()), ('sha256', pa.string()), ('content', pa.binary())]))
        episode_table = episode_table.append_column('roboeval/model_resources',
            pa.array([resources], type=resource_type))
        target = staging / EPISODES
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(episode_table, target, compression='zstd')
        layout = dict(version=2 if marker.get('continuous_recording') else 1, status='compact_layout_v1',
            documents=documents,
            documents_source_text={name:(source / f'meta/roboeval/{name}.json').read_bytes().decode('utf-8')
                                   for name in DOCUMENTS},
            stream_sha256={name:stream_digest(table) for name,table in tables.items()},
            stream_schemas={name:base64.b64encode(table.schema.serialize().to_pybytes()).decode('ascii')
                            for name, table in tables.items()},
            stream_columns={name:f'roboeval/{name}' for name in STREAMS},
            model_resources_column='roboeval/model_resources',
            model_paths='Virtual paths relative to the embedded model resources; extract only into a cache when loading',
            video_semantics='MP4 stream 0 retains the original native training view; streams 1+ retain continuous source segments. Frame records select a video_stream_index explicitly.',
            videos=video_checks, source_files=before)
        marker = {**marker, 'roboeval_layout': layout}
        marker['files'] = {name:item['sha256'] for name,item in inventory(staging).items()}
        write_json(staging / MARKER, marker)
        if set(inventory(staging)) != native_files:
            raise ValueError('Compacted output is not exactly the original nine paths')
        if inventory(source) != before:
            raise ValueError('Source changed during compaction')
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)  # Only this invocation's rebuildable staging.
    return destination


def load_stream(root, name, *, original=False):
    if name not in STREAMS:
        raise ValueError(name)
    root = Path(root)
    marker = read_json(root / MARKER)
    layout = marker['roboeval_layout']
    rows = pq.read_table(root / EPISODES, columns=[f'roboeval/{name}']).column(0)[0].as_py()
    if original and name == 'frames':
        for row in rows:
            row['video_path'] = row['source_video_path']
    schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(layout['stream_schemas'][name])))
    if original:
        return pa.Table.from_pylist(rows, schema=schema)
    if name == 'frames':
        schema = schema.append(pa.field('source_video_path', pa.string()))
        schema = schema.append(pa.field('video_stream_index', pa.int32()))
    return pa.Table.from_pylist(rows, schema=schema)


def restore_models(root, destination):
    """Restore byte-identical model resources to a new caller-owned cache directory."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    resources = pq.read_table(Path(root) / EPISODES,
        columns=['roboeval/model_resources']).column(0)[0].as_py()
    seen = set()
    for row in resources:
        name = row['path']
        safe_relative(name)
        if name in seen or not name.startswith('models/'):
            raise ValueError('Invalid or duplicate model resource')
        seen.add(name)
        if hashlib.sha256(row['content']).hexdigest() != row['sha256']:
            raise ValueError('Embedded model resource checksum differs')
    destination.mkdir(parents=True)
    for row in resources:
        path = destination / row['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(row['content'])
    return destination


def validate_compact(root):
    actual, marker = check_inventory(root)
    if len(actual) != 9:
        raise ValueError('Expected nine files')
    layout = marker['roboeval_layout']
    if layout['version'] not in (1, 2) or (layout['version'] == 2) != bool(marker.get('continuous_recording')):
        raise ValueError('Unsupported compact layout')
    docs = layout['documents']
    for name in DOCUMENTS:
        raw = layout['documents_source_text'][name]
        if json.loads(raw) != docs[name]:
            raise ValueError('Embedded source metadata differs')
        if name == 'manifest' and hashlib.sha256(raw.encode('utf-8')).hexdigest() != marker['roboeval_source_sha256']:
            raise ValueError('Source manifest provenance differs')
    for key in ('assignment_id', 'run_id', 'policy'):
        if key in marker and docs['setup'].get('identity', {}).get(key) != marker[key]:
            raise ValueError('Setup identity differs from this dataset')
    for name in STREAMS:
        if stream_digest(load_stream(root, name, original=True)) != layout['stream_sha256'][name]:
            raise ValueError('Embedded source stream differs')
    resources = pq.read_table(Path(root) / EPISODES,
        columns=['roboeval/model_resources']).column(0)[0].as_py()
    seen = set()
    for resource in resources:
        name = resource['path']
        safe_relative(name)
        if name in seen or not name.startswith('models/'):
            raise ValueError('Invalid embedded model resource path')
        seen.add(name)
        expected = docs['manifest']['files'].get('meta/roboeval/' + name)
        if hashlib.sha256(resource['content']).hexdigest() != resource['sha256'] or resource['sha256'] != expected:
            raise ValueError('Embedded model resource modified')
    expected_models = {name.removeprefix('meta/roboeval/') for name in docs['manifest']['files']
                       if name.startswith('meta/roboeval/models/')}
    if seen != expected_models:
        raise ValueError('Embedded model inventory differs')
    if not docs['status']['complete'] or docs['outcome']['self_run_id'] != marker['run_id']:
        raise ValueError('Capture completion or run identity differs')
    rows = pq.read_table(Path(root) / EPISODES,
        columns=['roboeval/self_run_id', 'roboeval/self_side', 'roboeval/score_0_100']).to_pylist()
    if rows[0]['roboeval/self_run_id'] != marker['run_id']:
        raise ValueError('Episode run identity differs')
    if rows[0]['roboeval/self_side'] != docs['outcome']['self_side']:
        raise ValueError('Episode side differs')
    if rows[0]['roboeval/score_0_100'] != docs['outcome']['score_0_100']:
        raise ValueError('Episode score differs')
    frame_rows = load_stream(root, 'frames').to_pylist()
    checked = {}
    for frame in frame_rows:
        safe_relative(frame['video_path'])
        key = frame['video_path'], frame['video_stream_index']
        if key not in checked:
            if key[1] < 1:
                raise ValueError('Continuous frame refers to the native training track')
            try:
                checked[key] = packet_signature(Path(root) / key[0], key[1])['packets']
            except StopIteration as exc:
                raise ValueError('Unknown video stream') from exc
        packets = sorted(checked[key], key=lambda item: float(item['pts']))
        index = frame['video_frame_index']
        if index < 0 or index >= len(packets):
            raise ValueError('Video frame index out of range')
        if abs(float(packets[index]['pts']) - frame['video_timestamp_s']) > 1e-5:
            raise ValueError('Source frame PTS differs from its container')
    return dict(file_count=9, stream_rows={name:load_stream(root,name).num_rows for name in STREAMS},
                source_tracks=len(checked))
