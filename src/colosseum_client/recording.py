"""Per-trial multi-camera recording with capture timing and restartable encoding."""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
import shutil
import subprocess
import time
import tempfile

import numpy as np
from PIL import Image


def ffmpeg_vfr_args():
    """ffmpeg options that keep each frame's recorded duration (variable frame rate).

    ffmpeg 5.1 added -fps_mode to replace -vsync, and recent releases reject -vsync;
    Ubuntu 22.04 still ships ffmpeg 4.4, which has only -vsync.
    """
    # Bytes, not text: the help includes non-ASCII units (µs, °C) that a non-UTF-8 locale cannot decode.
    usage = subprocess.run(['ffmpeg', '-hide_banner', '-h', 'full'], capture_output=True).stdout
    return ['-fps_mode' if b'\n-fps_mode' in usage else '-vsync', 'vfr']


class OptionalRecording:
    """Keep extra instrumentation errors out of control, without certifying bad data.

    Producers only latch an error in memory. Disk reporting happens at close,
    off the control thread; the strict recorder still owns capture validation.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.recorder = None
        self.error = None
        self._error_lock = threading.Lock()
        self._reported = False

    def fail(self, error):
        with self._error_lock:
            if self.error is None:
                self.error = error
        if self.recorder is not None:
            try:
                self.recorder.fail(error)
            except Exception:
                pass

    def call(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            self.fail(error)
            return None

    def start(self, robot):
        if self.recorder is not None and self.error is None:
            self.call(self.recorder.start, robot, observer=self)

    def _record(self, method, *args, **kwargs):
        if self.recorder is not None and self.error is None:
            self.call(getattr(self.recorder, method), *args, **kwargs)

    def plan(self, *args, **kwargs):
        self._record('plan', *args, **kwargs)

    def select(self, *args, **kwargs):
        self._record('select', *args, **kwargs)

    def sent(self, *args, **kwargs):
        self._record('sent', *args, **kwargs)

    def frame(self, *args, **kwargs):
        self._record('frame', *args, **kwargs)

    def close(self, *, complete=False):
        if self.recorder is not None:
            self.call(self.recorder.close, complete=complete and self.error is None)
        if self.error is not None and not self._reported:
            self._reported = True
            print(f'RoboEval data incomplete: {type(self.error).__name__}: {self.error}. '
                  'PK/result submission continues; partial recordings remain local.', flush=True)
            try:
                self.path.mkdir(parents=True, exist_ok=True)
                (self.path / 'roboeval-error.json').write_text(json.dumps({
                    'incomplete': True, 'error_type': type(self.error).__name__,
                    'error': str(self.error)}, indent=2) + '\n', encoding='utf-8')
            except Exception as error:
                print(f'Cannot persist RoboEval error report: {error}', flush=True)


class TrialRecorder:
    def __init__(self, path: Path, cameras: list[str], *, queue_size: int = 256, store_images=True):
        if shutil.which('ffmpeg') is None:
            raise RuntimeError('Install ffmpeg before starting a trial')
        self.path = path
        self.cameras = cameras
        self.store_images = store_images
        path.mkdir(parents=True, exist_ok=True)
        self.frames = 0
        self.written_frames = 0
        self.queue_peak = 0
        self.started = time.monotonic()
        if queue_size < 1:
            raise ValueError('Recording queue_size must be positive')
        self._queue = queue.Queue(maxsize=queue_size)
        self._error = None
        self._closed = False
        self._worker = threading.Thread(target=self._write_loop, name='colosseum-recording')
        self._worker.start()

    def _check_error(self):
        if self._error is not None:
            raise RuntimeError('Recording failed; trial video is incomplete') from self._error

    def add(self, observation, action=None, *, captured_at=None, policy_action=None,
            action_executed_monotonic_s=None):
        """Copy buffers and enqueue without waiting for compression or disk I/O."""
        if self._closed:
            raise RuntimeError('Recorder is closed')
        self._check_error()
        stamp = (time.monotonic() if captured_at is None else captured_at) - self.started
        images = {}
        for camera in self.cameras:
            if camera not in observation.images:
                raise ValueError(f'Missing required camera: {camera}')
            if self.store_images:
                images[camera] = np.array(observation.images[camera], dtype=np.uint8, copy=True)
        record = {'frame': self.frames, 'seconds': stamp, 'captured_monotonic_s': self.started + stamp,
                  'joints': np.asarray(observation.joints).tolist(),
                  'gripper': np.asarray(observation.gripper).tolist(),
                  'cartesian_position': np.asarray(observation.cartesian_position).tolist(),
                  'action': None if action is None else np.asarray(action).tolist()}
        if action_executed_monotonic_s is not None:
            record['action_executed_monotonic_s'] = float(action_executed_monotonic_s)
        if policy_action is not None:
            record['policy_action'] = np.asarray(policy_action).tolist()
        extra_state = getattr(observation, 'extra_state', None)
        if extra_state:
            record['extra_state'] = {name: np.asarray(value).tolist() for name, value in extra_state.items()}
        try:
            self._queue.put_nowait((images, record))
            self.queue_peak = max(self.queue_peak, self._queue.qsize())
        except queue.Full as exc:
            self._error = RuntimeError('Recording queue full: disk writer cannot keep up')
            raise self._error from exc
        self.frames += 1

    def _write_frame(self, images, record):
        for camera, pixels in images.items():
            folder = self.path / camera
            folder.mkdir(exist_ok=True)
            Image.fromarray(pixels).save(folder / f"{record['frame']:06d}.png", compress_level=0)
        with (self.path / 'frames.jsonl').open('a') as output:
            output.write(json.dumps(record) + '\n')

    def _write_loop(self):
        write_failed = False
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                # Producer overflow must not erase previously accepted records.
                # Only an actual write failure prevents further writes.
                if not write_failed:
                    self._write_frame(*item)
                    self.written_frames += 1
            except Exception as exc:
                self._error = exc
                write_failed = True
            finally:
                self._queue.task_done()

    def close(self):
        """Drain accepted frames before encoding. Call off the asyncio event loop."""
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._worker.join()
            (self.path / 'recording-status.json').write_text(json.dumps({
                'complete': self._error is None, 'frames': self.frames,
                'written_frames': self.written_frames, 'queue_peak': self.queue_peak,
                'image_storage': 'png' if self.store_images else 'continuous_source'}) + '\n')
        self._check_error()

    @staticmethod
    def encode(path: Path, cameras: list[str]) -> dict[str, Path]:
        status = path / 'recording-status.json'
        if status.exists() and not json.loads(status.read_text())['complete']:
            raise RuntimeError('Cannot encode an incomplete recording')
        if status.exists() and json.loads(status.read_text()).get('image_storage') == 'continuous_source':
            videos = {camera: continuous_preview(path, camera, path/camera/'video.mp4') for camera in cameras}
            if not all(videos.values()):
                raise ValueError('Continuous source videos are missing')
            return videos
        records = [json.loads(line) for line in (path / 'frames.jsonl').read_text().splitlines()]
        if not records:
            raise ValueError('No recorded frames')
        videos = {}
        for camera in cameras:
            folder = path / camera
            lines = []
            for index, row in enumerate(records):
                duration = records[index+1]['seconds'] - row['seconds'] if index+1 < len(records) else 1/15
                lines.extend([f"file '{row['frame']:06d}.png'", f'duration {max(duration, .001):.6f}'])
            lines.append(f"file '{records[-1]['frame']:06d}.png'")
            (folder / 'frames.txt').write_text('\n'.join(lines) + '\n')
            temp = folder / 'video.pending.mp4'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat',
                '-safe', '1', '-i', 'frames.txt', *ffmpeg_vfr_args(), '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', temp.name],
                cwd=folder, check=True)
            final = folder / 'video.mp4'
            temp.replace(final)
            videos[camera] = final
        return videos


def continuous_cameras(path):
    """Discover source streams for review before any training export exists."""
    import pyarrow.parquet as pq
    root=(Path(path)/'roboeval').resolve()
    index=root/'meta/roboeval/frames.parquet'
    if not index.is_file():
        return []
    rows=pq.read_table(index,columns=['camera_id','video_path']).to_pylist()
    cameras=[]
    for camera,relative in dict.fromkeys((r['camera_id'],r['video_path']) for r in rows):
        source=(root/relative).resolve()
        if root not in source.parents or not source.is_file():
            raise ValueError('Missing or unsafe continuous video segment')
        if camera not in cameras:
            cameras.append(camera)
    return cameras


def continuous_preview(path, camera, destination):
    """Reuse recorded source segments for review; never invent missing images."""
    import pyarrow.parquet as pq
    root = (Path(path)/'roboeval').resolve()
    index = root/'meta/roboeval/frames.parquet'
    if not index.exists():
        return None
    rows = pq.read_table(index, columns=['camera_id','video_path']).to_pylist()
    relatives = list(dict.fromkeys(r['video_path'] for r in rows if r['camera_id']==camera))
    if not relatives:
        return None
    sources = [(root/relative).resolve() for relative in relatives]
    if any(root not in p.parents or not p.is_file() for p in sources):
        raise ValueError('Missing or unsafe continuous video segment')
    if len(sources)==1:
        return sources[0]
    destination = Path(destination)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.source-preview-',dir=destination.parent) as temp:
        listing=Path(temp)/'segments.txt'
        listing.write_text(''.join("file '"+str(p).replace("'", "'\\''")+"'\n" for p in sources))
        pending=Path(temp)/'video.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat',
                        '-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(pending)],check=True)
        pending.replace(destination)
    return destination
