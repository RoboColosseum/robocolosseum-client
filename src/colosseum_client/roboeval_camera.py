"""Single-owner camera acquisition; policy reads the latest source frame."""
import threading
import time


class ContinuousCamera:
    def __init__(self, source, side, recorder):
        self.source, self.side, self.recorder = source, side, recorder
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.last_observation = None
        self.error = None
        self.thread = threading.Thread(target=self._loop, name=f'roboeval-camera-{side}', daemon=True)
        self.thread.start()

    def _loop(self):
        try:
            while not self.stop.is_set():
                frame = self.source.read_frame()
                with self.lock:
                    self.latest = frame
                    recorder = self.recorder
                if recorder is not None:
                    recorder.frame(self.side, frame)
        except Exception as exc:
            if not self.stop.is_set():
                self.error = exc
                with self.lock:
                    recorder = self.recorder
                if recorder is not None:
                    recorder.fail(exc)

    def read(self):
        # This only waits for initial acquisition, never competes for the pipeline.
        limit = time.monotonic() + self.source.timeout_ms / 1000 + .1
        while True:
            if self.error is not None:
                raise RuntimeError('Continuous wrist camera failed') from self.error
            with self.lock:
                frame = self.latest
            if frame is not None:
                if time.monotonic() - frame['received_monotonic_s'] > self.source.timeout_ms / 1000:
                    raise RuntimeError('Continuous wrist camera has stale frames')
                self.last_observation = {key: value for key, value in frame.items() if key != 'pixels'}
                return frame['pixels'].copy()
            if time.monotonic() >= limit or self.stop.wait(.005):
                raise RuntimeError('Continuous wrist camera has no frames')

    def observation_timing(self):
        return dict(self.last_observation or {})

    def detach(self):
        with self.lock:
            self.recorder = None

    def close(self):
        self.stop.set()
        try:
            self.source.close()  # Stops the one pipeline and unblocks its reader.
        finally:
            self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError('Continuous wrist reader did not stop')
