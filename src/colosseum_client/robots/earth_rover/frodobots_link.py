"""The real Earth Rover link: a FrodoBots ride plus Agora video, RTM telemetry and REST commands."""
from __future__ import annotations

import time

from ...navigation import FRONT_CAMERA
from .session import DEFAULT_API_URL, FrodobotsSession


class FrodobotsRoverLink:
    shared = False

    def __init__(self, settings, cameras, size):
        for key in ('sdk_token', 'bot_slug'):
            if not settings.get(key):
                raise ValueError(f'adapter_config.{key} is required for the FrodoBots link')
        if settings.get('telemetry', 'rtm') != 'rtm':
            raise ValueError("adapter_config.telemetry: only 'rtm' is valid for trials; 'shadow' is spike-only")
        self.session = FrodobotsSession(settings['sdk_token'], settings['bot_slug'], settings.get('mission_slug') or '',
                                        api_url=settings.get('frodobots_api_url', DEFAULT_API_URL))
        self.cameras = list(cameras)
        self.size = size
        self.first_frame_timeout = float(settings.get('first_frame_timeout_s', 15))
        self.credentials = None
        self.video = None
        self.telemetry = None
        self.sender = None
        self.started = False

    @property
    def mission_slug(self):
        return self.session.mission_slug or None

    def _open(self, mission_slug):
        """(Re)join the backend session: a ride on ``mission_slug``, or the token-only session when empty.

        A ride uses its own Agora channel, so switching tears the media down and joins again.
        """
        from .commands import RtmRestSender
        from .telemetry import RtmTelemetry
        from .video import AgoraVideo
        self._teardown_media()
        self.credentials = self.session.start_ride(mission_slug or '')
        self.sender = RtmRestSender(self.credentials)
        self.video = AgoraVideo(self.credentials, self.cameras)
        self.video.start()
        self.telemetry = RtmTelemetry(self.credentials)
        self.telemetry.start()
        if not self.video.wait_for(FRONT_CAMERA, self.first_frame_timeout):
            raise RuntimeError(f'No front camera frame within {self.first_frame_timeout:.0f}s; '
                               'check that the rover is online and the mission is active')
        deadline = time.monotonic() + self.first_frame_timeout
        while self.telemetry.latest() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        if self.telemetry.latest() is None:
            raise RuntimeError('No rover telemetry received; the RTM receive path is not working for this ride')
        self.started = True

    def _teardown_media(self):
        for component in (self.telemetry, self.video):
            if component is not None:
                try:
                    component.close()
                except Exception as exc:
                    print(f'Earth Rover link cleanup failed ({type(exc).__name__})', flush=True)
        self.telemetry = self.video = None
        if self.sender is not None:
            self.sender.close()
            self.sender = None

    def start(self, mission_slug=None):
        target = mission_slug or ''
        if self.started and (not target or target == (self.session.mission_slug or '')):
            return  # the preview connection, or already riding this mission
        self._open(target)

    def end_trial(self):
        """Shared connection: confirmed stop, release the ride, return to the token-only preview."""
        confirmed = self.stop(confirm=True)
        if not confirmed:
            raise RuntimeError('Rover stop was not confirmed; the ride is still open. '
                               'Check the rover, then end the ride from the FrodoBots app.')
        if self.session.mission_slug:
            self.session.end_ride()
            try:
                self._open('')
            except Exception as exc:
                self.started = False
                print(f'Preview connection not restored ({type(exc).__name__}); reconnect from the page.', flush=True)

    def latest_frames(self):
        return self.video.latest() if self.video is not None else {}

    def latest_telemetry(self):
        return self.telemetry.latest() if self.telemetry is not None else None

    def send(self, linear, angular):
        self.sender.send(linear, angular)

    def stop(self, confirm=True):
        return self.sender.stop(confirm) if self.sender is not None else True

    def close(self):
        """Confirmed stop, drop the media sessions, then release the ride; keep the ride if the stop is unconfirmed."""
        confirmed = self.stop(confirm=True) if self.started else True
        self._teardown_media()
        try:
            if confirmed and self.credentials is not None:
                self.session.end_ride()
        finally:
            self.session.close()
            self.started = False
        if not confirmed:
            raise RuntimeError('Rover stop was not confirmed; the ride is still open. '
                               'Check the rover, then end the ride from the FrodoBots app.')
