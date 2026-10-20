"""Earth Rover Mini+ transports: the FrodoBots platform (Agora video, RTM commands and telemetry) or a simulator."""
from __future__ import annotations

from .link import RoverLink, SimulatedRoverLink

__all__ = ['RoverLink', 'SimulatedRoverLink', 'make_link']


def make_link(settings, *, cameras, size):
    """Select the transport named by ``adapter_config.link`` without importing hardware SDKs for the simulator."""
    kind = settings.get('link', 'frodobots')
    if kind == 'simulated':
        return SimulatedRoverLink(cameras, size)
    if kind == 'frodobots':
        from .frodobots_link import FrodobotsRoverLink
        return FrodobotsRoverLink(settings, cameras, size)
    raise ValueError("adapter_config.link must be 'frodobots' or 'simulated'")
