"""Validate SDK instrumentation before constructing cameras or robot arms."""

def require_feedback_capabilities(*, continuous=False, chain_type=None):
    if chain_type is None:
        try:
            from i2rt.motor_drivers.dm_driver import DMChainCanInterface
        except ImportError as error:
            raise RuntimeError('RoboEval requires the pinned I2RT SDK and feedback patch; see docs/r16-dataset-changes.md') from error
        chain_type = DMChainCanInterface
    required = ['get_feedback_snapshot']
    if continuous:
        required += ['start_feedback_history', 'drain_feedback_history', 'stop_feedback_history']
    missing = [name for name in required if not callable(getattr(chain_type, name, None))]
    if missing:
        raise RuntimeError('I2RT feedback patch is missing capabilities: ' + ', '.join(missing))
