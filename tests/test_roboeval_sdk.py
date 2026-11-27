from types import SimpleNamespace
import pytest
from colosseum_client.roboeval_sdk import require_feedback_capabilities

def test_missing_sdk_capabilities_fail_before_construction():
    with pytest.raises(RuntimeError, match='get_feedback_snapshot'):
        require_feedback_capabilities(chain_type=object)

def test_snapshot_only_is_insufficient_for_continuous_recording():
    sdk = SimpleNamespace(get_feedback_snapshot=lambda: None)
    require_feedback_capabilities(chain_type=sdk)
    with pytest.raises(RuntimeError, match='start_feedback_history'):
        require_feedback_capabilities(chain_type=sdk, continuous=True)

def test_all_required_methods_are_callable():
    sdk = SimpleNamespace(**{name: lambda: None for name in (
        'get_feedback_snapshot', 'start_feedback_history', 'drain_feedback_history', 'stop_feedback_history')})
    require_feedback_capabilities(chain_type=sdk, continuous=True)
    sdk.drain_feedback_history = None
    with pytest.raises(RuntimeError, match='drain_feedback_history'):
        require_feedback_capabilities(chain_type=sdk, continuous=True)


def test_depth_continuous_combination_fails_before_device_initialization():
    from colosseum_client.robots.yam import YAMRobot
    # No camera settings or hardware dependencies: guard must execute first.
    with pytest.raises(ValueError, match='cannot be combined'):
        YAMRobot(SimpleNamespace(roboeval_state_hz=100), depth=True)
