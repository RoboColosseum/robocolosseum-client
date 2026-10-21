"""The local operator page runs the terminal flow behind forms."""
import json
import threading
import time
import urllib.request

import pytest

from colosseum_client import evaluation as e
from colosseum_client import rover_ui
from colosseum_client.navigation import ROVER_TYPE
from colosseum_client.robot_config import RobotClientConfig


def config(tmp_path, **settings):
    return RobotClientConfig(url='ws://localhost', token='t', cameras={'front_image': 'front'}, robot_type=ROVER_TYPE,
                             control_hz=10, evaluation_dir=str(tmp_path), adapter_config={'link': 'simulated', **settings})


@pytest.fixture
def page(tmp_path):
    server, operator = rover_ui.serve(config(tmp_path), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    def call(path, body=None):
        request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'}, method='POST' if body is not None else 'GET')
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
    yield operator, call
    server.shutdown()
    server.server_close()
    e.ask = lambda prompt, options=None: input(prompt)
    from colosseum_client.robots.earth_rover import use_shared_link
    use_shared_link(None)


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_prompts_become_forms_and_answers_flow_back(page, monkeypatch):
    operator, call = page
    monkeypatch.setattr(operator, 'router_ok', lambda: True)
    seen = {}
    def fake_run(cfg, **kwargs):
        seen['bot'] = cfg.adapter_config.get('bot_slug')
        seen['mission'] = cfg.adapter_config.get('mission_slug')
        seen['track'] = kwargs.get('track')
        print('Missions on the bot:')
        seen['choice'] = e.ask('Select mission number: ', options=[('1', 'courtyard-loop'), ('2', 'plaza')])
        seen['free'] = e.ask('Feedback (optional): ')
        print('Evaluation result submitted.')
    monkeypatch.setattr(e, 'run_evaluation', fake_run)
    status, body = call('/api/start', {'bot': 'frodobot_x', 'mission': 'plaza', 'checkpoint': '', 'track': 'open', 'execute': False})
    assert status == 200 and json.loads(body)['ok']
    assert wait_until(lambda: json.loads(call('/api/state')[1])['pending'] is not None)
    state = json.loads(call('/api/state')[1])
    assert state['running'] and state['pending']['prompt'] == 'Select mission number:'
    assert state['pending']['options'] == [{'value': '1', 'label': 'courtyard-loop'}, {'value': '2', 'label': 'plaza'}]
    assert 'Missions on the bot:' in state['log']
    assert call('/api/start', {'bot': 'x'})[0] == 409  # one session at a time
    assert json.loads(call('/api/answer', {'value': '2'})[1])['ok']
    assert wait_until(lambda: (json.loads(call('/api/state')[1])['pending'] or {}).get('prompt') == 'Feedback (optional):')
    assert json.loads(call('/api/state')[1])['pending']['options'] is None
    call('/api/answer', {'value': 'smooth run'})
    assert wait_until(lambda: json.loads(call('/api/state')[1])['phase'] == 'finished')
    assert seen == {'bot': 'frodobot_x', 'mission': 'plaza', 'track': 'open', 'choice': '2', 'free': 'smooth run'}
    assert json.loads(call('/api/answer', {'value': 'late'})[1])['ok'] is False


def test_stop_abort_frame_and_results(page, monkeypatch, tmp_path):
    operator, call = page
    monkeypatch.setattr(operator, 'router_ok', lambda: True)
    def fake_run(cfg, **kwargs):
        e.ask('press Enter to start', options=[('', 'Start trial')])
    monkeypatch.setattr(e, 'run_evaluation', fake_run)
    call('/api/start', {'bot': 'frodobot_x'})
    assert wait_until(lambda: json.loads(call('/api/state')[1])['pending'] is not None)
    assert json.loads(call('/api/abort', {})[1])['ok'] and e.INTERRUPT.is_set()
    e.INTERRUPT.clear()
    call('/api/stop', {})
    assert wait_until(lambda: json.loads(call('/api/state')[1])['phase'] == 'stopped')
    assert call('/frame.jpg')[0] == 404
    status, body = call('/')
    assert status == 200 and b'Earth Rover operator' in body
    manifest = tmp_path / 'FT-1' / 'manifest.json'
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({'assignment': {'id': 'FT-1', 'track': 'fine-tuning', 'task': {'instruction': 'Go', 'goal': {'mission_slug': 'plaza'}}},
                                    'outcomes': {'trial': {'success': True, 'partial_success': 1.0}}, 'submitted': True,
                                    'datasets_uploaded': {'run_1': True}}))
    results = json.loads(call('/api/state')[1])['results']
    assert results[0]['id'] == 'FT-1' and results[0]['submitted'] and results[0]['uploaded'] == ['run_1']


def test_missions_endpoint_uses_the_backend_session(page, monkeypatch):
    operator, call = page
    class Session:
        def missions(self): return [{'slug': 'loop', 'type': 'checkpoints'}]
        def checkpoints(self, slug): return [{'sequence': 1, 'latitude': '1', 'longitude': '2'}]
        def close(self): pass
    monkeypatch.setattr(e, 'rover_session', lambda cfg: Session() if cfg.adapter_config['bot_slug'] == 'frodobot_x' else None)
    assert json.loads(call('/api/missions?bot=frodobot_x')[1])['missions'][0]['checkpoints'][0]['sequence'] == 1
    assert 'note' in json.loads(call('/api/missions?bot=other')[1])
    assert 'error' in json.loads(call('/api/missions')[1])


def test_terminal_prompts_still_read_stdin(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda prompt: 'y')
    assert e.yes_no('Continue?') is True
    monkeypatch.setattr('builtins.input', lambda prompt: '3')
    assert e.rover_score('trial') == {'success': False, 'partial_success': 0.75}



def test_connect_first_then_the_trial_rides_on_the_shared_link(page, monkeypatch):
    operator, call = page
    monkeypatch.setattr(operator, 'router_ok', lambda: True)
    assert call('/api/connect', {'bot': ''})[0] == 409
    assert json.loads(call('/api/connect', {'bot': 'frodobot_x'})[1])['ok']
    assert wait_until(lambda: json.loads(call('/api/state')[1])['link_phase'] == 'connected')
    state = json.loads(call('/api/state')[1])
    assert state['connected'] and state['mission'] is None and state['telemetry']['battery'] == 100
    status, body = call('/frame.jpg')
    assert status == 200 and body[:2] == b'\xff\xd8'
    assert call('/api/connect', {'bot': 'frodobot_x'})[0] == 409  # already connected
    assert call('/api/start', {'bot': 'other'})[0] == 409  # another rover needs a disconnect first
    preview = operator.link
    seen = {}
    def fake_run(cfg, **kwargs):
        from colosseum_client.robots import earth_rover_mini_plus as rover
        robot = rover.EarthRoverMiniPlusRobot(cfg)
        robot.prepare_for_task({'goal': {'type': 'gps', 'latitude': 1.0, 'longitude': 2.0, 'mission_slug': 'plaza'}})
        seen['same_link'] = robot.link is preview
        seen['mission'] = preview.mission_slug
        robot.get_observation()
        robot.close()
        seen['after'] = (preview.started, preview.mission_slug, preview.stops)
    monkeypatch.setattr(e, 'run_evaluation', fake_run)
    assert json.loads(call('/api/start', {'bot': 'frodobot_x', 'mission': 'plaza'})[1])['ok']
    assert wait_until(lambda: json.loads(call('/api/state')[1])['phase'] == 'finished')
    assert seen['same_link'] and seen['mission'] == 'plaza'
    assert seen['after'][0] is True and seen['after'][1] is None and seen['after'][2] >= 1
    assert json.loads(call('/api/state')[1])['connected']  # the preview survives the trial
    assert json.loads(call('/api/disconnect', {})[1])['ok']
    assert preview.closed and json.loads(call('/api/state')[1])['link_phase'] == 'disconnected'


def test_start_explains_an_unreachable_router(page, monkeypatch):
    operator, call = page
    monkeypatch.setattr(e, 'run_evaluation', lambda cfg, **kwargs: None)
    status, body = call('/api/start', {'bot': 'frodobot_x'})
    assert status == 409 and 'not reachable' in json.loads(body)['error'] and 'start-router.sh' in json.loads(body)['error']
