"""Local operator page for Earth Rover trials.

It runs the same evaluation flow as ``colosseum-robot`` in a background thread;
every question the terminal would ask is shown as a form on the page, with the
rover's live camera, telemetry, an abort button and the submitted results.
Serves on localhost only, like the Earth Rovers SDK dashboard.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import queue
import sys
import threading
import time
import webbrowser
from collections import deque
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import evaluation
from .navigation import FRONT_CAMERA, ROVER_TYPE
from .robot_config import RobotClientConfig

STOP_SENTINEL = object()


class Operator:
    """Bridges the evaluation flow's prompts and output to the page."""

    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.pending = None
        self.answers = queue.Queue()
        self.log = deque(maxlen=500)
        self.thread = None
        self.error = ''
        self.phase = 'idle'
        self.counter = 0

    # --- prompts and output -------------------------------------------------
    def ask(self, prompt, options=None):
        with self.lock:
            self.counter += 1
            self.pending = {'id': self.counter, 'prompt': prompt.strip(), 'options': None if not options else [
                {'value': option[0], 'label': option[1]} if isinstance(option, tuple) else {'value': option, 'label': option}
                for option in options]}
        try:
            answer = self.answers.get()
        finally:
            with self.lock:
                self.pending = None
        if answer is STOP_SENTINEL:
            raise evaluation.SessionStopped('session stopped by the operator')
        return answer

    def answer(self, value):
        with self.lock:
            if self.pending is None:
                return False
        self.answers.put(str(value))
        return True

    def write(self, text):
        for line in text.splitlines():
            if line.strip():
                self.log.append(line.rstrip())
        sys.__stdout__.write(text)

    def flush(self):
        sys.__stdout__.flush()

    # --- session lifecycle --------------------------------------------------
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, *, bot='', mission='', checkpoint='', track='', execute=True):
        if self.running():
            raise RuntimeError('A session is already running')
        settings = dict(self.config.adapter_config)
        if bot:
            settings['bot_slug'] = bot
        if mission:
            settings['mission_slug'] = mission
        else:
            settings.pop('mission_slug', None)
        if checkpoint:
            settings['checkpoint'] = int(checkpoint)
        else:
            settings.pop('checkpoint', None)
        config = replace(self.config, adapter_config=settings)
        self.error = ''
        self.log.clear()
        self.phase = 'running'
        evaluation.INTERRUPT.clear()
        evaluation.STOP.clear()
        evaluation.ask = self.ask

        def worker():
            try:
                with contextlib.redirect_stdout(self):
                    evaluation.run_evaluation(config, track=track or None, execute_action=execute)
                self.phase = 'finished'
            except evaluation.SessionStopped:
                self.phase = 'stopped'
            except BaseException as exc:  # The page must show why the session ended.
                self.error = f'{type(exc).__name__}: {exc}'[:400]
                self.phase = 'failed'

        self.thread = threading.Thread(target=worker, name='colosseum-evaluation', daemon=True)
        self.thread.start()

    def abort_trial(self):
        evaluation.INTERRUPT.set()

    def stop_session(self):
        evaluation.STOP.set()
        with self.lock:
            pending = self.pending is not None
        if pending:
            self.answers.put(STOP_SENTINEL)

    # --- what the page shows ------------------------------------------------
    @staticmethod
    def _robot():
        from .robots.earth_rover_mini_plus import EarthRoverMiniPlusRobot
        robot = EarthRoverMiniPlusRobot.active
        return robot if robot is not None and robot.started and not robot.closed else None

    def frame_jpeg(self):
        robot = self._robot()
        if robot is None:
            return None
        frames = robot.link.latest_frames()
        if FRONT_CAMERA not in frames:
            return None
        from PIL import Image
        output = io.BytesIO()
        Image.fromarray(frames[FRONT_CAMERA][0]).save(output, format='JPEG', quality=80)
        return output.getvalue()

    def results(self):
        root = Path(self.config.evaluation_dir)
        rows = []
        manifests = sorted(root.glob('*/manifest.json'), key=lambda path: path.stat().st_mtime, reverse=True)
        for manifest in manifests[:20]:
            try:
                data = json.loads(manifest.read_text())
            except (ValueError, OSError):
                continue
            assignment = data.get('assignment') or {}
            rows.append({'id': assignment.get('id'), 'track': assignment.get('track'),
                         'task': (assignment.get('task') or {}).get('instruction'),
                         'goal': (assignment.get('task') or {}).get('goal'),
                         'outcomes': data.get('outcomes') or {}, 'submitted': bool(data.get('submitted')),
                         'uploaded': sorted(data.get('datasets_uploaded') or {}),
                         'when': time.strftime('%Y-%m-%d %H:%M', time.localtime(manifest.stat().st_mtime))})
        return rows

    def state(self):
        robot = self._robot()
        telemetry = None
        mission = None
        if robot is not None:
            latest = robot.link.latest_telemetry()
            if latest:
                telemetry = {name: round(value, 6) for name, value in latest[0].items()}
            mission = getattr(robot.link, 'mission_slug', None) or getattr(getattr(robot.link, 'session', None), 'mission_slug', None)
        with self.lock:
            pending = dict(self.pending) if self.pending else None
        return {'phase': self.phase, 'running': self.running(), 'error': self.error, 'pending': pending,
                'log': list(self.log)[-80:], 'connected': robot is not None, 'telemetry': telemetry, 'mission': mission,
                'results': self.results(), 'api_url': evaluation.api_url(self.config),
                'bot': self.config.adapter_config.get('bot_slug', ''), 'track': self.config.track,
                'robot_type': self.config.robot_type}

    def missions(self, bot):
        settings = {**self.config.adapter_config, 'bot_slug': bot}
        session = evaluation.rover_session(replace(self.config, adapter_config=settings))
        if session is None:
            return {'missions': [], 'note': 'Simulated or test mode: no FrodoBots missions.'}
        try:
            missions = session.missions()
            for mission in missions:
                try:
                    mission['checkpoints'] = session.checkpoints(mission['slug'])
                except Exception as exc:
                    mission['checkpoints'] = []
                    mission['error'] = type(exc).__name__
            return {'missions': missions}
        finally:
            session.close()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoboColosseum Rover Operator</title>
<style>
:root{--bg:#f4f6f3;--card:#fff;--ink:#17201b;--muted:#5d6a63;--line:#d9e0db;--accent:#2a5fb0;--ok:#1f6f44;--bad:#a8251d;--warn:#8a5300}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,"Segoe UI",Inter,system-ui,sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--line);background:var(--card);display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap}
header h1{font-size:18px;margin:0}header .phase{font-family:ui-monospace,Menlo,monospace;font-size:13px;color:var(--muted)}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;padding:16px 24px;max-width:1400px;margin:0 auto}
@media(max-width:900px){main{grid-template-columns:minmax(0,1fr)}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;display:grid;gap:10px;min-width:0}
.card h2{font-size:14px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin:0}
label{display:grid;gap:4px;font-size:13px;color:var(--muted)}input,select{font:inherit;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--ink)}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:end}.row>label{flex:1 1 180px}
button{font:inherit;padding:9px 14px;border-radius:6px;border:1px solid var(--line);background:#fff;color:var(--ink);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button.danger{background:var(--bad);border-color:var(--bad);color:#fff}
button:disabled{opacity:.5;cursor:default}
.prompt{border:2px solid var(--accent);background:#eef3fb}.prompt .q{font-weight:600}.choices{display:flex;gap:8px;flex-wrap:wrap}
img.cam{width:100%;aspect-ratio:16/9;background:#1a211d;border-radius:8px;object-fit:contain}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:500}
pre.log{margin:0;max-height:280px;overflow:auto;font:12px/1.5 ui-monospace,Menlo,monospace;background:#f7f8f6;border:1px solid var(--line);border-radius:6px;padding:10px;white-space:pre-wrap}
.err{color:var(--bad)}.ok{color:var(--ok)}.muted{color:var(--muted)}.chip{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;background:#e6ebe8}
</style></head><body>
<header><div><h1>RoboColosseum · Earth Rover operator</h1><div class="phase" id="phase">idle</div></div>
<div class="row"><button id="abort" class="danger" disabled>Abort trial</button><button id="stop" disabled>Stop session</button></div></header>
<main>
<section class="card" id="connect"><h2>1 · Robot and mission</h2>
<div class="row"><label>Bot slug<input id="bot" placeholder="link-node-inert"></label><button id="list">List missions</button></div>
<div class="row"><label>Mission<select id="mission"><option value="">— choose after listing —</option></select></label>
<label>Destination checkpoint<select id="checkpoint"><option value="">last</option></select></label></div>
<div class="row"><label>Track<select id="track"><option value="fine-tuning">Fine-tuning (one policy per trial)</option><option value="open">Open (A/B pair)</option></select></label>
<label>Rover motion<select id="execute"><option value="1">Execute actions</option><option value="0">Observe only (no motion)</option></select></label>
<button id="start" class="primary">Start session</button></div>
<div id="missionInfo" class="muted"></div></section>

<section class="card"><h2>Live rover</h2><img class="cam" id="cam" alt="front camera"><div id="telemetry" class="muted">Not connected.</div></section>

<section class="card prompt" id="promptCard" hidden><h2>Waiting for you</h2><div class="q" id="promptText"></div>
<div class="choices" id="choices"></div><div class="row" id="freeText" hidden><label>Answer<input id="answer"></label><button id="send" class="primary">Send</button></div></section>

<section class="card"><h2>Results</h2><div id="results" class="muted">No results yet.</div><div class="muted">Router: <a id="admin" href="#" target="_blank">admin overview</a> · <a id="review" href="#" target="_blank">fine-tuning review API</a></div></section>

<section class="card" style="grid-column:1/-1"><h2>Session log</h2><pre class="log" id="log"></pre><div class="err" id="error"></div></section>
</main>
<script>
const $ = id => document.getElementById(id);
let missions = [], lastPrompt = 0, connected = false;
async function api(path, body){ const r = await fetch(path, body ? {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)} : {}); return r.json(); }
$('list').onclick = async () => {
  $('missionInfo').textContent = 'Listing missions…';
  const data = await api('/api/missions?bot=' + encodeURIComponent($('bot').value.trim()));
  missions = data.missions || [];
  $('mission').innerHTML = '<option value="">— choose —</option>' + missions.map(m => `<option value="${m.slug}">${m.slug} (${m.type}, ${(m.checkpoints||[]).length} checkpoints)</option>`).join('');
  $('missionInfo').textContent = data.note || data.error || (missions.length ? `${missions.length} SDK-enabled mission(s).` : 'This bot has no SDK-enabled active missions; create one in the FrodoBots admin.');
  fillCheckpoints();
};
function fillCheckpoints(){
  const m = missions.find(x => x.slug === $('mission').value); const cps = (m && m.checkpoints) || [];
  $('checkpoint').innerHTML = '<option value="">last' + (cps.length ? ` (#${cps[cps.length-1].sequence})` : '') + '</option>' + cps.map(c => `<option value="${c.sequence}">#${c.sequence} · ${Number(c.latitude).toFixed(6)}, ${Number(c.longitude).toFixed(6)}</option>`).join('');
}
$('mission').onchange = fillCheckpoints;
$('start').onclick = async () => {
  const data = await api('/api/start', {bot: $('bot').value.trim(), mission: $('mission').value, checkpoint: $('checkpoint').value, track: $('track').value, execute: $('execute').value === '1'});
  if (data.error) $('error').textContent = data.error;
};
$('abort').onclick = () => api('/api/abort', {});
$('stop').onclick = () => api('/api/stop', {});
$('send').onclick = () => api('/api/answer', {value: $('answer').value}).then(() => { $('answer').value = ''; });
$('answer').addEventListener('keydown', e => { if (e.key === 'Enter') $('send').onclick(); });
function renderPrompt(p){
  $('promptCard').hidden = !p; if (!p) return;
  if (p.id === lastPrompt) return; lastPrompt = p.id;
  $('promptText').textContent = p.prompt; $('choices').innerHTML = ''; $('freeText').hidden = !!p.options;
  (p.options || []).forEach(o => { const b = document.createElement('button'); b.textContent = o.label; b.className = 'primary'; b.onclick = () => api('/api/answer', {value: o.value}); $('choices').appendChild(b); });
  if (!p.options) $('answer').focus();
}
function renderResults(rows){
  if (!rows.length) { $('results').textContent = 'No results yet.'; return; }
  $('results').innerHTML = '<table><tr><th>When</th><th>Evaluation</th><th>Task</th><th>Score</th><th>Status</th></tr>' + rows.map(r => {
    const score = Object.entries(r.outcomes || {}).map(([side, o]) => `${side}: ${o.success ? 'success' : 'no'} (${Math.round((o.partial_success||0)*4)}/4)`).join('<br>');
    const status = r.submitted ? '<span class="ok">submitted</span>' + (r.uploaded.length ? ', dataset uploaded' : ', upload pending') : '<span class="muted">in progress</span>';
    return `<tr><td>${r.when}</td><td><span class="chip">${r.track}</span><br><small>${r.id||''}</small></td><td>${r.task||''}${r.goal && r.goal.mission_slug ? `<br><small class="muted">mission ${r.goal.mission_slug}${r.goal.checkpoint ? ' · checkpoint ' + r.goal.checkpoint : ''}</small>` : ''}</td><td>${score || '—'}</td><td>${status}</td></tr>`; }).join('') + '</table>';
}
async function poll(){
  try {
    const s = await api('/api/state');
    $('phase').textContent = `${s.robot_type} · ${s.phase}` + (s.mission ? ` · mission ${s.mission}` : '');
    if (!$('bot').value && s.bot) $('bot').value = s.bot;
    if (s.track && !$('track').dataset.set) { $('track').value = s.track; $('track').dataset.set = '1'; }
    $('start').disabled = s.running; $('abort').disabled = !s.running; $('stop').disabled = !s.running;
    renderPrompt(s.pending);
    $('log').textContent = s.log.join('\n'); $('log').scrollTop = $('log').scrollHeight;
    $('error').textContent = s.error || '';
    connected = s.connected;
    if (s.connected) { $('cam').src = '/frame.jpg?t=' + Date.now(); }
    $('telemetry').innerHTML = s.telemetry ? Object.entries(s.telemetry).map(([k, v]) => `<span class="chip">${k} ${v}</span>`).join(' ') : (s.running ? 'Connecting to the rover…' : 'Not connected.');
    renderResults(s.results);
    $('admin').href = s.api_url + '/admin/overview'; $('review').href = s.api_url + '/api/eval/review/fine-tuning';
  } catch (e) { $('phase').textContent = 'page cannot reach the operator process'; }
  setTimeout(poll, 700);
}
poll();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    operator: Operator = None  # set by serve()

    def log_message(self, format, *args):  # keep the terminal for the evaluation log
        pass

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parts = urlsplit(self.path)
        if parts.path == '/':
            body = PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parts.path == '/api/state':
            self._json(self.operator.state())
        elif parts.path == '/api/missions':
            bot = parse_qs(parts.query).get('bot', [''])[0].strip()
            if not bot:
                return self._json({'missions': [], 'error': 'Enter a bot slug first.'})
            try:
                self._json(self.operator.missions(bot))
            except Exception as exc:
                self._json({'missions': [], 'error': f'{type(exc).__name__}: {exc}'[:300]})
        elif parts.path == '/frame.jpg':
            frame = self.operator.frame_jpeg()
            if frame is None:
                return self._json({'error': 'no frame'}, 404)
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)
        else:
            self._json({'error': 'not found'}, 404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length') or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b'{}')
        except ValueError:
            return self._json({'error': 'invalid JSON'}, 400)
        if self.path == '/api/start':
            try:
                self.operator.start(bot=str(payload.get('bot', '')), mission=str(payload.get('mission', '')),
                                    checkpoint=str(payload.get('checkpoint', '')), track=str(payload.get('track', '')),
                                    execute=bool(payload.get('execute', True)))
                self._json({'ok': True})
            except Exception as exc:
                self._json({'ok': False, 'error': f'{type(exc).__name__}: {exc}'[:300]}, 409)
        elif self.path == '/api/answer':
            self._json({'ok': self.operator.answer(payload.get('value', ''))})
        elif self.path == '/api/abort':
            self.operator.abort_trial()
            self._json({'ok': True})
        elif self.path == '/api/stop':
            self.operator.stop_session()
            self._json({'ok': True})
        else:
            self._json({'error': 'not found'}, 404)


def serve(config, *, host='127.0.0.1', port=8770):
    operator = Operator(config)
    handler = type('OperatorHandler', (Handler,), {'operator': operator})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server, operator


def main():
    parser = argparse.ArgumentParser(description='Local operator page for Earth Rover evaluations')
    parser.add_argument('config', nargs='?', default='configs/robot.yaml')
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    config = RobotClientConfig.from_yaml(args.config)
    if config.robot_type != ROVER_TYPE:
        print(f'Note: {config.robot_type} prompts are shown on the page too, but the live view is for the Earth Rover.', flush=True)
    server, _ = serve(config, port=args.port)
    url = f'http://127.0.0.1:{server.server_address[1]}/'
    print(f'Operator page: {url}  (Ctrl-C to quit)', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
