// The operator page: the terminal flow's prompts as forms, the live camera and telemetry,
// a map of the rover and the mission's checkpoints, and the Earth Rovers SDK dashboard's
// drive controls (d-pad / WASD streamed at 10 Hz while held, exactly one zero on release).
(function () {
  const $ = id => document.getElementById(id);
  let missions = [], lastPrompt = 0, state = {};
  async function api(path, body) {
    const r = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
    return r.json();
  }

  /* ---------- Map (after the SDK's dashboard-map.js) ---------- */
  const RoverMap = (function () {
    const MAX_TRAIL = 500, CHECKPOINT_RADIUS_M = 15, ZOOM = 18;
    const ROVER_SVG = '<svg width="34" height="34" viewBox="0 0 34 34" xmlns="http://www.w3.org/2000/svg">' +
      '<circle cx="17" cy="17" r="9" fill="#171e26" stroke="#ffb224" stroke-width="2"/>' +
      '<path d="M17 3 L22 13 L17 10.5 L12 13 Z" fill="#ffb224"/></svg>';
    let map = null, rover = null, arrow = null, trail = null, follow = true, suppress = false;
    let checkpointLayer = null, goalMarker = null, checkpoints = [], destination = null, goal = null;

    function init() {
      map = L.map('map', { zoomControl: true }).setView([0, 0], 2);
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap contributors' }).addTo(map);
      trail = L.polyline([], { color: '#ffb224', weight: 3, opacity: 0.7 }).addTo(map);
      checkpointLayer = L.layerGroup().addTo(map);
      map.on('dragstart', () => { if (!suppress) { follow = false; $('follow').checked = false; } });  // a drag turns follow off
      setTimeout(() => map.invalidateSize(), 200);
    }
    function pan(latlng, zoom) { suppress = true; if (zoom) map.setView(latlng, zoom); else map.panTo(latlng); suppress = false; }
    // Coinciding checkpoints share one marker as a segmented pill; the destination pulses.
    function icon(items) {
      const segs = items.map(cp => `<span class="cp-seg${cp.sequence === destination ? ' next' : ''}">${cp.sequence}</span>`).join('');
      const hasNext = items.some(cp => cp.sequence === destination);
      const title = items.map(cp => 'Checkpoint ' + cp.sequence + (cp.sequence === destination ? ' (destination)' : '')).join('\n');
      const width = Math.max(26, items.length * 26);
      return L.divIcon({ className: '', html: `<div class="cp-marker${hasNext ? ' has-next' : ''}" title="${title}">${segs}</div>`, iconSize: [width, 24], iconAnchor: [width / 2, 12] });
    }
    function setCheckpoints(list, dest) {
      checkpoints = (list || []).map(cp => ({ sequence: Number(cp.sequence), lat: parseFloat(cp.latitude), lng: parseFloat(cp.longitude) }))
        .filter(cp => isFinite(cp.lat) && isFinite(cp.lng));
      destination = dest == null || dest === '' ? (checkpoints.length ? checkpoints[checkpoints.length - 1].sequence : null) : Number(dest);
      checkpointLayer.clearLayers();
      const grouped = {};
      checkpoints.forEach(cp => { const key = cp.lat + ',' + cp.lng; (grouped[key] = grouped[key] || []).push(cp); });
      Object.values(grouped).forEach(items => {
        items.sort((a, b) => a.sequence - b.sequence);
        const latlng = [items[0].lat, items[0].lng];
        L.circle(latlng, { radius: CHECKPOINT_RADIUS_M, color: '#6cb2e6', weight: 1.5, opacity: 0.7, fillColor: '#6cb2e6', fillOpacity: 0.12, interactive: false }).addTo(checkpointLayer);
        L.marker(latlng, { icon: icon(items) }).addTo(checkpointLayer);
      });
      if (checkpoints.length) {
        const bounds = checkpoints.map(cp => [cp.lat, cp.lng]);
        if (rover) bounds.push([rover.getLatLng().lat, rover.getLatLng().lng]);
        suppress = true; map.fitBounds(bounds, { maxZoom: ZOOM, padding: [30, 30] }); suppress = false;
      }
    }
    function setGoal(next) {
      const same = (!next && !goal) || (next && goal && next.latitude === goal.latitude && next.longitude === goal.longitude);
      if (same) return;
      goal = next || null;
      if (goalMarker) { map.removeLayer(goalMarker); goalMarker = null; }
      if (goal) goalMarker = L.marker([goal.latitude, goal.longitude], { icon: L.divIcon({ className: '', html: '<div class="goal-marker" title="Trial goal"></div>', iconSize: [22, 22], iconAnchor: [11, 11] }), zIndexOffset: 900 }).addTo(map);
    }
    function updateRover(lat, lng, heading) {
      const latlng = [lat, lng];
      if (!rover) {
        rover = L.marker(latlng, { icon: L.divIcon({ className: 'rover-icon', html: ROVER_SVG, iconSize: [34, 34], iconAnchor: [17, 17] }), zIndexOffset: 1000 }).addTo(map);
        pan(latlng, checkpoints.length ? null : ZOOM);
        if (checkpoints.length) setCheckpoints(checkpoints.map(cp => ({ sequence: cp.sequence, latitude: cp.lat, longitude: cp.lng })), destination);
      } else rover.setLatLng(latlng);
      if (!arrow && rover.getElement()) arrow = rover.getElement().querySelector('svg');
      if (arrow && isFinite(heading)) arrow.style.transform = `rotate(${heading}deg)`;
      const points = trail.getLatLngs(), last = points.length ? points[points.length - 1] : null;
      const moved = !last || Math.abs(last.lat - lat) > 1e-7 || Math.abs(last.lng - lng) > 1e-7;
      if (moved) {
        points.push(L.latLng(lat, lng));
        if (points.length > MAX_TRAIL) points.shift();
        trail.setLatLngs(points);
        if (follow) pan(latlng);
      }
    }
    // Where the rover should end up: the destination checkpoint, else the trial's GPS goal.
    function target() {
      const cp = checkpoints.find(c => c.sequence === destination);
      if (cp) return { lat: cp.lat, lng: cp.lng, label: 'checkpoint #' + cp.sequence };
      if (goal) return { lat: goal.latitude, lng: goal.longitude, label: 'the trial goal' };
      return null;
    }
    function setFollow(value) { follow = !!value; if (follow && rover) pan(rover.getLatLng()); }
    function clearTrail() { trail.setLatLngs([]); }
    return { init, setCheckpoints, setGoal, updateRover, target, setFollow, clearTrail };
  })();

  function distanceM(aLat, aLng, bLat, bLng) {  // haversine
    const rad = x => x * Math.PI / 180, dLat = rad(bLat - aLat), dLng = rad(bLng - aLng);
    const h = Math.sin(dLat / 2) ** 2 + Math.cos(rad(aLat)) * Math.cos(rad(bLat)) * Math.sin(dLng / 2) ** 2;
    return 6371000 * 2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
  }

  /* ---------- Drive (after the SDK's dashboard.js) ---------- */
  const held = { forward: false, back: false, left: false, right: false };
  const pads = Array.from(document.querySelectorAll('.pad[data-dir]'));
  let driveTimer = null, inFlight = false, pendingCommand = null, lamp = 0, feedbackTimer = null;
  function speedScale() { return parseInt($('speed').value, 10) / 100; }
  $('speed').addEventListener('input', function () { $('speedValue').textContent = this.value + '%'; });
  function feedback(text, isError, sticky) {
    const el = $('driveStatus');
    el.textContent = text; el.className = isError ? 'err' : 'muted';
    clearTimeout(feedbackTimer);
    if (text && !sticky) feedbackTimer = setTimeout(() => { el.textContent = ''; }, 4000);
  }
  // Latest wins, but delivery stays ordered: a zero waits behind an in-flight motion command.
  function sendCommand(command) { pendingCommand = command; flush(); }
  function flush() {
    if (inFlight || !pendingCommand) return;
    const command = pendingCommand; pendingCommand = null; inFlight = true;
    fetch('/api/drive', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(command), keepalive: true })
      .then(r => r.json()).then(data => { if (data.error) { feedback(data.error, true); clearDriveState(); } })
      .catch(err => feedback(String(err.message || err), true))
      .finally(() => { inFlight = false; flush(); });
  }
  function currentCommand() {
    const scale = speedScale();  // +angular turns left, as in the FrodoBots command
    return { linear: ((held.forward ? 1 : 0) - (held.back ? 1 : 0)) * scale, angular: ((held.left ? 1 : 0) - (held.right ? 1 : 0)) * scale };
  }
  function anyHeld() { return held.forward || held.back || held.left || held.right; }
  function refreshPads() { pads.forEach(pad => pad.classList.toggle('active', held[pad.dataset.dir])); }
  function startDriving() {
    if (driveTimer) return;
    sendCommand(currentCommand());
    driveTimer = setInterval(() => { if (anyHeld()) sendCommand(currentCommand()); }, 100);  // 10 Hz while held
  }
  function stopDriving() {
    if (driveTimer) { clearInterval(driveTimer); driveTimer = null; }
    sendCommand({ linear: 0, angular: 0 });  // exactly one zero so the rover stops promptly
  }
  function clearDriveState() {  // halt the loop without transmitting (STOP and 409s have their own stop)
    if (driveTimer) { clearInterval(driveTimer); driveTimer = null; }
    held.forward = held.back = held.left = held.right = false;
    pendingCommand = null;
    refreshPads();
  }
  function setHeld(dir, value) {
    if (held[dir] === value || (value && !state.manual)) return;
    held[dir] = value;
    refreshPads();
    if (anyHeld()) startDriving(); else stopDriving();
  }
  function releaseAll() {  // page-safety events: the rover must not keep a held command
    if (!anyHeld() && !driveTimer) return;
    held.forward = held.back = held.left = held.right = false;
    refreshPads();
    stopDriving();
  }
  async function halt() {  // confirmed stop; while a trial drives the rover it is the trial abort
    clearDriveState();
    feedback('Stopping…', false, true);
    const data = await api('/api/halt', {});
    if (data.error) feedback(data.error, true);
    else if (data.aborting) feedback('Aborting the trial; its adapter stops the rover.');
    else feedback(data.ok ? 'Stop confirmed by the rover.' : 'Stop NOT confirmed: check the rover.', !data.ok, !data.ok);
  }
  pads.forEach(pad => {
    const dir = pad.dataset.dir;
    pad.addEventListener('pointerdown', e => { e.preventDefault(); pad.setPointerCapture(e.pointerId); setHeld(dir, true); });
    ['pointerup', 'pointercancel'].forEach(evt => pad.addEventListener(evt, () => setHeld(dir, false)));
  });
  $('halt').addEventListener('click', halt);
  const KEYMAP = { KeyW: 'forward', ArrowUp: 'forward', KeyS: 'back', ArrowDown: 'back', KeyA: 'left', ArrowLeft: 'left', KeyD: 'right', ArrowRight: 'right' };
  const typing = e => ['input', 'textarea', 'select'].includes((e.target.tagName || '').toLowerCase());
  document.addEventListener('keydown', e => {
    if (e.repeat || typing(e)) return;
    const dir = KEYMAP[e.code];
    if (dir) { e.preventDefault(); setHeld(dir, true); }
    if (e.code === 'Space' && !$('halt').disabled) { e.preventDefault(); halt(); }
  });
  document.addEventListener('keyup', e => { const dir = KEYMAP[e.code]; if (dir) setHeld(dir, false); });
  window.addEventListener('blur', releaseAll);
  document.addEventListener('visibilitychange', () => { if (document.hidden) releaseAll(); });
  function renderLamp() { $('lamp').textContent = lamp ? 'Lamp off' : 'Lamp on'; $('lamp').classList.toggle('on', !!lamp); }
  $('lamp').onclick = async () => {
    const data = await api('/api/lamp', { on: !lamp });
    if (data.error) feedback(data.error, true); else { lamp = data.lamp; renderLamp(); }
  };

  /* ---------- Robot, mission and session ---------- */
  $('list').onclick = async () => {
    $('missionInfo').textContent = 'Listing missions…';
    const data = await api('/api/missions?bot=' + encodeURIComponent($('bot').value.trim()));
    missions = data.missions || [];
    $('mission').innerHTML = '<option value="">— choose —</option>' + missions.map(m => `<option value="${m.slug}">${m.slug} (${m.type}, ${(m.checkpoints || []).length} checkpoints)</option>`).join('');
    $('missionInfo').textContent = data.note || data.error || (missions.length ? `${missions.length} SDK-enabled mission(s).` : 'This bot has no SDK-enabled active missions; create one in the FrodoBots admin.');
    fillCheckpoints();
  };
  function selectedCheckpoints() { const m = missions.find(x => x.slug === $('mission').value); return (m && m.checkpoints) || []; }
  function fillCheckpoints() {
    const cps = selectedCheckpoints();
    $('checkpoint').innerHTML = '<option value="">last' + (cps.length ? ` (#${cps[cps.length - 1].sequence})` : '') + '</option>' +
      cps.map(c => `<option value="${c.sequence}">#${c.sequence} · ${Number(c.latitude).toFixed(6)}, ${Number(c.longitude).toFixed(6)}</option>`).join('');
    RoverMap.setCheckpoints(cps, $('checkpoint').value);
  }
  $('mission').onchange = fillCheckpoints;
  $('checkpoint').onchange = () => RoverMap.setCheckpoints(selectedCheckpoints(), $('checkpoint').value);
  $('start').onclick = async () => {
    const data = await api('/api/start', { bot: $('bot').value.trim(), mission: $('mission').value, checkpoint: $('checkpoint').value, track: $('track').value, execute: $('execute').value === '1' });
    if (data.error) $('error').textContent = data.error;
  };
  $('abort').onclick = () => api('/api/abort', {});
  $('connectBtn').onclick = async () => { const data = await api('/api/connect', { bot: $('bot').value.trim() }); if (data.error) $('error').textContent = data.error; else setTimeout(() => $('list').onclick(), 500); };
  $('disconnectBtn').onclick = async () => { clearDriveState(); const data = await api('/api/disconnect', {}); if (data.error) $('error').textContent = data.error; };
  $('stop').onclick = () => api('/api/stop', {});
  $('send').onclick = () => api('/api/answer', { value: $('answer').value }).then(() => { $('answer').value = ''; });
  $('answer').addEventListener('keydown', e => { if (e.key === 'Enter') $('send').onclick(); });
  $('follow').onchange = function () { RoverMap.setFollow(this.checked); };
  $('clearTrail').onclick = () => RoverMap.clearTrail();

  function renderPrompt(p) {
    $('promptCard').hidden = !p; if (!p) return;
    if (p.id === lastPrompt) return; lastPrompt = p.id;
    $('promptText').textContent = p.prompt; $('choices').innerHTML = ''; $('freeText').hidden = !!p.options;
    (p.options || []).forEach(o => { const b = document.createElement('button'); b.textContent = o.label; b.className = 'primary'; b.onclick = () => api('/api/answer', { value: o.value }); $('choices').appendChild(b); });
    if (!p.options) $('answer').focus();
  }
  function renderResults(rows) {
    if (!rows.length) { $('results').textContent = 'No results yet.'; return; }
    $('results').innerHTML = '<table><tr><th>When</th><th>Evaluation</th><th>Task</th><th>Score</th><th>Status</th></tr>' + rows.map(r => {
      const score = Object.entries(r.outcomes || {}).map(([side, o]) => `${side}: ${o.success ? 'success' : 'no'} (${Math.round((o.partial_success || 0) * 4)}/4)`).join('<br>');
      const status = r.submitted ? '<span class="ok">submitted</span>' + (r.uploaded.length ? ', dataset uploaded' : ', upload pending') : '<span class="muted">in progress</span>';
      return `<tr><td>${r.when}</td><td><span class="chip">${r.track}</span><br><small>${r.id || ''}</small></td><td>${r.task || ''}${r.goal && r.goal.mission_slug ? `<br><small class="muted">mission ${r.goal.mission_slug}${r.goal.checkpoint ? ' · checkpoint ' + r.goal.checkpoint : ''}</small>` : ''}</td><td>${score || '—'}</td><td>${status}</td></tr>`; }).join('') + '</table>';
  }
  function renderMap(t, goal) {
    RoverMap.setGoal(goal);
    const lat = t ? Number(t.latitude) : NaN, lng = t ? Number(t.longitude) : NaN;
    const fix = isFinite(lat) && isFinite(lng) && (Math.abs(lat) > 1e-9 || Math.abs(lng) > 1e-9);
    if (!fix) { $('mapInfo').textContent = t ? 'No GPS fix yet (rover reports 0, 0).' : 'Waiting for a GPS fix…'; return; }
    RoverMap.updateRover(lat, lng, Number(t.heading_deg));
    const target = RoverMap.target();
    const parts = [`Rover at ${lat.toFixed(6)}, ${lng.toFixed(6)}`, `heading ${Math.round(t.heading_deg)}°`, `speed ${t.speed}`];
    if (target) parts.push(`<b>${Math.round(distanceM(lat, lng, target.lat, target.lng))} m</b> to ${target.label}`);
    if (t.age_s > 5) parts.push(`<span class="err">telemetry ${t.age_s}s old</span>`);
    $('mapInfo').innerHTML = parts.join(' · ');
  }
  function renderDrive(s) {
    pads.forEach(pad => { pad.disabled = !s.manual; });
    $('halt').disabled = !(s.manual || s.trial);
    $('lamp').disabled = !s.manual;
    if (!s.manual) clearDriveState();
    if (lamp !== s.lamp) { lamp = s.lamp; renderLamp(); }
    if (s.drive_error) feedback(s.drive_error, true, true);
    else if (!$('driveStatus').textContent || $('driveStatus').dataset.auto) {
      $('driveStatus').dataset.auto = '1';
      $('driveStatus').className = 'muted';
      $('driveStatus').textContent = s.trial ? 'A trial is driving the rover; STOP aborts the trial.' :
        s.manual ? 'Manual control: hold a key or pad; release to stop.' : 'Connect to a rover to drive it.';
    }
  }
  async function poll() {
    try {
      const s = await api('/api/state');
      state = s;
      $('phase').textContent = `${s.robot_type} · ${s.phase}` + (s.mission ? ` · mission ${s.mission}` : '');
      if (!$('bot').value && s.bot) $('bot').value = s.bot;
      if (s.track && !$('track').dataset.set) { $('track').value = s.track; $('track').dataset.set = '1'; }
      $('start').disabled = s.running; $('abort').disabled = !s.running; $('stop').disabled = !s.running;
      const linked = s.link_phase === 'connected' || s.link_phase === 'connecting';
      $('connectBtn').hidden = linked; $('disconnectBtn').hidden = !linked; $('disconnectBtn').disabled = s.running || s.link_phase === 'connecting';
      $('bot').disabled = linked;
      $('status').innerHTML = `Router ${s.router_ok ? '<span class="ok">reachable</span>' : '<span class="err">not reachable at ' + s.api_url + ' (start it: start-router.sh)</span>'} · Rover ${
        s.link_phase === 'connected' ? '<span class="ok">connected' + (s.mission ? ' · riding ' + s.mission : ' (preview, no ride)') + '</span>' :
        s.link_phase === 'connecting' ? '<span class="muted">connecting…</span>' :
        s.link_phase === 'error' ? '<span class="err">' + (s.link_error || 'connection failed') + '</span>' : '<span class="muted">not connected</span>'}`;
      renderPrompt(s.pending);
      $('log').textContent = s.log.join('\n'); $('log').scrollTop = $('log').scrollHeight;
      $('error').textContent = s.error || '';
      if (s.connected) { $('cam').src = '/frame.jpg?t=' + Date.now(); }
      $('telemetry').innerHTML = s.telemetry ? Object.entries(s.telemetry).map(([k, v]) => `<span class="chip">${k} ${v}</span>`).join(' ') : (s.link_phase === 'connecting' || s.running ? 'Connecting to the rover…' : 'Not connected. Enter the bot slug and press Connect.');
      renderMap(s.telemetry, s.goal);
      renderDrive(s);
      renderResults(s.results);
      $('admin').href = s.api_url + '/admin/overview'; $('review').href = s.api_url + '/api/eval/review/fine-tuning';
    } catch (e) { $('phase').textContent = 'page cannot reach the operator process'; }
    setTimeout(poll, 700);
  }
  RoverMap.init();
  poll();
})();
