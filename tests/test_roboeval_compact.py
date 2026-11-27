import base64
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from colosseum_client.evaluation import EvalAPI
from colosseum_client.lerobot_export import export_trial
from colosseum_client.roboeval_compact import EPISODES, MARKER, inventory, load_stream
from colosseum_client.roboeval_export import validate_uploaded_extension
from test_roboeval import setup_file
from test_roboeval_shared import assignment, result, make_recording
from colosseum_client.roboeval_feedback import build_comparison, outcome_for_run


def build(path):
    calibration = setup_file(path.parent / (path.name + '-cal'))
    cameras = make_recording(path, calibration, 'policy_first')
    options = dict(fps=30, robot_type='yam', task='orange', metadata={
        'assignment_id':'ev_shared', 'run_id':'policy_first', 'side':'A',
        'roboeval_outcome':outcome_for_run(build_comparison(assignment(), result()), 'policy_first')})
    return cameras, options


@pytest.fixture(scope='module')
def packed(tmp_path_factory):
    run = tmp_path_factory.mktemp('compact') / 'policy_first'
    cameras, options = build(run)
    output = export_trial(run, cameras, **options)
    return run, output


def test_real_uploader_announces_exactly_nine_files_and_skips_completed_retry(packed, monkeypatch):
    _, output = packed
    api = EvalAPI.__new__(EvalAPI)
    manifests, completed, puts, calls = [], set(), [], []
    def request(method, route, body=None):
        calls.append(route)
        if route.endswith('/dataset'):
            manifests.append(body['files'])
            return {}
        if '/upload-url' in route:
            name = parse_qs(urlsplit(route).query)['path'][0]
            if name in completed:
                return {'complete':True}
            return {'storage':'s3', 'url':'https://storage.invalid/file?path=' + name,
                    'headers':{'Content-Type':'application/octet-stream'}}
        assert route.endswith('/complete')
        assert len(completed) == 9
        return {'complete':True}
    api.request = request
    def send(request):
        name = parse_qs(request.url.query.decode())['path'][0]
        assert request.read() == (output / name).read_bytes()
        completed.add(name)
        puts.append(name)
        return httpx.Response(200)
    constructor = httpx.Client
    monkeypatch.setattr(httpx,'Client',lambda **kw:constructor(transport=httpx.MockTransport(send),**kw))
    api.upload_dataset('policy_first',output)
    api.upload_dataset('policy_first',output)
    assert len(puts) == 9 and len(manifests) == 2 and manifests[0] == manifests[1]
    assert len(manifests[0]) == 9
    assert calls[-1].endswith('/complete')
    assert any(item['path'] == EPISODES for item in manifests[0])




@pytest.mark.parametrize('kind',['model','episode_label','frame_track','action_tail'])
def test_semantic_corruption_is_rejected_even_after_file_hash_recommit(packed,tmp_path,kind):
    _,original=packed
    output=tmp_path/'lerobot'
    shutil.copytree(original,output)
    table=pq.read_table(output/EPISODES)
    rows=table.to_pylist()
    if kind == 'model':
        rows[0]['roboeval/model_resources'][0]['content'] += b'corruption'
    elif kind == 'episode_label':
        rows[0]['roboeval/drop']='no'
    elif kind == 'frame_track':
        rows[0]['roboeval/frames'][0]['video_stream_index']=999
    else:
        rows[0]['roboeval/actions'][0]['values'][0] += .25
    pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),output/EPISODES)
    marker=json.loads((output/MARKER).read_text())
    marker['files'][EPISODES]=hashlib.sha256((output/EPISODES).read_bytes()).hexdigest()
    (output/MARKER).write_text(json.dumps(marker),encoding='utf-8')
    with pytest.raises(ValueError):
        validate_uploaded_extension(output)


def test_compaction_failure_keeps_raw_images_and_retry_is_atomic(tmp_path,monkeypatch):
    from colosseum_client import roboeval_compact
    run=tmp_path/'policy_first'
    cameras,options=build(run)
    original=roboeval_compact.remux_tracks
    def fail(*args):
        raise RuntimeError('injected remux failure')
    monkeypatch.setattr(roboeval_compact,'remux_tracks',fail)
    with pytest.raises(RuntimeError,match='injected'):
        export_trial(run,cameras,**options)
    assert not (run/'lerobot').exists()
    assert not list(run.glob('.lerobot-*')) and not list(run.glob('.ninefile-*'))
    assert list(run.glob('*/*.png'))
    monkeypatch.setattr(roboeval_compact,'remux_tracks',original)
    output=export_trial(run,cameras,**options)
    validate_uploaded_extension(output)
    assert len(inventory(output)) == 9
    assert not list(run.glob('*/*.png'))
