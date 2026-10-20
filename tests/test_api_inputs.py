"""Transport regression fixtures; these do not run or evaluate a vision model."""
import base64
import json
from pathlib import Path

import pytest

from tracebrief import server


@pytest.mark.parametrize('observation', [None, {'id': [], 'text': 'invalid'}, 'invalid'])
def test_invalid_observation_returns_400_without_writing(tmp_path, monkeypatch, observation):
    monkeypatch.setattr(server, 'RUNTIME', tmp_path)
    job_id = 'ffffffffffffffff'
    job = {'id': job_id, 'status': 'completed', 'result': {'events': [
        {'id': 'event-001', 'title': 'Fixture', 'summary': 'Fixture', 'notes': '',
         'status': 'draft', 'observations': [{'id': 'obs-1', 'text': 'Fixture observation'}], 'history': []}
    ]}}
    server.write_json(server.job_path(job_id), job)
    before = server.job_path(job_id).read_bytes()
    handler = object.__new__(server.Handler)
    handler.path = f'/api/jobs/{job_id}/events/event-001'
    handler.read_json = lambda: {'observations': [observation]}
    responses = []
    handler.json_response = lambda body, status=200: responses.append((status, body))
    handler.do_PATCH()
    assert responses[0][0] == 400
    assert 'Invalid observation' in responses[0][1]['error']
    assert server.job_path(job_id).read_bytes() == before


def test_html_export_embeds_evidence_bytes_and_escapes_notes(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'RUNTIME', tmp_path)
    job_id = 'ffffffffffffffff'
    photo = tmp_path/'jobs'/job_id/'frames'/'fixture.jpg'
    photo.parent.mkdir(parents=True)
    # Transport sentinel only, not an inference image or image-decoding test.
    payload = b'fixture-image-bytes'
    photo.write_bytes(payload)
    result = {'title': 'Fixture review', 'source_label': 'Test fixture', 'generated_at': 'fixture',
              'source_sha256': 'fixture', 'hardware_label': 'test', 'events': [
        {'status': 'draft', 'start': 0, 'end': 1, 'title': 'Fixture', 'summary': 'Fixture',
         'observations': [{'id': 'obs-1', 'time': 0.5, 'text': 'Fixture observation',
                           'frame_url': f'/media/jobs/{job_id}/frames/fixture.jpg'}],
         'uncertainties': [], 'review_questions': [], 'notes': '<script>bad()</script>', 'history': []}
    ]}
    html = server.export_html(result)
    assert 'data:image/jpeg;base64,'+base64.b64encode(payload).decode() in html
    assert 'src="/media/' not in html
    assert '<script>bad()</script>' not in html
    assert '&lt;script&gt;bad()&lt;/script&gt;' in html


def test_numeric_upload_id_returns_400_without_starting_job():
    handler = object.__new__(server.Handler)
    handler.path = '/api/jobs'
    handler.read_json = lambda: {'upload_id': 1111111111111111}
    responses = []
    handler.json_response = lambda body, status=200: responses.append((status, body))
    handler.do_POST()
    assert responses[0][0] == 400


@pytest.mark.parametrize('notes', ['', '   '])
def test_dismissal_requires_reason_without_mutating_saved_job(tmp_path, monkeypatch, notes):
    monkeypatch.setattr(server, 'RUNTIME', tmp_path)
    job_id = 'eeeeeeeeeeeeeeee'
    job = {'id': job_id, 'status': 'completed', 'result': {'review_status': 'unreviewed', 'events': [
        {'id': 'event-001', 'title': 'Fixture', 'summary': 'Fixture', 'notes': '',
         'status': 'draft', 'observations': [], 'history': []}
    ]}}
    server.write_json(server.job_path(job_id), job)
    original = server.job_path(job_id).read_bytes()
    handler = object.__new__(server.Handler)
    handler.path = f'/api/jobs/{job_id}/events/event-001'
    handler.read_json = lambda: {'status': 'dismissed', 'reviewer_notes': notes}
    responses = []
    handler.json_response = lambda body, status=200: responses.append((status, body))
    handler.do_PATCH()
    assert responses[0][0] == 400
    assert server.job_path(job_id).read_bytes() == original
    handler.read_json = lambda: {'status': 'dismissed', 'reviewer_notes': 'Fixture reason: irrelevant interaction.'}
    handler.do_PATCH()
    assert responses[-1][0] == 200
    saved = server.read_job(job_id)['result']
    assert saved['review_status'] == 'reviewed'
    assert saved['events'][0]['status'] == 'dismissed'
    assert saved['events'][0]['history'][0]['previous_status'] == 'draft'
    assert saved['events'][0]['notes'] == 'Fixture reason: irrelevant interaction.'


def test_json_export_retains_generation_provenance_and_removes_machine_image_path():
    result = {'events': [{'generation_provenance': {
        'original_input_observations': [{'id': 'obs-1', 'text': 'Original fixture.'}],
        'raw_model_output': {'model_sha256': 'a'*64, 'draft': {'summary': 'Fixture draft.'}},
        'automatic_check_errors': [], 'original_accepted_draft': {'summary': 'Fixture draft.'}},
        'vision_record': {'evidence': [{'image_path': '/private/runtime/frame.jpg'}]}}]}
    exported = server.export_record(result)
    assert exported['events'][0]['generation_provenance'] == result['events'][0]['generation_provenance']
    assert exported['events'][0]['vision_record']['evidence'] == [{'image_file': 'frame.jpg'}]
    assert result['events'][0]['vision_record']['evidence'][0]['image_path'] == '/private/runtime/frame.jpg'


@pytest.mark.parametrize('profile_id, confidence, edge', [('broader', .35, 800), ('stricter', .65, 512)])
def test_queued_settings_survive_config_edit_and_export_keeps_run_metadata(tmp_path, monkeypatch, profile_id, confidence, edge):
    root = tmp_path/'source'
    (root/'config').mkdir(parents=True)
    (root/'data'/'samples').mkdir(parents=True)
    (root/'data'/'samples'/'fixture.mp4').write_bytes(b'fixture-video-no-inference')
    config = root/'config'/f'vision-{profile_id}.json'
    profile = {'id': profile_id, 'label': profile_id.title()+' search', 'description': 'Fixture profile'}
    (root/'config'/'search_profiles.json').write_text(json.dumps({'default': profile_id, 'profiles': [{**profile, 'settings_file': config.name}]}))
    original = {'confidence': confidence, 'processor_size': {'shortest_edge': edge, 'longest_edge': 853 if edge == 512 else 1333}}
    config.write_text(json.dumps(original))
    monkeypatch.setattr(server, 'ROOT', root)
    monkeypatch.setattr(server, 'RUNTIME', tmp_path/'runtime')
    monkeypatch.setattr(server, 'samples', lambda: [{'id': 'fixture', 'path': 'fixture.mp4',
                       'name': 'Fixture', 'source_label': 'Test fixture'}])
    queued = []
    class CapturingPool:
        def submit(self, function, *args):
            queued.append((function, args))
    monkeypatch.setattr(server, 'POOL', CapturingPool())
    handler = object.__new__(server.Handler)
    handler.path = '/api/jobs'
    handler.read_json = lambda: {'sample_id': 'fixture', 'search_profile': profile_id}
    responses = []
    handler.json_response = lambda body, status=200: responses.append((status, body))
    handler.do_POST()
    assert responses[0][0] == 202
    job_id = responses[0][1]['id']
    assert server.read_job(job_id)['requested_vision_settings'] == original
    # Configuration changes while a worker is occupied must not affect this queued job.
    config.write_text(json.dumps({'confidence': .35, 'processor_size': {'shortest_edge': 800, 'longest_edge': 1333}}))
    responses[0][1]['requested_vision_settings']['processor_size']['shortest_edge'] = 1
    function, args = queued[0]
    assert args[3] == original  # Nested settings were copied, not aliased to the response.
    captured = []
    implementation = {'vision_sha256': 'f'*64}
    class FakeProcess:
        pid = 123
        returncode = 0
        def __init__(self, command, **kwargs):
            request = json.loads(Path(command[2]).read_text())
            captured.append(request)
            server.write_json(Path(command[3]), {
                'video': {'duration_s': 1}, 'model': {'id': 'fixture'},
                'settings': request['settings'], 'implementation': implementation, 'events': []})
        def poll(self):
            return 0
    monkeypatch.setattr(server.subprocess, 'Popen', FakeProcess)
    monkeypatch.setattr(server, 'normalize_vision', lambda *args: [])
    function(*args)
    assert captured[0]['settings']['confidence'] == confidence
    assert captured[0]['settings']['processor_size'] == original['processor_size']
    assert 'output_dir' not in args[3]
    job = server.read_job(job_id)
    assert job['status'] == 'completed'
    exported = server.export_record(job['result'])
    assert exported['search_profile'] == profile
    assert exported['requested_vision_settings'] == original
    assert exported['vision_implementation'] == implementation
    assert exported['vision_settings'] == original
