'use strict';
const $ = (id) => document.getElementById(id);
const state = {
  job: null,
  result: null,
  event: null,
  busy: false,
  dirty: false,
  request: 0,
  samples: [],
  sourceSelection: '',
  profiles: [],
};
const formatTime = (value) => {
  const s = Math.max(0, Number(value) || 0);
  return `${Math.floor(s / 60)
    .toString()
    .padStart(2, '0')}:${Math.floor(s % 60)
    .toString()
    .padStart(2, '0')}`;
};
function element(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function notice(text, type = '') {
  $('notice').textContent = text;
  $('notice').className = `notice ${type}`;
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail;
    try {
      const body = await response.json();
      detail = body.error || body.detail;
    } catch {}
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status}).`);
  }
  return response.json();
}
function busy(value) {
  state.busy = value;
  $('analyze').disabled = value || !$('sample').value || !$('search-profile').value;
  $('file').disabled = value || !$('search-profile').value;
  $('search-profile').disabled = value || !state.profiles.length;
  $('sample').disabled = value;
  $('progress-wrap').hidden = !value;
  $('analyze').textContent = value ? 'Analyzing recording…' : 'Review recording →';
}
async function loadProfiles() {
  try {
    const data = await api('/api/search-profiles');
    state.profiles = (data.profiles || []).filter((p) =>
      ['broader', 'fast', 'stricter'].includes(p.id),
    );
    if (!state.profiles.length) throw new Error('No search approaches available.');
    $('search-profile').replaceChildren(...state.profiles.map((p) => new Option(p.label, p.id)));
    $('search-profile').value = state.profiles.some((p) => p.id === data.default)
      ? data.default
      : state.profiles[0].id;
    profileHelp();
  } catch {
    $('profile-help').textContent =
      'Search approaches could not be loaded. Reload to start a new review.';
  }
  busy(false);
}
function profileHelp() {
  const profile = state.profiles.find((p) => p.id === $('search-profile').value);
  $('profile-help').textContent = profile
    ? `For the next review: ${profile.description}`
    : 'Choose a search approach for the next review.';
}
async function init() {
  await loadProfiles();
  try {
    const data = await api('/api/samples');
    const samples = Array.isArray(data) ? data : data.samples || [];
    state.samples = samples;
    $('sample').replaceChildren(new Option('Choose a sample recording', ''));
    samples.forEach((s) => $('sample').add(new Option(s.name || s.title || s.id, s.id)));
    if (samples.length) {
      $('sample').selectedIndex = 1;
      $('analyze').disabled = !$('search-profile').value;
      notice(
        'Sample recordings are synthetic warehouse scenes. Upload your own footage to review another recording.',
      );
    } else notice('Upload a recording to start. No bundled samples are available.');
    await restoreReview();
    if (!state.result) previewSource();
    if (state.restoreError) notice(state.restoreError, 'error');
  } catch (error) {
    $('sample').replaceChildren(new Option('Sample recordings unavailable', ''));
    notice(
      `Could not connect: ${error.message} Start the local server and reload this page.`,
      'error',
    );
  }
}
async function restoreReview() {
  const requested = new URLSearchParams(location.search).get('review');
  let id = requested;
  try {
    if (!id) id = localStorage.getItem('tracebrief.lastReview');
  } catch {}
  if (!id) return;
  try {
    if (!/^[a-f0-9]{16}$/.test(id)) throw new Error('Invalid saved review identifier.');
    const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
    if (job.status !== 'completed' || !job.result)
      throw new Error('The saved review has not completed.');
    state.job = id;
    state.result = job.result;
    renderResult();
    try {
      localStorage.setItem('tracebrief.lastReview', id);
    } catch {}
    notice(
      requested ? 'Saved review loaded.' : 'Your most recent saved review has been restored.',
      'success',
    );
  } catch {
    if (requested)
      state.restoreError =
        'This saved review is unavailable. Choose a recording to start a new review.';
    else
      try {
        localStorage.removeItem('tracebrief.lastReview');
      } catch {}
  }
}
async function start(payload) {
  if (state.busy || !$('search-profile').value) return;
  payload = { ...payload, search_profile: $('search-profile').value };
  if (state.dirty && !confirm('Start a new review without saving your current edits?')) return;
  busy(true);
  state.request += 1;
  const request = state.request;
  notice('Preparing recording for local analysis…');
  $('progress').value = 0;
  try {
    const job = await api('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    state.job = job.id;
    await poll(job.id, request);
  } catch (error) {
    notice(`Analysis could not finish: ${error.message}`, 'error');
    busy(false);
  }
}
const METHOD_LABELS = {
  'local-qwen-draft': 'Local Qwen draft',
  'fixed-template': 'Fixed template',
  'fixed-template-after-local-rejection': 'Fixed template (model draft rejected by source checks)',
};
function methodLabel(id) {
  return METHOD_LABELS[id] || id || 'Not recorded';
}
function plural(n, one, many = `${one}s`) {
  return `${n} ${n === 1 ? one : many}`;
}
async function poll(id, request) {
  if (request !== state.request) return;
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
    $('progress').value = Number(job.progress) || 0;
    notice(job.message || 'Analyzing the recording locally…');
    if (job.status === 'completed' || job.status === 'complete') {
      if (!job.result) throw new Error('Analysis returned no review record.');
      state.result = job.result;
      state.dirty = false;
      try {
        localStorage.setItem('tracebrief.lastReview', id);
      } catch {}
      renderResult();
      busy(false);
      notice(
        `${plural(state.result.events?.length || 0, 'candidate moment')} found. Inspect the evidence before marking a brief reviewed.`,
        'success',
      );
      return;
    }
    if (job.status === 'failed') throw new Error(job.error || job.message || 'Processing failed.');
    setTimeout(() => poll(id, request), 1000);
  } catch (error) {
    busy(false);
    notice(`Analysis could not finish: ${error.message}`, 'error');
  }
}
function renderResult() {
  const r = state.result;
  $('recorded-profile').hidden = false;
  $('recorded-profile').textContent =
    `This review: ${r.search_profile?.label || 'Earlier analysis'}`;
  if (r.source?.id && state.samples.some((s) => s.id === r.source.id))
    $('sample').value = r.source.id;
  state.sourceSelection = $('sample').value;
  r.events = r.events || [];
  if (r.video_url) $('video').src = r.video_url;
  else $('video').removeAttribute('src');
  $('video-empty').hidden = Boolean(r.video_url);
  $('recording-title').textContent = r.title || 'Recording';
  $('source-badge').textContent = r.source_label || 'Uploaded recording';
  $('source-badge').hidden = false;
  $('recording-type').textContent = /synthetic|simulat/i.test(r.source_label || '')
    ? 'Synthetic footage'
    : 'Uploaded footage';
  $('duration').textContent = formatTime(r.duration);
  $('track-end').textContent = formatTime(r.duration);
  $('method-label').textContent =
    [r.method_label, r.generation_method && methodLabel(r.generation_method)]
      .filter(Boolean)
      .join(' · ') || 'Local analysis';
  $('event-count').textContent = r.events.length;
  $('export').disabled = false;
  renderEvents();
  if (r.events.length) selectEvent(r.events[0].id, false);
  else {
    state.event = null;
    $('brief-form').hidden = true;
    $('brief-empty').hidden = false;
    $('brief-status').textContent = 'No candidates';
    $('frames').replaceChildren(
      element('div', 'frame-placeholder', 'No candidate evidence frames.'),
    );
    $('evidence-count').textContent = '0 frames';
    $('events').replaceChildren(
      element(
        'div',
        'empty-state',
        'No candidate interactions found. This does not establish that the recording is free of risk. Review the source footage as needed.',
      ),
    );
  }
  updateQueue();
}
function renderEvents() {
  const list = $('events');
  list.replaceChildren();
  $('scrubber').replaceChildren();
  for (const [index, event] of state.result.events.entries()) {
    const card = element('button', `event-card${state.event?.id === event.id ? ' selected' : ''}`);
    card.type = 'button';
    card.setAttribute('aria-label', `${event.title || 'Candidate'} at ${formatTime(event.start)}`);
    card.setAttribute('aria-pressed', String(state.event?.id === event.id));
    const meta = element('div', 'event-meta');
    meta.append(
      element('span', '', `MOMENT ${String(index + 1).padStart(2, '0')}`),
      element('span', '', formatTime(event.start)),
    );
    card.append(
      meta,
      element('h3', '', event.title || 'Candidate interaction'),
      element('p', '', event.summary || 'Inspect the linked source evidence.'),
    );
    const look = lookSummary(event.visual_check);
    if (look)
      card.append(
        element('p', `look-chip${event.visual_check.status === 'confirmed' ? '' : ' warn'}`, look),
      );
    const bottom = element('div', 'event-bottom');
    bottom.append(
      element(
        'span',
        '',
        event.status === 'dismissed'
          ? '− Dismissed'
          : event.status === 'reviewed'
            ? '✓ Reviewed'
            : '● Needs review',
      ),
      element('span', '', `${plural((event.observations || []).length, 'observation')} →`),
    );
    card.append(bottom);
    card.addEventListener('click', () => selectEvent(event.id));
    list.append(card);
    const marker = element(
      'button',
      `track-event${state.event?.id === event.id ? ' selected' : ''}`,
    );
    const duration = Number(state.result.duration) || 1;
    marker.style.left = `${Math.min(99, (100 * event.start) / duration)}%`;
    marker.style.width = `${Math.max(1, Math.min(100 - (100 * event.start) / duration, (100 * (event.end - event.start)) / duration))}%`;
    marker.title = `${event.title} · ${formatTime(event.start)}`;
    marker.setAttribute('aria-label', marker.title);
    marker.addEventListener('click', () => selectEvent(event.id));
    $('scrubber').append(marker);
  }
}
function lookSummary(check) {
  if (!check || !['confirmed', 'check-suggested'].includes(check.status)) return null;
  return check.status === 'confirmed'
    ? 'Visual check: person and forklift seen'
    : 'Visual check: check for a false candidate';
}
function renderLook(event) {
  const check = event.visual_check;
  const box = $('look-box');
  if (!check) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  const ok = ['confirmed', 'check-suggested'].includes(check.status);
  $('look-verdict').textContent =
    check.status === 'confirmed'
      ? '✓ Both seen'
      : check.status === 'check-suggested'
        ? 'Check candidate'
        : 'Unavailable';
  $('look-verdict').className =
    `pill${check.status === 'confirmed' ? ' reviewed' : check.status === 'check-suggested' ? ' dismissed' : ''}`;
  $('look-description').textContent = ok
    ? check.description
    : check.reason || 'Visual check unavailable.';
  $('look-meta').textContent = ok
    ? `Local vision-language model · key frame ${formatTime(check.time)} · forklift ${check.forklift}, person ${check.person}${check.status === 'check-suggested' ? ' · it did not see both, so check the frame before dismissing' : ''}`
    : '';
  const crop = $('look-crop');
  if (ok && check.crop_url) {
    crop.src = check.crop_url;
    crop.hidden = false;
  } else {
    crop.removeAttribute('src');
    crop.hidden = true;
  }
}
function selectEvent(id, checkDirty = true) {
  if (checkDirty && state.dirty && !confirm('Switch candidates without saving your edits?')) return;
  const event = state.result.events.find((e) => e.id === id);
  if (!event) return;
  state.event = event;
  state.dirty = false;
  renderEvents();
  $('brief-form').hidden = false;
  $('brief-empty').hidden = true;
  $('brief-title').value = event.title || 'Candidate interaction';
  $('summary').value = event.summary || '';
  renderLook(event);
  $('review-notes').value = event.reviewer_notes || '';
  $('event-range').textContent =
    `${formatTime(event.start)}–${formatTime(event.end)} · Candidate ${state.result.events.indexOf(event) + 1}`;
  status(event.status);
  $('observations').replaceChildren();
  $('frames').replaceChildren();
  for (const [index, observation] of (event.observations || []).entries()) {
    const row = element('div', 'observation');
    const timestamp = element('button', 'timestamp', formatTime(observation.time));
    timestamp.type = 'button';
    timestamp.setAttribute('aria-label', `View evidence at ${formatTime(observation.time)}`);
    timestamp.addEventListener('click', () => seek(observation.time));
    const input = element('textarea');
    input.value = observation.text || '';
    input.rows = 2;
    input.dataset.index = index;
    input.setAttribute('aria-label', `Observation ${index + 1}`);
    input.addEventListener('input', markDirty);
    row.append(timestamp, input);
    $('observations').append(row);
    if (index < 3) {
      const frame = element('button', 'frame');
      frame.type = 'button';
      frame.setAttribute('aria-label', `Inspect evidence frame at ${formatTime(observation.time)}`);
      if (observation.frame_url) {
        const img = element('img');
        img.src = observation.frame_url;
        img.alt = `Source frame at ${formatTime(observation.time)}`;
        img.addEventListener('error', () => {
          img.remove();
          frame.prepend(element('span', '', 'Frame unavailable'));
        });
        frame.append(img);
      }
      frame.append(element('span', '', `↗ ${formatTime(observation.time)}`));
      frame.addEventListener('click', () => seek(observation.time));
      $('frames').append(frame);
    }
  }
  $('evidence-count').textContent = plural((event.observations || []).length, 'linked observation');
  if (!event.observations?.length)
    $('frames').append(element('div', 'frame-placeholder', 'No linked frames for this candidate.'));
  $('uncertainties').replaceChildren();
  const uncertainties = event.uncertainties?.length
    ? event.uncertainties
    : ['Confirm the visible sequence against the source footage.'];
  uncertainties.forEach((item) =>
    $('uncertainties').append(
      element('li', '', typeof item === 'string' ? item : item.text || String(item)),
    ),
  );
  const warning = event.generation_warning || state.result.generation_warning;
  if (warning) $('uncertainties').append(element('li', '', warning));
  $('review-questions').replaceChildren();
  const questions = event.review_questions || [];
  questions.forEach((q) => $('review-questions').append(element('li', '', q)));
  $('questions-box').hidden = !questions.length;
  $('generation-label').textContent =
    `Draft method: ${methodLabel(event.generation_method || state.result.generation_method)}`;
  seek(event.start);
}
function seek(time) {
  const video = $('video');
  if (!video.src) return;
  const target = Number(time) || 0;
  if (video.readyState >= 1)
    video.currentTime = Math.min(target, Number.isFinite(video.duration) ? video.duration : target);
  else
    video.addEventListener(
      'loadedmetadata',
      () => {
        video.currentTime = Math.min(target, video.duration);
      },
      { once: true },
    );
  video.pause();
}
function status(value) {
  const reviewed = value === 'reviewed';
  const dismissed = value === 'dismissed';
  $('brief-status').textContent = dismissed ? 'Dismissed' : reviewed ? '✓ Reviewed' : 'Draft';
  $('brief-status').className = `pill${dismissed ? ' dismissed' : reviewed ? ' reviewed' : ''}`;
  $('approve').textContent = reviewed ? 'Reviewed ✓' : 'Mark reviewed ✓';
  $('dismiss').textContent = dismissed ? 'Update dismissal' : 'Dismiss candidate';
  $('save-status').textContent = dismissed
    ? 'Dismissed · reason retained. Save as draft or mark reviewed to reopen.'
    : reviewed
      ? 'Reviewed · the record remains editable.'
      : 'Draft · check the evidence before marking this record reviewed.';
}
function markDirty() {
  state.dirty = true;
  $('save-status').textContent = 'Unsaved changes · save the draft or mark it reviewed.';
  $('brief-status').textContent = 'Edited draft';
  $('brief-status').className = 'pill';
}
async function save(target = 'draft') {
  if (!state.event) return;
  if (target === 'dismissed' && !$('review-notes').value.trim()) {
    $('save-status').textContent =
      'Add a reason in reviewer notes before dismissing this candidate.';
    $('review-notes').focus();
    return;
  }
  const event = {
    ...state.event,
    title: $('brief-title').value,
    summary: $('summary').value,
    reviewer_notes: $('review-notes').value,
    status: target,
    observations: state.event.observations.map((o, i) => ({
      ...o,
      text: $('observations').querySelector(`[data-index="${i}"]`).value,
    })),
  };
  $('save').disabled = true;
  $('approve').disabled = true;
  $('dismiss').disabled = true;
  try {
    await api(`/api/jobs/${encodeURIComponent(state.job)}/events/${encodeURIComponent(event.id)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(event),
    });
    Object.assign(state.event, event);
    state.dirty = false;
    status(event.status);
    $('save-status').textContent =
      target === 'dismissed'
        ? 'Candidate dismissed. Your reason and evidence are retained in the export.'
        : target === 'reviewed'
          ? 'Saved and marked reviewed. Your edits are included in the export.'
          : 'Draft saved. Review status is included in the export.';
    renderEvents();
    updateQueue();
  } catch (error) {
    $('save-status').textContent = `Could not save: ${error.message}`;
    notice('Your edits are still on this page. Retry saving before leaving.', 'error');
  } finally {
    $('save').disabled = false;
    $('approve').disabled = false;
    $('dismiss').disabled = false;
  }
}
function updateQueue() {
  const events = state.result.events;
  const count = events.filter((e) =>
    ['reviewed', 'dismissed', 'approved'].includes(e.status),
  ).length;
  $('queue-status').textContent = `${count} of ${plural(events.length, 'candidate')} resolved`;
}
function previewSource() {
  $('recorded-profile').hidden = true;
  const sample = state.samples.find((s) => s.id === $('sample').value);
  state.sourceSelection = $('sample').value;
  state.job = null;
  state.result = null;
  state.event = null;
  state.dirty = false;
  $('export').disabled = true;
  $('brief-form').hidden = true;
  $('brief-empty').hidden = false;
  $('brief-status').textContent = 'No selection';
  $('brief-status').className = 'pill';
  $('events').replaceChildren(
    element('div', 'empty-state', 'Run a review to find candidate moments in this recording.'),
  );
  $('scrubber').replaceChildren();
  $('frames').replaceChildren(
    element('div', 'frame-placeholder', 'Linked frames appear here after analysis.'),
  );
  $('event-count').textContent = '0';
  $('evidence-count').textContent = 'Awaiting analysis';
  $('queue-status').textContent = 'Waiting for analysis';
  if (sample?.video_url) {
    $('video').src = sample.video_url;
    $('video-empty').hidden = true;
    $('source-badge').hidden = false;
    $('source-badge').textContent = sample.source_label || 'Sample recording';
    $('recording-title').textContent = sample.name || sample.id;
    $('recording-type').textContent = /synthetic|simulat/i.test(sample.source_label || '')
      ? 'Synthetic footage'
      : 'Sample footage';
    $('duration').textContent = formatTime(sample.duration);
    $('track-end').textContent = formatTime(sample.duration);
    notice(
      sample.description || 'Preview the recording, then run a review to find candidate moments.',
    );
  } else {
    $('video').removeAttribute('src');
    $('video-empty').hidden = false;
    $('source-badge').hidden = true;
    $('recording-title').textContent = 'No recording selected';
    $('recording-type').textContent = 'Awaiting footage';
    $('duration').textContent = '—';
    $('track-end').textContent = '—';
  }
}
$('search-profile').addEventListener('change', profileHelp);
$('sample').addEventListener('change', () => {
  if (state.dirty && !confirm('Change recording without saving your current edits?')) {
    $('sample').value = state.sourceSelection;
    return;
  }
  previewSource();
  $('analyze').disabled = !$('sample').value || state.busy || !$('search-profile').value;
});
$('analyze').addEventListener('click', () => start({ sample_id: $('sample').value }));
$('file').addEventListener('change', async () => {
  const file = $('file').files[0];
  if (!file) return;
  if (state.dirty && !confirm('Upload another recording without saving your edits?')) {
    $('file').value = '';
    return;
  }
  busy(true);
  notice(`Uploading ${file.name} to the local workspace…`);
  try {
    const uploaded = await api('/api/uploads', {
      method: 'POST',
      headers: {
        'Content-Type': file.type || 'application/octet-stream',
        'X-Filename': encodeURIComponent(file.name),
      },
      body: file,
    });
    busy(false);
    state.dirty = false;
    await start({ upload_id: uploaded.id });
  } catch (error) {
    notice(`Upload failed: ${error.message}`, 'error');
    busy(false);
  } finally {
    $('file').value = '';
  }
});
$('brief-title').addEventListener('input', markDirty);
$('summary').addEventListener('input', markDirty);
$('review-notes').addEventListener('input', markDirty);
$('save').addEventListener('click', () => save('draft'));
$('approve').addEventListener('click', () => save('reviewed'));
$('dismiss').addEventListener('click', () => save('dismissed'));
$('brief-form').addEventListener('submit', (event) => event.preventDefault());
$('export').addEventListener('click', () => {
  if (state.dirty) {
    notice('Save your current changes before exporting the review.', 'error');
    $('save').focus();
    return;
  }
  const base = `/api/jobs/${encodeURIComponent(state.job)}/export`;
  $('export-html').href = `${base}?format=html`;
  $('export-json').href = `${base}?format=json`;
  $('export-dialog').showModal();
});
$('close-export').addEventListener('click', () => $('export-dialog').close());
$('export-dialog').addEventListener('click', (event) => {
  if (event.target === $('export-dialog')) {
    const box = $('export-dialog').getBoundingClientRect();
    if (
      event.clientX < box.left ||
      event.clientX > box.right ||
      event.clientY < box.top ||
      event.clientY > box.bottom
    )
      $('export-dialog').close();
  }
});
window.addEventListener('beforeunload', (event) => {
  if (state.dirty) {
    event.preventDefault();
    event.returnValue = '';
  }
});
$('video').addEventListener('error', () => {
  if (state.result)
    notice(
      'This recording could not be played in your browser. Evidence frames and the review record remain available.',
      'error',
    );
});
init();
