/* H3 Flash WebUI — vanilla JS, no build step. */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const FPS = 24;

  const state = {
    config: null,
    status: null,
    presetId: 'balanced',
    jobId: null,
    pollTimer: null,
    anchors: { first: null, last: null },
    refs: [],
    uploadTarget: null,
    busy: false
  };

  /* ------------------------------------------------------------------ api */

  async function api(path, options) {
    const response = await fetch(path, options);
    const text = await response.text();
    let body = null;
    if (text) {
      try { body = JSON.parse(text); } catch { body = text; }
    }
    if (!response.ok) {
      const message = (body && body.error) || `HTTP ${response.status}`;
      const error = new Error(message);
      error.status = response.status;
      error.body = body;
      throw error;
    }
    return body;
  }

  const postJson = (path, body) => api(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {})
  });

  function toast(message, kind) {
    const node = $('toast');
    node.textContent = message;
    node.className = 'toast' + (kind ? ' ' + kind : '');
    node.hidden = false;
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => { node.hidden = true; }, kind === 'err' ? 7000 : 3200);
  }

  function showAlert(message) {
    const node = $('alert');
    if (!message) { node.hidden = true; node.textContent = ''; return; }
    node.textContent = message;
    node.hidden = false;
  }

  /* --------------------------------------------------------------- config */

  async function init() {
    try {
      state.config = await api('/api/config');
    } catch (error) {
      showAlert('无法连接后端：' + error.message);
      return;
    }

    const config = state.config;
    const advanced = !!config.advancedEnabled;

    $('editionBadge').textContent = advanced ? '高级版' : '通用版';
    $('editionBadge').className = 'badge' + (advanced ? ' adv' : '');
    document.title = `H3 Flash · ${advanced ? '高级版' : '通用版'}`;
    $('advancedPanel').hidden = !advanced;
    $('modelPath').textContent = config.modelDir || '模型未加载';

    renderPresets();
    renderResolutions();
    renderDurations();
    renderPacks();

    if (advanced) {
      $('advancedPanel').open = true;
      $('advSteps').value = 8;
      $('advDitLayers').value = 50;
      $('advDenoiseReuse').value = 1;
      $('advCoreReuse').value = 1;
      $('advWidth').value = 864;
      $('advHeight').value = 480;
      $('advFrames').value = 56;
      $('advRenderWidth').value = 0;
      $('advRenderHeight').value = 0;
      $('refHint').textContent = config.packs.some((p) => p.hasRef2Va)
        ? '按顺序生效，需要 Ref2VA 权重'
        : '当前模型包不含 Ref2VA，参考素材会被忽略';
    }

    selectPreset('balanced', true);
    loadHistory();
    loadStatus();
  }

  async function loadStatus() {
    try {
      const status = await api('/api/status');
      state.status = status;

      let device = {};
      try { device = JSON.parse(status.deviceInfoJson || '{}'); } catch { /* ignore */ }

      const name = device.name || device.device_name;
      $('gpuName').textContent = name ? `${name}` : '未检测到设备';
      $('gpuName').className = 'chip ' + (name ? 'chip-ok' : 'chip-warn');

      const mismatched = device.arch_matches_build === false;
      $('archWarn').hidden = !mismatched;
      if (mismatched) {
        $('archWarn').textContent = `算力 ${device.compute_capability} ≠ 编译 ${device.build_arch}`;
        $('archWarn').title = '内核会静默返回 0，请用 -DH3_CUDA_ARCH=当前算力 重新编译';
      }

      $('modelPath').textContent = status.modelDir || '模型未加载';
      $('modelPath').title = status.modelDir || '';

      if (!status.modelLoaded && status.modelError) {
        showAlert('模型加载失败：' + status.modelError);
      }
    } catch (error) {
      $('gpuName').textContent = '设备信息不可用';
      $('gpuName').className = 'chip chip-warn';
      if (error.message) console.warn(error);
    }
  }

  /* -------------------------------------------------------------- presets */

  function renderPresets() {
    const container = $('presets');
    container.innerHTML = '';
    for (const preset of state.config.presets) {
      const button = document.createElement('button');
      button.className = 'preset';
      button.dataset.id = preset.id;
      button.dataset.desc = preset.description;
      button.innerHTML = `<b></b><small></small>`;
      button.querySelector('b').textContent = preset.name;
      button.querySelector('small').textContent = preset.description;
      button.addEventListener('click', () => selectPreset(preset.id, false));
      container.appendChild(button);
    }
  }

  function selectPreset(id, silent) {
    const preset = state.config.presets.find((p) => p.id === id) || state.config.presets[0];
    if (!preset) return;
    state.presetId = preset.id;

    for (const node of $('presets').children) {
      node.classList.toggle('active', node.dataset.id === preset.id);
      node.querySelector('small').textContent = node.dataset.desc;
    }

    // The preset carries the canvas too; the user may still override it.
    setResolution(preset.width, preset.height);
    setDuration(preset.frames);

    if (state.config.advancedEnabled) {
      $('advWidth').value = preset.width;
      $('advHeight').value = preset.height;
      $('advFrames').value = preset.frames;
      $('advSteps').value = preset.steps;
      $('advPdd8').checked = !!preset.pdd8;
      $('advDitLayers').value = preset.ditLayers;
      $('advDenoiseReuse').value = preset.denoiseReuse;
    }
    if (!silent) markPresetDirty();
  }

  function markPresetDirty() {
    const width = Number($('resolution').value.split('x')[0]);
    const height = Number($('resolution').value.split('x')[1]);
    const frames = Number($('duration').value);
    const preset = state.config.presets.find((p) => p.id === state.presetId);
    if (!preset) return;
    const changed = width !== preset.width || height !== preset.height || frames !== preset.frames;
    for (const node of $('presets').children) {
      if (node.dataset.id !== state.presetId) continue;
      node.querySelector('small').textContent = changed ? '画布已自定义' : node.dataset.desc;
    }
  }

  /* ------------------------------------------------------------ selectors */

  function renderResolutions() {
    const select = $('resolution');
    select.innerHTML = '';
    for (const item of state.config.resolutions) {
      const option = document.createElement('option');
      option.value = `${item.width}x${item.height}`;
      option.textContent = item.label;
      select.appendChild(option);
    }
    select.addEventListener('change', () => {
      if (state.config.advancedEnabled) {
        const [w, h] = select.value.split('x');
        $('advWidth').value = w;
        $('advHeight').value = h;
      }
      markPresetDirty();
    });
  }

  function setResolution(width, height) {
    const select = $('resolution');
    const value = `${width}x${height}`;
    if (![...select.options].some((o) => o.value === value)) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = `${width} × ${height}`;
      select.appendChild(option);
    }
    select.value = value;
  }

  function renderDurations() {
    const select = $('duration');
    select.innerHTML = '';
    for (const frames of state.config.frameChoices) {
      const option = document.createElement('option');
      option.value = String(frames);
      option.textContent = `${frames} 帧 · ${(frames / FPS).toFixed(2)} 秒`;
      select.appendChild(option);
    }
    select.addEventListener('change', () => {
      if (state.config.advancedEnabled) $('advFrames').value = select.value;
      markPresetDirty();
    });
  }

  function setDuration(frames) {
    const select = $('duration');
    const value = String(frames);
    if (![...select.options].some((o) => o.value === value)) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = `${frames} 帧 · ${(frames / FPS).toFixed(2)} 秒`;
      select.appendChild(option);
    }
    select.value = value;
  }

  function renderPacks() {
    const select = $('packSelect');
    select.innerHTML = '';
    const packs = state.config.packs || [];
    if (packs.length === 0) {
      const option = document.createElement('option');
      option.textContent = '未发现模型包';
      select.appendChild(option);
      $('btnSwitchPack').disabled = true;
      return;
    }
    for (const pack of packs) {
      const option = document.createElement('option');
      option.value = pack.path;
      option.textContent = pack.name + (pack.hasRef2Va ? ' · Ref2VA' : '') + (pack.active ? ' (当前)' : '');
      select.appendChild(option);
    }
    if (state.config.modelDir) select.value = state.config.modelDir;
  }

  /* --------------------------------------------------------------- upload */

  function pickFile(target) {
    state.uploadTarget = target;
    const picker = $('filePicker');
    picker.value = '';
    picker.click();
  }

  async function uploadFile(file) {
    const form = new FormData();
    form.append('file', file, file.name);
    const response = await fetch('/api/upload', { method: 'POST', body: form });
    const body = await response.json().catch(() => null);
    if (!response.ok) throw new Error((body && body.error) || `HTTP ${response.status}`);
    return body;
  }

  async function handlePickedFile(file) {
    if (!file) return;
    const target = state.uploadTarget;
    state.uploadTarget = null;
    if (!target) return;
    try {
      toast('上传中…');
      const result = await uploadFile(file);
      if (target.type === 'anchor') {
        state.anchors[target.slot] = result.url;
        renderAnchor(target.slot, result.url);
      } else if (target.type === 'ref') {
        const row = state.refs[target.index];
        if (row) { row.path = result.url; row.name = result.name; renderRefs(); }
      }
      toast('已上传：' + result.name, 'ok');
    } catch (error) {
      toast('上传失败：' + error.message, 'err');
    }
  }

  function renderAnchor(slot, url) {
    const zone = document.querySelector(`.dropzone[data-slot="${slot}"]`);
    if (!zone) return;
    const img = zone.querySelector('img');
    const clear = zone.querySelector('.dz-clear');
    const text = zone.querySelector('.dz-text');
    if (url) {
      img.src = url;
      img.hidden = false;
      clear.hidden = false;
      text.hidden = true;
    } else {
      img.hidden = true;
      img.removeAttribute('src');
      clear.hidden = true;
      text.hidden = false;
    }
  }

  function bindDropzones() {
    for (const zone of document.querySelectorAll('.dropzone')) {
      const slot = zone.dataset.slot;
      zone.addEventListener('click', (event) => {
        if (event.target.classList.contains('dz-clear')) return;
        pickFile({ type: 'anchor', slot });
      });
      zone.querySelector('.dz-clear').addEventListener('click', (event) => {
        event.stopPropagation();
        state.anchors[slot] = null;
        renderAnchor(slot, null);
      });
      zone.addEventListener('dragover', (event) => {
        event.preventDefault();
        zone.classList.add('drag');
      });
      zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
      zone.addEventListener('drop', async (event) => {
        event.preventDefault();
        zone.classList.remove('drag');
        const file = event.dataTransfer.files[0];
        if (!file) return;
        state.uploadTarget = { type: 'anchor', slot };
        await handlePickedFile(file);
      });
    }

    $('filePicker').addEventListener('change', async (event) => {
      await handlePickedFile(event.target.files[0]);
    });
  }

  /* ----------------------------------------------------------- references */

  function renderRefs() {
    const container = $('refList');
    container.innerHTML = '';
    state.refs.forEach((row, index) => {
      const node = document.createElement('div');
      node.className = 'refrow';

      const kind = document.createElement('select');
      for (const [value, label] of [['image', '图片'], ['video', '视频'], ['audio', '音频'], ['video_audio', '视频+音轨']]) {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = label;
        kind.appendChild(option);
      }
      kind.value = row.kind;
      kind.addEventListener('change', () => { row.kind = kind.value; });

      const path = document.createElement('input');
      path.type = 'text';
      path.readOnly = true;
      path.placeholder = '点击右侧选择文件';
      path.value = row.name || row.path || '';
      path.addEventListener('click', () => pickFile({ type: 'ref', index }));

      const del = document.createElement('button');
      del.className = 'del';
      del.textContent = '×';
      del.title = '移除';
      del.addEventListener('click', () => { state.refs.splice(index, 1); renderRefs(); });

      node.append(kind, path, del);
      container.appendChild(node);
    });
  }

  /* ------------------------------------------------------------- generate */

  function num(id) {
    const value = $(id).value;
    return value === '' ? null : Number(value);
  }

  function buildRequest() {
    const [width, height] = $('resolution').value.split('x').map(Number);
    const frames = Number($('duration').value);
    const seed = $('seed').value === '' ? null : Number($('seed').value);
    const advanced = !!state.config.advancedEnabled;

    const request = {
      prompt: $('prompt').value.trim(),
      preset: state.presetId,
      width, height, frames,
      seed,
      livePreview: $('livePreview').checked,
      previewEvery: 4,
      firstFrame: state.anchors.first,
      lastFrame: state.anchors.last
    };

    if (advanced) {
      Object.assign(request, {
        pdd8: $('advPdd8').checked,
        steps: num('advSteps'),
        ditLayers: num('advDitLayers'),
        denoiseReuse: num('advDenoiseReuse'),
        coreReuse: num('advCoreReuse'),
        renderWidth: num('advRenderWidth') || 0,
        renderHeight: num('advRenderHeight') || 0,
        referenceImageSize: $('advRefSize').value || null,
        tokenReduction: $('advTokenReduction').checked,
        solAttn: $('advSolAttn').checked,
        fbc: $('advFbc').checked,
        ssdStreaming: $('advSsd').checked,
        useInt8RowFc2: $('advInt8RowFc2').checked,
        useReferenceRope: $('advRefRope').checked,
        previewDenoise: $('advPreviewDenoise').checked
      });
      if (num('advWidth')) request.width = num('advWidth');
      if (num('advHeight')) request.height = num('advHeight');
      if (num('advFrames')) request.frames = num('advFrames');
      if ($('packSelect').value) request.modelDir = $('packSelect').value;

      const references = state.refs
        .filter((row) => row.path)
        .map((row) => ({ kind: row.kind, path: row.path }));
      if (references.length > 0) request.references = references;
    }

    return request;
  }

  async function generate() {
    if (state.busy) return;
    showAlert(null);

    const request = buildRequest();
    if (!request.prompt) {
      showAlert('请先填写提示词。');
      $('prompt').focus();
      return;
    }

    setBusy(true);
    try {
      const job = await postJson('/api/generate', request);
      state.jobId = job.id;
      resetStage();
      renderJob(job);
      poll();
    } catch (error) {
      setBusy(false);
      showAlert('提交失败：' + error.message);
    }
  }

  function setBusy(busy) {
    state.busy = busy;
    $('btnGenerate').disabled = busy;
    $('btnCancel').hidden = !busy;
    $('btnGenerate').textContent = busy ? '生成中…' : '生成视频';
  }

  function poll() {
    clearTimeout(state.pollTimer);
    state.pollTimer = setTimeout(async () => {
      if (!state.jobId) return;
      try {
        const job = await api('/api/jobs/' + state.jobId);
        renderJob(job);
        if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
          setBusy(false);
          loadHistory();
          return;
        }
        poll();
      } catch (error) {
        setBusy(false);
        showAlert('轮询任务失败：' + error.message);
      }
    }, 550);
  }

  /* --------------------------------------------------------------- render */

  function resetStage() {
    $('placeholder').hidden = true;
    $('resultVideo').hidden = true;
    $('resultVideo').removeAttribute('src');
    $('previewImg').hidden = true;
    $('previewImg').removeAttribute('src');
    $('vbadge').hidden = true;
    $('resultbar').hidden = true;
    $('phasePanel').hidden = true;
    $('rawPanel').hidden = true;
    $('barFill').style.width = '0%';
  }

  function renderJob(job) {
    if (!job) return;

    if (job.status === 'queued') {
      $('phaseText').textContent = job.queuePosition > 0 ? `排队中（第 ${job.queuePosition} 位）` : '排队中';
      $('progressText').textContent = '';
      $('barFill').classList.add('indeterminate');
    } else if (job.status === 'running' || job.status === 'cancelling') {
      const progress = job.progress;
      if (job.status === 'cancelling') {
        // Cancellation lands at the next engine checkpoint, which can be a
        // whole phase away; say so instead of looking hung.
        $('phaseText').textContent = '正在取消（当前阶段收尾后生效）';
        $('barFill').classList.add('indeterminate');
        $('progressText').textContent = progress
          ? `已运行 ${(progress.elapsedMs / 1000).toFixed(1)}s`
          : '';
      } else if (progress && progress.fraction != null) {
        $('phaseText').textContent = progress.phase || '生成中';
        $('barFill').classList.remove('indeterminate');
        $('barFill').style.width = (progress.fraction * 100).toFixed(1) + '%';
        $('progressText').textContent =
          `${progress.completed} / ${progress.total} · ${(progress.elapsedMs / 1000).toFixed(1)}s`;
      } else {
        $('phaseText').textContent = (progress && progress.phase) || '生成中';
        $('barFill').classList.add('indeterminate');
        $('progressText').textContent = progress
          ? `${(progress.elapsedMs / 1000).toFixed(1)}s`
          : '';
      }
      if (job.previewUrl && job.livePreview) {
        const img = $('previewImg');
        if (img.getAttribute('src') !== job.previewUrl) img.src = job.previewUrl;
        img.hidden = false;
        $('placeholder').hidden = true;
        $('vbadge').hidden = false;
        $('vbadge').textContent = '实时预览';
      }
    } else if (job.status === 'completed') {
      $('barFill').classList.remove('indeterminate');
      $('barFill').style.width = '100%';
      $('phaseText').textContent = '已完成';
      const result = job.result || {};
      $('progressText').textContent = result.elapsedMs
        ? `${(result.elapsedMs / 1000).toFixed(1)}s · ${result.framesPerSecond ? result.framesPerSecond.toFixed(2) : '?'} fps`
        : '';
      showResult(job);
    } else if (job.status === 'failed') {
      $('barFill').classList.remove('indeterminate');
      $('barFill').style.width = '0%';
      $('phaseText').textContent = '失败';
      $('progressText').textContent = '';
      showAlert(job.error || '生成失败。');
    } else if (job.status === 'cancelled') {
      $('barFill').classList.remove('indeterminate');
      $('barFill').style.width = '0%';
      $('phaseText').textContent = '已取消';
      $('progressText').textContent = '';
    }
  }

  function showResult(job) {
    const result = job.result || {};
    const video = $('resultVideo');

    if (job.videoUrl) {
      if (video.getAttribute('src') !== job.videoUrl) {
        video.src = job.videoUrl;
        video.load();
      }
      video.hidden = false;
      if (job.posterUrl) video.poster = job.posterUrl;
      $('previewImg').hidden = true;
      $('placeholder').hidden = true;
      $('vbadge').hidden = true;
      $('btnDownload').href = job.videoUrl;
      $('btnDownload').setAttribute('download', job.id + '.mp4');
    }

    $('resultbar').hidden = false;
    $('chipRes').textContent = `${result.width || '?'} × ${result.height || '?'}`;
    $('chipFrames').textContent = `${result.frames || '?'} 帧 · ${result.fps || FPS} fps`;
    $('chipFps').textContent = result.framesPerSecond
      ? `${result.framesPerSecond.toFixed(2)} fps 生成速度`
      : '—';
    $('chipElapsed').textContent = result.elapsedMs
      ? `${(result.elapsedMs / 1000).toFixed(1)} 秒`
      : '—';
    $('chipSeed').textContent = 'seed ' + (result.seedString || result.seed || '—');

    renderPhases(result.phases || []);

    if (state.config.advancedEnabled && $('advRawJson').checked) {
      $('rawPanel').hidden = false;
      $('rawJson').textContent = JSON.stringify(result, null, 2);
    }
  }

  function renderPhases(phases) {
    if (!phases.length) { $('phasePanel').hidden = true; return; }
    $('phasePanel').hidden = false;
    const list = $('phaseList');
    list.innerHTML = '';

    const total = phases.length > 1 ? phases[phases.length - 1].at_ms - phases[0].at_ms : 1;
    const span = Math.max(total, 1);

    phases.forEach((phase, index) => {
      const next = index + 1 < phases.length ? phases[index + 1].at_ms : phase.at_ms;
      const duration = Math.max(next - phase.at_ms, 0);
      const row = document.createElement('div');
      row.className = 'phaserow';

      const name = document.createElement('span');
      name.textContent = phase.phase;

      const track = document.createElement('div');
      track.className = 'track';
      const fill = document.createElement('i');
      fill.style.width = Math.max((duration / span) * 100, 1.5).toFixed(1) + '%';
      track.appendChild(fill);

      const value = document.createElement('span');
      value.textContent = (duration / 1000).toFixed(2) + 's';
      value.style.textAlign = 'right';

      row.append(name, track, value);
      list.appendChild(row);
    });
  }

  /* -------------------------------------------------------------- history */

  async function loadHistory() {
    try {
      const items = await api('/api/history');
      renderHistory(items);
    } catch (error) {
      console.warn('history unavailable', error);
    }
  }

  function renderHistory(items) {
    const gallery = $('gallery');
    gallery.innerHTML = '';
    $('historyCount').textContent = items.length ? `共 ${items.length} 条` : '';

    if (!items.length) {
      const empty = document.createElement('p');
      empty.className = 'empty';
      empty.textContent = '还没有生成记录';
      gallery.appendChild(empty);
      return;
    }

    for (const item of items) {
      const card = document.createElement('div');
      card.className = 'card';

      const thumb = document.createElement('div');
      thumb.className = 'thumb';
      if (item.posterUrl) thumb.style.backgroundImage = `url("${item.posterUrl}")`;
      else thumb.textContent = item.status === 'completed' ? '▶' : '—';

      const body = document.createElement('div');
      body.className = 'body';
      const prompt = document.createElement('div');
      prompt.className = 'prompt';
      prompt.textContent = item.prompt || '(无提示词)';
      prompt.title = item.prompt || '';

      const foot = document.createElement('div');
      foot.className = 'foot';
      const meta = document.createElement('span');
      meta.textContent = `${item.width}×${item.height} · ${item.frames}f`;
      const time = document.createElement('span');
      time.textContent = (item.elapsedMs / 1000).toFixed(1) + 's';
      foot.append(meta, time);

      body.append(prompt, foot);
      card.append(thumb, body);

      card.addEventListener('click', () => {
        resetStage();
        showResult({
          id: item.id,
          videoUrl: item.videoUrl,
          posterUrl: item.posterUrl,
          result: {
            width: item.width,
            height: item.height,
            frames: item.frames,
            seed: item.seed,
            elapsedMs: item.elapsedMs
          }
        });
        $('phaseText').textContent = '历史记录';
        $('progressText').textContent = '';
        $('barFill').style.width = '100%';
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });

      gallery.appendChild(card);
    }
  }

  /* ----------------------------------------------------------------- wire */

  function bind() {
    bindDropzones();
    renderRefs();

    $('prompt').addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        generate();
      }
    });

    $('btnGenerate').addEventListener('click', generate);

    $('btnCancel').addEventListener('click', async () => {
      if (!state.jobId) return;
      try {
        await postJson('/api/jobs/' + state.jobId + '/cancel', {});
        toast('已请求取消');
      } catch (error) {
        toast('取消失败：' + error.message, 'err');
      }
    });

    $('btnDice').addEventListener('click', () => {
      $('seed').value = Math.floor(Math.random() * 2147483647);
    });

    $('mode').addEventListener('change', () => {
      $('anchorField').hidden = $('mode').value !== 'fl2v';
    });

    $('btnPreload').addEventListener('click', async () => {
      const button = $('btnPreload');
      button.disabled = true;
      button.textContent = '预热中…';
      try {
        const request = buildRequest();
        const result = await postJson('/api/preload', {
          width: request.width,
          height: request.height,
          frames: request.frames,
          steps: request.steps,
          pdd8: request.pdd8,
          ditLayers: request.ditLayers,
          denoiseReuse: request.denoiseReuse,
          coreReuse: request.coreReuse
        });
        toast(`预热完成：${(result.elapsedMs / 1000).toFixed(1)} 秒`, 'ok');
      } catch (error) {
        toast('预热失败：' + error.message, 'err');
      } finally {
        button.disabled = false;
        button.textContent = '预热模型';
      }
    });

    $('btnCache').addEventListener('click', async () => {
      try {
        await postJson('/api/cache/clear', {});
        toast('缓存已清理', 'ok');
        loadStatus();
      } catch (error) {
        toast('清理失败：' + error.message, 'err');
      }
    });

    $('btnRefresh').addEventListener('click', () => {
      loadStatus();
      loadHistory();
      toast('已刷新');
    });

    $('btnAddRef').addEventListener('click', () => {
      if (state.refs.length >= 16) { toast('最多 16 个参考素材', 'err'); return; }
      state.refs.push({ kind: 'image', path: null, name: null });
      renderRefs();
    });

    $('btnSwitchPack').addEventListener('click', async () => {
      const modelDir = $('packSelect').value;
      if (!modelDir) return;
      try {
        await postJson('/api/model/select', { modelDir });
        toast('模型包已切换，正在重新加载…', 'ok');
        await loadStatus();
      } catch (error) {
        toast('切换失败：' + error.message, 'err');
      }
    });

    $('advRawJson').addEventListener('change', () => {
      const checked = $('advRawJson').checked;
      $('rawPanel').hidden = !checked || !$('rawJson').textContent;
    });

    // Advanced canvas inputs drive the shared selectors so both stay in sync.
    for (const inputId of ['advWidth', 'advHeight']) {
      $(inputId).addEventListener('change', () => {
        const width = num('advWidth');
        const height = num('advHeight');
        if (width && height) setResolution(width, height);
      });
    }
    $('advFrames').addEventListener('change', () => {
      const frames = num('advFrames');
      if (frames) setDuration(frames);
    });

    window.addEventListener('beforeunload', () => clearTimeout(state.pollTimer));
  }

  document.addEventListener('DOMContentLoaded', () => {
    bind();
    init();
  });
})();
