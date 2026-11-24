/* H3 Flash WebUI — vanilla JS, no build step. */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const FPS = 24;

  const state = {
    config: null,
    status: null,
    presetId: 'balanced',
    demoId: null,
    mode: 't2v',
    jobId: null,
    pollTimer: null,
    anchors: { first: null, last: null },
    refImages: [],   // ordered reference images (Ref2VA)
    refs: [],        // advanced: reference videos / audio
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
    renderModes();
    renderComponents();
    placeComponentPanel();
    updateModelBanner();

    if (advanced) {
      $('advancedPanel').open = true;
      $('advSteps').value = 6;
      $('advDitLayers').value = 50;
      $('advDenoiseReuse').value = 1;
      $('advCoreReuse').value = 1;
      $('advWidth').value = 864;
      $('advHeight').value = 480;
      $('advFrames').value = 56;
      $('advRenderWidth').value = 0;
      $('advRenderHeight').value = 0;
      $('refHint').textContent = config.ref2vaAvailable
        ? '按顺序生效，需要 Ref2VA 权重'
        : '当前模型包不含 Ref2VA，参考素材会被忽略';
    }

    selectPreset('balanced', true);
    renderDemos();
    loadHistory();
    loadStatus();
  }

  /* ---------------------------------------------------------------- modes */

  function modeList() {
    return state.config.modes || [];
  }

  function renderModes() {
    const container = $('modes');
    container.innerHTML = '';
    for (const mode of modeList()) {
      const button = document.createElement('button');
      button.className = 'mode';
      button.dataset.id = mode.id;
      button.innerHTML = '<b></b><small></small>';
      button.querySelector('b').textContent = mode.name;
      button.querySelector('small').textContent = mode.description;
      button.addEventListener('click', () => selectMode(mode.id));
      container.appendChild(button);
    }
    // ?mode=ref2v opens the UI on a given mode (handy for shortcuts).
    const requested = new URLSearchParams(location.search).get('mode');
    selectMode(requested || 't2v', true);
  }

  function selectMode(id, silent) {
    const mode = modeList().find((m) => m.id === id) || modeList()[0];
    if (!mode) return;
    state.mode = mode.id;

    for (const node of $('modes').children)
      node.classList.toggle('active', node.dataset.id === mode.id);

    $('anchorField').hidden = mode.id !== 'fl2v';
    $('refField').hidden = mode.id !== 'ref2v';
    // Each mode has its own examples, so the card list follows the switch.
    renderDemos();

    // Reference conditioning is the only mode that needs the second DiT.
    const blocked = mode.requiresRef2Va && !state.config.ref2vaAvailable;
    $('modeHint').textContent = blocked
      ? '缺少 Ref2VA 权重，详见“模型指引”'
      : mode.requiresRef2Va
        ? '按顺序参考，越靠前权重越高'
        : mode.id === 'fl2v'
          ? '至少给一张首帧或尾帧'
          : '';

    if (blocked) {
      toast('当前模型包不含 Ref2VA，多图参考会被拒绝', 'err');
    } else if (!silent && mode.id === 'ref2v' && state.refImages.length === 0) {
      // A nudge, not an error: the user still has to pick the images.
      $('refImageHint').textContent = `按顺序生效，最多 ${maxRefImages()} 张`;
    }
  }

  function maxRefImages() {
    return (state.config && state.config.maxReferenceImages) || 9;
  }

  /* ---------------------------------------------------------- model health */

  function updateModelBanner() {
    const config = state.config;
    if (!config) return;
    const missing = (config.modelComponents || []).filter((c) => c.required && !c.present);
    const show = missing.length > 0;
    $('modelBanner').hidden = !show;
    if (!show) return;

    $('bannerTitle').textContent = config.modelPackDir
      ? '模型包不完整'
      : '还没有找到模型权重';
    $('bannerText').textContent = config.modelError || '';

    const list = $('bannerMissing');
    list.innerHTML = '';
    for (const item of missing) {
      const row = document.createElement('code');
      row.textContent = `${item.slot}`;
      row.title = item.path || '';
      list.appendChild(row);
    }
  }

  function renderComponents() {
    const container = $('components');
    if (!container) return;
    container.innerHTML = '';
    const items = state.config.modelComponents || [];
    if (items.length === 0) {
      const empty = document.createElement('p');
      empty.className = 'hint';
      empty.textContent = '没有解析到组件路径';
      container.appendChild(empty);
      return;
    }
    for (const item of items) {
      const row = document.createElement('div');
      row.className = 'component' + (item.present ? '' : ' missing');
      const mark = document.createElement('i');
      mark.textContent = item.present ? '✓' : '✗';
      const name = document.createElement('span');
      name.textContent = item.label;
      const path = document.createElement('code');
      path.textContent = item.present ? item.path : `${item.slot} 缺失`;
      path.title = item.path || '';
      row.append(mark, name, path);
      container.appendChild(row);
    }
  }

  async function recheckModel() {
    try {
      toast('正在重新检测模型…');
      const result = await postJson('/api/model/refresh', {});
      await refreshConfig();
      updateModelBanner();
      renderComponents();
      if (result.usable) toast('已找到模型权重', 'ok');
      else toast(result.error || '仍未找到完整权重', 'err');
    } catch (error) {
      toast('检测失败：' + error.message, 'err');
    }
  }

  // The advanced edition's left column is much taller than the preview column,
  // so the component list moves under the preview instead of stretching the
  // composer further down.
  function placeComponentPanel() {
    const target = $('stageExtra');
    if (!target || !state.config.advancedEnabled || target.dataset.placed) return;
    target.append($('componentLabel'), $('components'));
    target.hidden = false;
    target.dataset.placed = '1';
  }

  async function refreshConfig() {
    try {
      state.config = await api('/api/config');
      $('modelPath').textContent = state.config.modelDir || '模型未加载';
      $('modelPath').title = state.config.modelPackDir || '';
      renderPacks();
      renderModes();
      placeComponentPanel();
    } catch (error) {
      console.warn('config refresh failed', error);
    }
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

      // The banner is driven by /api/config (file probes), which is the only
      // source that knows which component is missing.
      updateModelBanner();
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

  /* ---------------------------------------------------------------- demos */

  /* Built-in examples, one set per generation mode. Each one is a working
   * prompt plus the spec it was tuned for, so the fastest way to learn the
   * model is to press a card and then edit one clause.
   *
   * They deliberately cover the three things that are hard to discover from
   * the model alone: how much concrete detail a prompt needs, how to place an
   * action in time (H3 follows explicit second ranges), and how to name a
   * camera move instead of hoping for one. The first/last-frame and reference
   * demos ship their source image so the whole path works in one click. */
  const DEMOS = [
    {
      id: 't2v-real', mode: 't2v', title: '写实 + 推镜', note: '黄金时刻 · dolly-in',
      preset: 'balanced',
      prompt: '一只赤狐在刚落的雪原上缓步前行，黄金时刻的低角度侧逆光把毛尖照成金色，鼻息呼出的白气清晰可见。'
        + '镜头从全景缓慢推近（dolly-in）到中景，35mm 浅景深，背景雪面虚化成奶油状散景，'
        + '轻微手持呼吸感，胶片颗粒，24fps 电影质感。'
    },
    {
      id: 't2v-timeline', mode: 't2v', title: '时间轴精确控制', note: '按秒分段描述动作',
      preset: 'balanced',
      prompt: '0-2 秒：雨夜的东京小巷，霓虹灯牌在积水里倒映成一片红色，镜头静止，只有雨滴打在铁皮雨棚上。'
        + '2-4 秒：一名穿米色风衣的女子从画面左侧入画，走进巷子深处，镜头开始缓慢跟随（tracking shot），'
        + '焦点从前景水洼拉到她的背影。'
        + '4-6 秒：她停下脚步回头看向镜头，霓虹光在脸上扫过，镜头轻微推近。'
        + '整体冷调，青蓝阴影配品红高光，35mm 变形宽银幕，轻微光晕。'
    },
    {
      id: 't2v-orbit', mode: 't2v', title: '高级运镜：环绕', note: 'orbit + 广角',
      preset: 'balanced',
      prompt: '一座被苔藓覆盖的石砌灯塔立在黑色玄武岩海崖上，正午硬光，海雾在脚下翻涌。'
        + '镜头以灯塔为中心做 180 度环绕（orbit shot），从背光剪影转到迎光的暖色石面，同时缓慢下降。'
        + '24mm 广角，边缘轻微桶形畸变，高快门凝固浪花，色彩浓郁。'
    },
    {
      id: 't2v-mood', mode: 't2v', title: '氛围与声音', note: '横移 · 环境音',
      preset: 'balanced',
      prompt: '深夜的旧书店，只有一盏黄铜台灯亮着，尘埃在光柱里缓慢漂浮。'
        + '镜头极缓慢地横向平移（slider），扫过一排排书脊。'
        + '暖色钨丝灯，深绿与琥珀色调，浅景深，细微颗粒。'
        + '环境音：远处雨声、翻书声、老旧木地板的轻响。'
    },
    {
      id: 'fl2v-continue', mode: 'fl2v', title: '首帧续写 + 推镜', note: '内置首帧，一键可跑',
      preset: 'balanced',
      sample: { url: 'demo/fox-snow.jpg', slot: 'first' },
      prompt: '以这一帧为起点：赤狐保持原有的姿态与毛色，向前迈出两步，耳朵轻微转动，鼻息带出白气。'
        + '镜头继续缓慢推近并轻微右移，黄金时刻的侧逆光保持不变，脚步扬起细雪。'
        + '严格保持原片的色调、胶片颗粒与浅景深。'
    },
    {
      id: 'fl2v-bridge', mode: 'fl2v', title: '首尾帧过渡', note: '内置两帧，一键可跑',
      preset: 'balanced',
      samples: [
        { url: 'demo/fox-snow.jpg', slot: 'first' },
        { url: 'demo/snow-forest.jpg', slot: 'last' }
      ],
      prompt: '从首帧的近景赤狐拉成尾帧的雪林远景：镜头持续后拉（dolly-out）并轻微上升，'
        + '赤狐转身小跑着没入画面深处的雪林，只剩一串脚印。'
        + '光线从首帧的暖金侧逆光自然过渡到尾帧的冷蓝暮色，雪面反射由暖转冷。'
        + '两帧之间的主体毛色、雪地纹理与镜头焦段保持一致，运动连续，不要跳切。'
    },
    {
      id: 'ref2v-subject', mode: 'ref2v', title: '主体一致性', note: '内置参考图，一键可跑',
      preset: 'balanced',
      sample: { url: 'demo/fox-snow.jpg', slot: 'ref' },
      prompt: '参考图中的赤狐作为主体：把它放进一间铺着波斯地毯的旧书房，午后斜光从百叶窗落在它身上。'
        + '它抬头看向镜头，耳朵前倾。镜头低角度缓慢推近。'
        + '保持参考图里狐狸的毛色、体态与面部特征完全一致，只改变环境与光线。'
    },
    {
      id: 'ref2v-compose', mode: 'ref2v', title: '多图组合', note: '内置两张，一键可跑',
      preset: 'balanced',
      samples: [
        { url: 'demo/fox-snow.jpg', slot: 'ref' },
        { url: 'demo/snow-forest.jpg', slot: 'ref' }
      ],
      prompt: '参考图 1 提供主体（赤狐的毛色、体态与面部特征），参考图 2 提供场景（积雪针叶林与暮色天空）。'
        + '把这只赤狐放进参考图 2 的林线前，它站在齐踝的深雪里回头望向镜头，呼出的白气在冷空气里散开。'
        + '镜头从低角度缓慢推近并轻微右移，前景雪粒被风吹起。'
        + '主体特征严格取自图 1，环境结构与光线方向严格取自图 2，整体统一为冷蓝阴影配暖橘高光。'
    }
  ];

  function renderDemos() {
    const container = $('demos');
    if (!container) return;
    const list = DEMOS.filter((demo) => demo.mode === state.mode);
    $('demoField').hidden = list.length === 0;
    container.innerHTML = '';
    for (const demo of list) {
      const card = document.createElement('button');
      card.className = 'demo';
      card.innerHTML = '<b></b><small></small>';
      card.querySelector('b').textContent = demo.title;
      card.querySelector('small').textContent = demo.note;
      if (demo.id === state.demoId) card.classList.add('active');
      card.addEventListener('click', () => applyDemo(demo));
      container.appendChild(card);
    }
  }

  async function applyDemo(demo) {
    state.demoId = demo.id;
    $('prompt').value = demo.prompt;
    $('prompt').scrollTop = 0;
    if (demo.preset) selectPreset(demo.preset, false);
    renderDemos();
    const samples = demo.samples || (demo.sample ? [demo.sample] : []);
    if (samples.length === 0) {
      $('prompt').focus();
      return;
    }
    try {
      toast(samples.length > 1 ? `正在载入 ${samples.length} 张示例素材…` : '正在载入示例素材…');
      const uploaded = [];
      for (const sample of samples) {
        const response = await fetch(sample.url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const blob = await response.blob();
        const name = sample.url.split('/').pop();
        const file = new File([blob], name, { type: blob.type || 'image/jpeg' });
        uploaded.push({ slot: sample.slot, ...(await uploadFile(file)) });
      }
      const references = uploaded.filter((row) => row.slot === 'ref');
      if (references.length) {
        state.refImages = references.map((row) => ({ url: row.url, name: row.name }));
        renderRefImages();
      }
      for (const row of uploaded) {
        if (row.slot === 'ref') continue;
        state.anchors[row.slot] = row.url;
        renderAnchor(row.slot, row.url);
      }
      toast('示例素材已就位，可直接生成', 'ok');
    } catch (error) {
      toast('示例素材载入失败：' + error.message, 'err');
    }
    $('prompt').focus();
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
      } else if (target.type === 'image') {
        if (state.refImages.length >= maxRefImages()) {
          toast(`参考图最多 ${maxRefImages()} 张`, 'err');
        } else {
          state.refImages.push({ url: result.url, name: result.name });
          renderRefImages();
        }
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

  /* Ordered reference images. The engine treats references as a sequence, so
   * the badge is the index the prompt has to refer to (image 1, image 2 ...). */
  function renderRefImages() {
    const grid = $('refGrid');
    if (!grid) return;
    grid.innerHTML = '';

    state.refImages.forEach((row, index) => {
      const card = document.createElement('div');
      card.className = 'refcard';

      const img = document.createElement('img');
      img.src = row.url;
      img.alt = row.name || `参考图 ${index + 1}`;
      img.title = row.name || '';

      const order = document.createElement('span');
      order.className = 'order';
      order.textContent = String(index + 1);

      const controls = document.createElement('div');
      controls.className = 'refcontrols';
      const left = document.createElement('button');
      left.className = 'mini';
      left.textContent = '‹';
      left.title = '前移';
      left.disabled = index === 0;
      left.addEventListener('click', () => moveRef(index, -1));
      const right = document.createElement('button');
      right.className = 'mini';
      right.textContent = '›';
      right.title = '后移';
      right.disabled = index === state.refImages.length - 1;
      right.addEventListener('click', () => moveRef(index, 1));
      controls.append(left, right);

      const del = document.createElement('button');
      del.className = 'del';
      del.textContent = '×';
      del.title = '移除';
      del.addEventListener('click', () => {
        state.refImages.splice(index, 1);
        renderRefImages();
      });

      card.append(img, order, controls, del);
      grid.appendChild(card);
    });

    if (state.refImages.length < maxRefImages()) {
      const add = document.createElement('button');
      add.className = 'refadd';
      add.type = 'button';
      add.innerHTML = '<b>+</b><span>添加参考图</span>';
      add.addEventListener('click', () => pickFile({ type: 'image' }));
      grid.appendChild(add);
    }

    $('refImageHint').textContent = state.refImages.length
      ? `已选 ${state.refImages.length} / ${maxRefImages()} 张，顺序即优先级`
      : `按顺序生效，最多 ${maxRefImages()} 张`;
  }

  function moveRef(index, delta) {
    const target = index + delta;
    if (target < 0 || target >= state.refImages.length) return;
    const [row] = state.refImages.splice(index, 1);
    state.refImages.splice(target, 0, row);
    renderRefImages();
  }

  function renderRefs() {
    const container = $('refList');
    container.innerHTML = '';
    state.refs.forEach((row, index) => {
      const node = document.createElement('div');
      node.className = 'refrow';

      const kind = document.createElement('select');
      for (const [value, label] of [['video', '视频'], ['audio', '音频'], ['video_audio', '视频+音轨']]) {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = label;
        kind.appendChild(option);
      }
      kind.value = row.kind === 'image' ? 'video' : row.kind;
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
      mode: state.mode,
      width, height, frames,
      seed,
      livePreview: $('livePreview').checked,
      previewEvery: 4
    };

    if (state.mode === 'fl2v') {
      request.firstFrame = state.anchors.first;
      request.lastFrame = state.anchors.last;
    }

    if (state.mode === 'ref2v') {
      // Images first (they are the ordered part), then any advanced video /
      // audio the user added.
      const references = state.refImages.map((row) => ({ kind: 'image', path: row.url }));
      if (advanced) {
        for (const row of state.refs.filter((r) => r.path))
          references.push({ kind: row.kind, path: row.path });
      }
      if (references.length > 0) request.references = references;
    }

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
    if (state.mode === 'fl2v' && !request.firstFrame && !request.lastFrame) {
      showAlert('首尾帧模式至少需要一张首帧或尾帧。');
      return;
    }
    if (state.mode === 'ref2v' && (!request.references || request.references.length === 0)) {
      showAlert('多图参考模式至少需要一张参考图。');
      return;
    }
    if (state.mode === 'ref2v' && !state.config.ref2vaAvailable) {
      showAlert('当前模型包不含 Ref2VA 权重，无法做多图参考。请按“模型指引”补上 Ref2VA。');
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

  /* The engine reports stages on their own scales; the server folds them into
   * one overall fraction. These chips show where that fraction sits. */
  const STAGE_PLAN = [
    { label: '文本编码', match: ['tokenizer', 'text encoder', 'Qwen vision'] },
    { label: '加载 DiT', match: ['AdaLN', 'load transformer core'] },
    { label: '去噪', match: ['denoise', 'denoise enqueue'] },
    { label: '视频解码', match: ['video VAE load', 'preview VAE load'] },
    { label: '音频', match: ['audio VAE'] },
    { label: '封装', match: ['FFmpeg', 'save'] }
  ];

  function stageIndexOf(phase) {
    if (!phase) return -1;
    return STAGE_PLAN.findIndex((stage) => stage.match.includes(phase));
  }

  function renderStageStrip(phase, status) {
    const strip = $('stageStrip');
    if (!strip) return;
    if (!phase) { strip.hidden = true; strip.textContent = ''; return; }
    const active = stageIndexOf(phase);
    strip.hidden = false;
    strip.textContent = '';
    STAGE_PLAN.forEach((stage, index) => {
      const chip = document.createElement('span');
      const state = status === 'completed' || index < active ? 'done'
        : index === active ? 'active' : 'todo';
      chip.className = 'stagechip ' + state;
      chip.textContent = stage.label;
      strip.appendChild(chip);
    });
  }

  function formatDuration(seconds) {
    if (!isFinite(seconds) || seconds < 0) return '?';
    if (seconds < 60) return seconds.toFixed(0) + 's';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return minutes + 'm' + String(Math.round(seconds % 60)).padStart(2, '0') + 's';
    return Math.floor(minutes / 60) + 'h' + String(minutes % 60).padStart(2, '0') + 'm';
  }

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
    renderStageStrip(null, 'idle');
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
      } else if (progress) {
        const overall = typeof progress.overall === 'number' ? progress.overall : 0;
        const stage = progress.stage || progress.phase || '生成中';
        const detail = progress.total > 0
          ? `${progress.completed}/${progress.total}` : '';
        $('phaseText').textContent = detail ? `${stage} · ${detail}` : stage;
        $('barFill').classList.remove('indeterminate');
        $('barFill').style.width = (overall * 100).toFixed(1) + '%';
        const elapsed = (progress.elapsedMs || 0) / 1000;
        const parts = [`${(overall * 100).toFixed(0)}%`,
                       `已用 ${formatDuration(elapsed)}`];
        if (progress.etaMs) parts.push(`剩余约 ${formatDuration(progress.etaMs / 1000)}`);
        $('progressText').textContent = parts.join(' · ');
        renderStageStrip(progress.phase, job.status);
      } else {
        $('phaseText').textContent = '生成中';
        $('barFill').classList.add('indeterminate');
        $('progressText').textContent = '';
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
      renderStageStrip('save', 'completed');
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
    renderRefImages();

    $('prompt').addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        generate();
      }
    });

    $('btnGenerate').addEventListener('click', generate);

    $('btnRecheck').addEventListener('click', recheckModel);

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

    $('btnRefresh').addEventListener('click', async () => {
      await refreshConfig();
      renderComponents();
      updateModelBanner();
      loadStatus();
      loadHistory();
      toast('已刷新');
    });

    $('btnAddRef').addEventListener('click', () => {
      const limit = (state.config.maxReferences || 12) - state.refImages.length;
      if (state.refs.length >= limit) { toast(`参考素材最多 ${limit} 个`, 'err'); return; }
      state.refs.push({ kind: 'video', path: null, name: null });
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
