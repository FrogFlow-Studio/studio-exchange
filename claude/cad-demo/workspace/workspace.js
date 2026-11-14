import {Viewport} from './viewport.js';

const $ = (id) => document.getElementById(id);
const selected = {job: null, mode: 'split', view: 'iso', section: false, direction: 'model', busy: false, globalRunning: false, inspector: 'overview', measuring: false};
const units = {mm: 1, cm: 10, m: 1000, in: 25.4};
const statusLabels = {ready: 'Результат', draft: 'Скан загружен', running: 'В расчёте', error: 'Ошибка'};
const number = (value, digits = 0) => Number(value).toLocaleString('ru-RU', {
  minimumFractionDigits: digits, maximumFractionDigits: digits,
});
const plural = (value, forms) => {
  const tens = value % 100, ones = value % 10;
  return forms[tens >= 11 && tens <= 14 ? 2 : ones === 1 ? 0 : ones >= 2 && ones <= 4 ? 1 : 2];
};
const stamp = (time) => new Date(time * 1000).toLocaleString('ru-RU', {
  day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
});
const dom = (tag, cls, value) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (value !== undefined) node.textContent = value;
  return node;
};

let scene;
try {
  scene = new Viewport($('viewport'), $('marker'));
} catch {
  $('view-message').textContent = 'Для 3D-просмотра включите WebGL в браузере.';
}

let jobEpoch = 0;
const config = {ai: false, ai_budget_usd: 0.3};
let timer = null;
let loadedKey = null;

async function request(path, options) {
  const response = await fetch(path, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || 'Не удалось выполнить действие.');
  return value;
}

function message(text) {
  $('notice').hidden = !text;
  $('notice').textContent = text || '';
}

function setMode(mode, direction = selected.direction) {
  const available = selected.job?.assets || {};
  if (mode !== 'scan' && !available.model) mode = 'scan';
  if (mode === 'deviation' && !available.model_deviation && !available.deviations) mode = 'model';
  if (direction === 'model' && !available.model_deviation) direction = 'scan';
  if (direction === 'scan' && !available.deviations && available.model_deviation) direction = 'model';
  if (mode !== selected.mode) setMeasure(false);
  selected.mode = mode;
  $('measure').disabled = !scene || !available.scan || mode === 'deviation';
  selected.direction = direction;
  scene?.setMode(mode, direction);
  $('directions').hidden = mode !== 'deviation';
  $('split-labels').hidden = mode !== 'split';
  $('viewport').classList.toggle('split', mode === 'split');
  for (const button of document.querySelectorAll('[data-direction]')) {
    button.disabled = !(button.dataset.direction === 'model' ? available.model_deviation : available.deviations);
    button.setAttribute('aria-pressed', String(button.dataset.direction === direction));
  }
  for (const button of document.querySelectorAll('[data-mode]')) {
    button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
    button.disabled = button.dataset.mode === 'deviation' ? !available.model_deviation && !available.deviations :
      button.dataset.mode !== 'scan' ? !available.model : !available.scan;
  }
  const legend = $('legend');
  legend.replaceChildren();
  if (mode === 'split') return;
  if (mode === 'deviation') {
    const gradient = dom('i', 'scale');
    const limit = selected.job?.check?.tolerance_mm || .3;
    legend.append(dom('span', '', '0'), gradient, dom('span', '', number(limit, 1) + ' мм — допуск'),
                  dom('span', '', number(2 * limit, 1) + ' мм и выше'));
    if (selected.direction === 'model') {
      const item = dom('span');
      const mark = dom('i');
      mark.style.setProperty('--swatch', '#edac60');
      item.append(mark, document.createTextNode('скана здесь нет: достроено или не отсканировано'));
      legend.append(item);
    }
  } else {
    for (const [text, color] of mode === 'compare' ? [['Скан', '#86d8d2'], ['Модель', '#b5c3cc'], ...(available.unobserved || available.model_deviation ? [['Вне покрытия скана', '#edac60']] : [])] :
         mode === 'scan' ? [['Исходный скан', '#86d8d2']] : [['Модель', '#b5c3cc']]) {
      const item = dom('span');
      const mark = dom('i');
      mark.style.setProperty('--swatch', color);
      item.append(mark, document.createTextNode(text));
      legend.append(item);
    }
  }
}

function checkRow(title, detail, good) {
  const row = dom('div', 'check-row');
  const content = dom('div');
  content.append(dom('strong', '', title), dom('small', '', detail));
  row.append(dom('span', good ? 'check-icon' : 'check-icon warn', good ? '✓' : '!'), content);
  return row;
}

function metric(label, value, direction) {
  const assets = selected.job.assets;
  const enabled = Boolean(scene && (direction === 'scan' ? assets.deviations : assets.model_deviation));
  const item = dom(enabled ? 'button' : 'div', 'metric');
  if (enabled) {
    item.type = 'button';
    item.title = 'Показать карту отклонений';
    item.onclick = () => setMode('deviation', direction);
  }
  const title = dom('div', 'metric-label');
  title.append(dom('span', '', label), dom('strong', '', (value < 1 && value >= .9995 ? '<100' : number(value * 100, 1)) + '%'));
  const track = dom('div', 'track');
  const fill = dom('i');
  fill.style.setProperty('--value', Math.min(100, Math.max(0, value * 100)) + '%');
  track.append(fill);
  item.append(title, track);
  return item;
}

function renderInspection(job) {
  const root = $('inspection');
  root.replaceChildren();
  const badge = $('inspection-state');
  const report = job.check;
  badge.classList.toggle('warn', Boolean(report?.issues?.length) || job.status === 'error');
  if (!report) {
    badge.textContent = job.status === 'running' ? 'В расчёте' : job.inspection_error ? 'Проверка прервана' : 'Ожидает проверки';
    const detail = job.inspection_error || job.error || (job.status === 'running'
      ? 'Результаты появятся после записи и перечитывания STEP.'
      : 'Подтвердите единицы скана и запустите построение модели.');
    root.append(dom('p', 'empty', detail));
    return;
  }
  badge.textContent = report.issues.length ? 'Требует внимания' :
    report.has_reference ? 'Проверка завершена' : 'Сверено со сканом';
  const top = report.topology;
  const file = dom('section', 'inspect-section');
  file.append(dom('h3', '', 'Тело'));
  file.append(
    checkRow('STEP перечитан', 'Экспорт доступен', Boolean(job.assets.step)),
    checkRow(top.valid && top.closed ? 'Тело валидно и замкнуто' : 'Проверка тела выявила проблему',
      number(top.solids) + ' ' + plural(top.solids, ['тело', 'тела', 'тел']) + ' · ' +
      number(top.faces) + ' ' + plural(top.faces, ['грань', 'грани', 'граней']) + ' · ' +
      number(top.edges) + ' ' + plural(top.edges, ['ребро', 'ребра', 'рёбер']),
      top.valid && top.closed && top.connected_solid),
  );

  const kinds = top.surface_types || {};
  const count = (kind) => kinds[kind] || 0;
  const faces = dom('dl', 'face-counts');
  const known = ['PLANE', 'CYLINDER', 'BSPLINE'];
  const other = Object.entries(kinds).reduce((sum, [kind, value]) => sum + (known.includes(kind) ? 0 : value), 0);
  for (const [label, value] of [['Плоскости', count('PLANE')], ['Цилиндры', count('CYLINDER')],
      ['B-сплайны', count('BSPLINE')], ['Прочие', other]]) {
    faces.append(dom('dt', '', label), dom('dd', '', number(value)));
  }
  if (Object.keys(kinds).length) file.append(faces);

  const coverage = dom('section', 'inspect-section');
  const heading = dom('div', 'metric-heading');
  heading.append(dom('h3', '', 'Сравнение со сканом'),
                 dom('span', '', '±' + number(report.tolerance_mm, 1) + ' мм'));
  coverage.append(heading);
  const forward = report.scan_surface?.reference_to_model;
  const back = report.scan_surface?.model_to_reference;
  if (forward) coverage.append(metric('Скан → модель', forward.within_share, 'scan'));
  if (back) coverage.append(metric('Модель → скан', back.within_share, 'model'));
  if (forward && back) coverage.append(dom('p', 'section-note',
    'P95 ' + number(forward.p95_mm, 2) + ' / ' + number(back.p95_mm, 2) + ' мм'));

  const audit = dom('section', 'inspect-section');
  audit.append(dom('h3', '', 'Проверка по эталону'));
  if (report.has_reference) {
    const counts = report.reference_outcomes || {};
    audit.append(dom('p', 'section-note', number(counts.ok || 0) + ' из ' +
      number((counts.ok || 0) + (counts.parameter_error || 0) + (counts.missing || 0)) + ' элементов в допуске'));
  } else {
    audit.append(dom('p', 'section-note', 'Эталонный STEP отсутствует. Параметры достроенных элементов не подтверждены.'));
  }
  for (const issue of report.issues) {
    const item = dom(issue.point && scene ? 'button' : 'div', 'issue');
    item.append(dom('strong', '', issue.label));
    if (typeof issue.value === 'number') item.append(dom('span', '',
      number(issue.value, 3) + ' ' + issue.unit + ' · допуск ' + number(issue.limit, 2) + ' ' + issue.unit));
    item.append(dom('span', '', issue.source));
    if (item instanceof HTMLButtonElement) {
      item.type = 'button';
      item.setAttribute('aria-pressed', 'false');
      item.onclick = () => {
        const active = item.getAttribute('aria-pressed') === 'true';
        for (const b of root.querySelectorAll('button.issue')) b.setAttribute('aria-pressed', 'false');
        if (active) {
          scene.clearFocus();
          setView(selected.view);
          return;
        }
        item.setAttribute('aria-pressed', 'true');
        if (selected.mode === 'scan') setMode('model');
        scene.focus(issue.point);
      };
    }
    audit.append(item);
  }
  if (job.assets.report) {
    const link = dom('a', 'report-link', 'Скачать отчёт проверки');
    link.href = job.assets.report;
    audit.append(link);
  }
  if (selected.inspector === 'body') {
    root.append(file);
  } else if (selected.inspector === 'review') {
    root.append(audit);
  } else {
    root.append(coverage);
    const overview = dom('section', 'inspect-section');
    overview.append(dom('h3', '', 'Готовность STEP'),
      checkRow(top.valid && top.closed && top.connected_solid ? 'Одно замкнутое тело' : 'Проверьте геометрию',
        number(top.faces) + ' ' + plural(top.faces, ['грань', 'грани', 'граней']),
        Boolean(top.valid && top.closed && top.connected_solid)));
    if (!report.has_reference) overview.append(dom('p', 'section-note', 'Посадка и размеры требуют проверки по оригиналу.'));
    if (report.issues.length) overview.append(dom('p', 'section-note', 'Замечаний: ' + number(report.issues.length)));
    root.append(overview);
  }

}

function clearFocus() {
  scene?.clearFocus();
  for (const button of $('inspection').querySelectorAll('button.issue')) button.setAttribute('aria-pressed', 'false');
}

function setView(view) {
  selected.view = view;
  const names = {iso: 'Изометрия', top: 'Вид сверху', bottom: 'Вид снизу',
    front: 'Вид спереди', back: 'Вид сзади', left: 'Вид слева', right: 'Вид справа'};
  $('view-label').textContent = names[view];
  $('view').value = view;
  scene?.view(view);
}

function showDimensions() {
  const job = selected.job;
  if (!job?.scan) return;
  const scale = job.status === 'ready' || job.aligned ? 1 : units[$('units').value];
  $('dimensions').textContent = job.scan.extents.map((x) => number(x * scale, 1)).join(' × ');
}

function showActions() {
  const job = selected.job;
  const draft = job?.status === 'draft';
  $('run').disabled = !job || selected.busy || selected.globalRunning || job.status === 'running';
  $('open-file').disabled = selected.busy;
  $('run-label').textContent = draft ? 'Построить модель' : 'Пересчитать';
  $('units').disabled = !draft || job.sample;
  $('run').classList.toggle('primary', Boolean(draft));
  $('run').classList.toggle('secondary', !draft);
  const locked = !job || selected.busy || selected.globalRunning || job.status === 'running';
  $('ai').disabled = locked || !config.ai;
  $('notes').disabled = locked;
  const macro = $('macro');
  macro.hidden = !job?.assets?.macro;
  if (job?.assets?.macro) macro.href = job.assets.macro;
  else macro.removeAttribute('href');
  const link = $('step');
  if (job?.assets?.step) {
    link.href = job.assets.step;
    link.removeAttribute('aria-disabled');
  } else {
    link.removeAttribute('href');
    link.setAttribute('aria-disabled', 'true');
  }
}

function elapsed(seconds) {
  const total = Math.floor(Math.max(0, seconds));
  return total < 60 ? total + ' с' : Math.floor(total / 60) + ' мин ' + (total % 60) + ' с';
}

function runLabel(identity) {
  if (!identity) return '';
  return (identity.engine_revision?.slice(0, 7) || 'Версия неизвестна') +
    ' · ' + (identity.ai ? 'с ИИ' : 'без ИИ');
}

function renderJob(job) {
  const old = selected.job;
  selected.job = job;
  const changed = old?.id !== job.id || old?.status !== job.status;
  if (changed) {
    $('part-name').textContent = job.name;
    $('sample-tag').hidden = !job.sample;
    $('sample-tag').textContent = job.example ? 'Сохранённый пример' : 'Тестовый скан';
    $('part-meta').textContent = job.filename + (job.scan ? ' · ' + number(job.scan.triangles) + ' треугольников' : '');
    $('run-identity').textContent = runLabel(job.identity);
    $('ai').checked = Boolean(job.identity?.ai) && config.ai;
    if (job.notes !== undefined) $('notes').value = job.notes || '';
    $('scan-type').textContent = job.filename.split('.').pop().toUpperCase();
    $('model-type').textContent = job.assets.model ? 'STEP' : 'Ожидает расчёта';
    $('tolerance').textContent = job.check ? '±' + number(job.check.tolerance_mm, 1) + ' мм' : '—';
    $('units').value = job.units;
    showDimensions();
    renderInspection(job);
    if (job.status === 'draft') $('settings').open = true;
    try { localStorage.setItem('cad.workspace.job', job.id); } catch { /* Browser storage is optional. */ }
    message(job.error || null);
    for (const entry of $('history').children) entry.setAttribute('aria-current', String(entry.dataset.id === job.id));
  }
  showActions();
  const detail = job.stage_detail;
  const label = detail ? detail[0].toUpperCase() + detail.slice(1) : job.stage;
  const stage = detail && job.attempt && !detail.startsWith('попытка') ?
    'Попытка ' + job.attempt + ' · ' + label : label;
  if ($('stage').textContent !== stage) $('stage').textContent = stage;
  $('stage-dot').className = job.status;
  const heartbeat = job.heartbeat;
  $('elapsed').textContent = heartbeat ? 'Идёт ' + heartbeat.minutes + ' мин' +
    (heartbeat.limit_minutes != null ? ' из предела ' + heartbeat.limit_minutes + ' мин' : '') :
    job.seconds != null ? 'Всего ' + elapsed(job.seconds) : '';
  const compared = (job.compared || []).filter((id) => !(job.rejected || []).includes(id));
  $('attempts').textContent = job.winner ? 'Выбрана попытка ' + job.winner :
    compared.length ? 'Со сканом сравнены попытки ' + compared.join(', ') : '';
  const log = $('log');
  const scroll = log.scrollTop;
  const pinned = log.scrollHeight - log.clientHeight - scroll < 25;
  const lines = job.log?.join('\n') || (job.example ? 'Пример сохранён после реального расчёта.' : 'Журнал пока пуст.');
  if (log.textContent !== lines) {
    log.textContent = lines;
    if (!log.hidden) log.scrollTop = pinned || changed ? log.scrollHeight : scroll;
  }

  const key = job.id + (job.status === 'ready' ? '/result' : '/scan');
  if (key !== loadedKey && scene && job.assets.scan) {
    loadedKey = key;
    setMeasure(false);
    $('view-message').hidden = false;
    $('viewport').classList.add('loading-geometry');
    $('view-message').textContent = 'Загрузка геометрии…';
    setMode(job.assets.model ? 'split' : 'scan', 'model');
    selected.section = false;
    $('section').setAttribute('aria-pressed', 'false');
    $('section-level').hidden = true;
    $('section-height').hidden = true;
    $('section-axis').hidden = true;
    $('section-flip').hidden = true;
    scene.load(job.assets, job.check?.tolerance_mm || .3).then(() => {
      if (key === loadedKey) {
        setView('iso');
        $('view-message').hidden = true;
        $('viewport').classList.remove('loading-geometry');
        updateSection();
      }
    }).catch((error) => {
      if (key === loadedKey) {
        $('view-message').textContent = error instanceof TypeError ?
          'Не удалось загрузить геометрию. Откройте расчёт заново.' : error.message;
      }
    });
  }
}

async function history() {
  const jobs = await request('/api/jobs');
  const list = $('history');
  list.replaceChildren();
  selected.globalRunning = jobs.some((job) => job.status === 'running');
  for (const job of jobs) {
    const entry = dom('button', 'history-entry');
    entry.type = 'button';
    entry.dataset.id = job.id;
    entry.dataset.status = job.status;
    entry.setAttribute('aria-current', String(job.id === selected.job?.id));
    const info = dom('span');
    info.append(dom('b', '', job.name), dom('small', '',
      (job.created ? stamp(job.created) + ' · ' : '') + (statusLabels[job.status] || job.status)));
    if (job.identity) info.append(dom('small', 'history-run', runLabel(job.identity)));
    entry.append(dom('i'), info);
    entry.onclick = () => activate(job.id);
    list.append(entry);
  }
  showActions();
  return jobs;
}

function schedule() {
  clearTimeout(timer);
  if (selected.globalRunning || selected.job?.status === 'running') timer = setTimeout(() => refresh(), 900);
}

async function activate(id) {
  const epoch = ++jobEpoch;
  clearTimeout(timer);
  try {
    const job = await request('/api/jobs/' + id);
    if (epoch !== jobEpoch) return;
    renderJob(job);
    await history();
    schedule();
  } catch (error) {
    if (epoch === jobEpoch) message(error.message);
  }
}

async function refresh() {
  const id = selected.job?.id;
  const epoch = jobEpoch;
  try {
    const job = await request('/api/jobs/' + id);
    if (epoch !== jobEpoch) return;
    renderJob(job);
    if (job.status !== 'running') await history();
  } catch (error) {
    if (epoch === jobEpoch) message(error.message);
  } finally {
    if (epoch === jobEpoch) schedule();
  }
}

async function mutate(path, body, headers = {}, pending = null) {
  if (selected.busy) return;
  selected.busy = true;
  showActions();
  message(pending);
  try {
    const job = await request(path, {
      method: 'POST', body,
      headers: {'X-CAD-Workspace': '1', 'X-Units': $('units').value, ...headers},
    });
    ++jobEpoch;
    renderJob(job);
    await history();
    schedule();
  } catch (error) {
    message(error.message);
  } finally {
    selected.busy = false;
    showActions();
  }
}

function upload(file) {
  if (!file) return;
  if (!/\.(stl|obj|ply)$/i.test(file.name) || file.size > 300 * 1024 ** 2 || !file.size) {
    message('Выберите STL, OBJ или PLY до 300 МБ.');
    return;
  }
  mutate('/api/jobs', file, {'X-Filename': encodeURIComponent(file.name)}, 'Читаю скан…');
}

$('open-file').onclick = () => $('file-input').click();
$('file-input').onchange = (event) => {
  upload(event.target.files[0]);
  event.target.value = '';
};
$('units').onchange = () => {
  showDimensions();
  if (selected.section) updateSection();
  if (selected.measuring) setMeasure(true);
};
$('run').onclick = () => selected.job && mutate('/api/jobs/' + selected.job.id + '/run', null,
  {'X-CAD-Regions': '0', 'X-AI': $('ai').checked ? '1' : '0', 'X-Notes': encodeURIComponent($('notes').value.trim())});
$('fit').onclick = () => { clearFocus(); setView(selected.view); };
$('section').onclick = () => {
  selected.section = !selected.section;
  $('section').setAttribute('aria-pressed', String(selected.section));
  updateSection();
};
function updateSection() {
  $('section-level').hidden = !selected.section;
  $('section-height').hidden = !selected.section;
  $('section-axis').hidden = !selected.section;
  $('section-flip').hidden = !selected.section;
  const axis = $('section-axis').value;
  const flipped = $('section-flip').getAttribute('aria-pressed') === 'true';
  if (selected.measuring) setMeasure(true);
  const height = scene?.clip(selected.section, Number($('section-level').value), axis, flipped);
  const scale = selected.job?.status === 'ready' || selected.job?.aligned ? 1 : units[$('units').value];
  $('section-height').textContent = Number.isFinite(height) ? number(height * scale, 2) + ' мм' : '—';
  $('section-level').setAttribute('aria-valuetext', $('section-height').textContent);
}
$('section-level').oninput = updateSection;
$('section-axis').onchange = updateSection;
$('section-flip').onclick = () => {
  const button = $('section-flip');
  button.setAttribute('aria-pressed', String(button.getAttribute('aria-pressed') !== 'true'));
  updateSection();
};

function setMeasure(active) {
  selected.measuring = active;
  selected.measureSource = null;
  $('measure').setAttribute('aria-pressed', String(active));
  $('measure-readout').hidden = !active;
  $('viewport').classList.toggle('measuring', active);
  $('measure-result').textContent = 'Выберите две точки на одной поверхности';
  scene?.clearMeasure();
}

$('measure').onclick = () => setMeasure(!selected.measuring);
$('clear-measure').onclick = () => setMeasure(true);
$('grid').onclick = () => {
  const button = $('grid');
  const active = button.getAttribute('aria-pressed') !== 'true';
  button.setAttribute('aria-pressed', String(active));
  scene?.setGrid(active);
};
let pointerStart;
$('viewport').addEventListener('pointerdown', (event) => {
  pointerStart = selected.measuring && event.button === 0 && event.target.tagName === 'CANVAS'
    ? [event.clientX, event.clientY] : null;
});
$('viewport').addEventListener('pointerup', (event) => {
  if (!selected.measuring || !pointerStart
      || Math.hypot(event.clientX - pointerStart[0], event.clientY - pointerStart[1]) > 5) return;
  pointerStart = null;
  const hit = scene?.pick(event.clientX, event.clientY);
  if (!hit) return;
  if (selected.measureSource && selected.measureSource !== hit.source) scene.clearMeasure();
  selected.measureSource = hit.source;
  const mm = scene.measure(hit);
  const scale = selected.job.status === 'ready' || selected.job.aligned ? 1 : units[$('units').value];
  const surface = hit.source === 'scan' ? 'Скан' : 'Модель';
  $('measure-result').textContent = mm === null ? surface + ' · выберите вторую точку'
    : surface + ' · ≈' + number(mm * scale, 2) + ' мм между точками';
});

for (const button of document.querySelectorAll('[data-inspect]')) {
  button.onclick = () => {
    clearFocus();
    selected.inspector = button.dataset.inspect;
    for (const tab of document.querySelectorAll('[data-inspect]')) tab.setAttribute('aria-pressed', String(tab === button));
    if (selected.job) renderInspection(selected.job);
  };
}
$('journal').onclick = () => {
  $('log').hidden = !$('log').hidden;
  $('journal').setAttribute('aria-expanded', String(!$('log').hidden));
  if (!$('log').hidden) $('log').scrollTop = $('log').scrollHeight;
};
for (const button of document.querySelectorAll('[data-mode]')) button.onclick = () => setMode(button.dataset.mode);
for (const button of document.querySelectorAll('[data-direction]')) {
  button.onclick = () => setMode('deviation', button.dataset.direction);
}
$('view').onchange = (event) => {
  clearFocus();
  setView(event.target.value);
};
const drop = $('viewport');
drop.addEventListener('dragover', (event) => {
  event.preventDefault();
  document.body.classList.add('dragging');
});
drop.addEventListener('dragleave', () => document.body.classList.remove('dragging'));
drop.addEventListener('drop', (event) => {
  event.preventDefault();
  document.body.classList.remove('dragging');
  upload(event.dataTransfer.files[0]);
});

function applyConfig(value) {
  Object.assign(config, value);
  $('ai-label').textContent = 'Построить с ИИ (платно, не больше $' + number(config.ai_budget_usd, 2) + ' за расчёт)';
  $('ai-note').textContent = config.ai ? 'Без галочки строит алгоритм, бесплатно.'
    : 'Ключ ИИ не задан: строит алгоритм без ИИ, бесплатно.';
  $('ai-key').hidden = config.ai;
  showActions();
}

request('/api/config').then(applyConfig).catch(() => {});

$('ai-key').addEventListener('submit', (event) => {
  event.preventDefault();
  const key = $('ai-key-value').value.trim();
  if (!key) return;
  request('/api/ai-key', {method: 'POST', headers: {'Content-Type': 'application/json', 'X-CAD-Workspace': '1'},
    body: JSON.stringify({key})}).then((value) => {
    $('ai-key-value').value = '';
    message('');
    applyConfig(value);
  }).catch((error) => message(error.message));
});

history().then((jobs) => {
  let previous;
  try { previous = localStorage.getItem('cad.workspace.job'); } catch { /* Browser storage is optional. */ }
  return activate(jobs.find((j) => j.status === 'running')?.id || jobs.find((j) => j.id === previous)?.id || 'example');
}).catch((error) => {
  message(error.message);
  $('view-message').textContent = 'Не удалось связаться с локальным приложением.';
});
