/* ── Brand DNA Agent — Frontend Application ──────────────────────────── */

const API = '/api';

// ─── State ───────────────────────────────────────────────────────────────
const state = { brands: [], currentView: 'dashboard', loading: true };

// ─── API Client ──────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

// ─── Toast ───────────────────────────────────────────────────────────────
function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

// ─── Router ──────────────────────────────────────────────────────────────
function navigate(hash) { window.location.hash = hash; }

async function route() {
  const hash = window.location.hash || '#/';
  const content = document.getElementById('content');
  const breadcrumb = document.getElementById('breadcrumb');

  // Update nav active state
  document.querySelectorAll('.nav-item').forEach(el => {
    el.classList.toggle('active', el.getAttribute('href') === hash || (hash.startsWith('#/brand/') && el.getAttribute('href') === '#/'));
  });

  try {
    if (hash === '#/' || hash === '') {
      breadcrumb.innerHTML = '<strong>Dashboard</strong>';
      await renderDashboard(content);
    } else if (hash === '#/new') {
      breadcrumb.innerHTML = '<a href="#/">Dashboard</a> <span>›</span> <strong>New Brand</strong>';
      renderNewBrand(content);
    } else if (hash.startsWith('#/brand/') && hash.includes('/run/')) {
      const parts = hash.replace('#/brand/', '').split('/run/');
      breadcrumb.innerHTML = `<a href="#/">Dashboard</a> <span>›</span> <a href="#/brand/${parts[0]}">${parts[0]}</a> <span>›</span> <strong>Run</strong>`;
      await renderRunDetail(content, parts[0], parts[1]);
    } else if (hash.startsWith('#/brand/')) {
      const slug = hash.replace('#/brand/', '');
      breadcrumb.innerHTML = `<a href="#/">Dashboard</a> <span>›</span> <strong>${slug}</strong>`;
      await renderBrandDetail(content, slug);
    }
  } catch (e) {
    content.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div><div class="empty-title">Error</div><div class="empty-text">${e.message}</div></div>`;
  }

  // Close mobile sidebar
  document.getElementById('sidebar').classList.remove('open');
}

// ─── Dashboard View ──────────────────────────────────────────────────────
async function renderDashboard(el) {
  el.innerHTML = '<div class="stats-bar"><div class="stat-card skeleton" style="height:80px"></div><div class="stat-card skeleton" style="height:80px"></div><div class="stat-card skeleton" style="height:80px"></div></div>';
  const brands = await api('/brands');
  state.brands = brands;

  const totalRuns = brands.reduce((s, b) => s + b.runs_count, 0);
  const lastRun = brands.map(b => b.last_run?.started_at).filter(Boolean).sort().pop();

  el.innerHTML = `
    <div class="stats-bar">
      <div class="stat-card"><div class="stat-label">Brands</div><div class="stat-value gradient">${brands.length}</div></div>
      <div class="stat-card"><div class="stat-label">Total Runs</div><div class="stat-value gradient">${totalRuns}</div></div>
      <div class="stat-card"><div class="stat-label">Last Run</div><div class="stat-value" style="font-size:1rem">${lastRun ? timeAgo(lastRun) : '—'}</div></div>
    </div>
    <div class="section-header">
      <h2 class="section-title">Brands</h2>
      <button class="btn btn-primary btn-sm" onclick="navigate('#/new')">+ New Brand</button>
    </div>
    ${brands.length ? `<div class="brands-grid">${brands.map(brandCard).join('')}</div>` : `
      <div class="empty-state">
        <div class="empty-icon">🧬</div>
        <div class="empty-title">No brands yet</div>
        <div class="empty-text">Create your first brand to get started with AI-powered brand intelligence.</div>
        <button class="btn btn-primary" onclick="navigate('#/new')">+ Create Brand</button>
      </div>`}`;
}

function brandCard(b) {
  const socials = Object.entries(b.social || {}).map(([k, v]) => `<span class="social-badge">${k === 'instagram' ? '📷' : '📌'} ${v}</span>`).join('');
  return `<div class="brand-card" onclick="navigate('#/brand/${b.slug}')">
    <div class="brand-card-name">${esc(b.name)}</div>
    <div class="brand-card-url">${esc(b.url)}</div>
    ${socials ? `<div class="brand-card-social">${socials}</div>` : ''}
    <div class="brand-card-footer">
      <span class="runs-badge">${b.runs_count} run${b.runs_count !== 1 ? 's' : ''}</span>
      <span class="last-run-time">${b.last_run ? timeAgo(b.last_run.started_at) : 'No runs yet'}</span>
    </div>
  </div>`;
}

// ─── Brand Detail View ───────────────────────────────────────────────────
async function renderBrandDetail(el, slug) {
  el.innerHTML = '<div class="skeleton" style="height:100px;margin-bottom:20px"></div>';
  const brand = await api(`/brands/${slug}`);
  const runs = await api(`/brands/${slug}/runs`);

  const cfg = brand.config || {};
  const socials = Object.entries(cfg.social || {}).map(([k, v]) => `<span class="social-badge">${k === 'instagram' ? '📷' : '📌'} ${v}</span>`).join('');
  const cats = (cfg.known_categories || []).map(c => `<span class="social-badge">🏷️ ${c}</span>`).join('');

  el.innerHTML = `
    <div class="brand-header">
      <div>
        <div class="detail-title">${esc(brand.name)}</div>
        <div class="detail-sub">${esc(brand.url)}</div>
        <div style="display:flex;gap:6px;margin-top:10px;flex-wrap:wrap">${socials}${cats}</div>
      </div>
      <div class="brand-header-actions">
        <button class="btn btn-primary btn-sm" id="run-btn" onclick="startRun('${slug}')">▶ Run Agent</button>
        <button class="btn btn-danger btn-sm" onclick="deleteBrand('${slug}')">Delete</button>
      </div>
    </div>
    <div class="section-header"><h2 class="section-title">Runs</h2></div>
    ${runs.length ? `<div class="runs-list">${runs.map(r => runCard(slug, r)).join('')}</div>` : `
      <div class="empty-state">
        <div class="empty-icon">🚀</div>
        <div class="empty-title">No runs yet</div>
        <div class="empty-text">Click "Run Agent" to generate the Brand DNA dossier.</div>
      </div>`}`;
}

function runCard(slug, r) {
  return `<div class="run-card" onclick="navigate('#/brand/${slug}/run/${r.run_id}')">
    <span class="run-id">${r.run_id}</span>
    <span class="run-status ${r.status}">${r.status}</span>
    <div class="run-metrics">
      ${r.pages_crawled != null ? `<span class="run-metric"><strong>${r.pages_crawled}</strong> pages</span>` : ''}
      ${r.images_after_filter != null ? `<span class="run-metric"><strong>${r.images_after_filter}</strong> images</span>` : ''}
      ${r.llm_cost != null ? `<span class="run-metric"><strong>$${r.llm_cost.toFixed(3)}</strong></span>` : ''}
    </div>
  </div>`;
}

// ─── Run Detail View ─────────────────────────────────────────────────────
async function renderRunDetail(el, slug, runId) {
  el.innerHTML = '<div class="skeleton" style="height:200px"></div>';
  const data = await api(`/brands/${slug}/runs/${runId}`);
  const rpt = data.report || {};
  const stages = rpt.stages || [];
  const maxDur = Math.max(...stages.map(s => s.duration_s), 0.01);
  const llm = rpt.llm_usage || {};
  const dossier = data.dossier || {};

  el.innerHTML = `
    <div class="detail-header">
      <div class="detail-title">${esc(rpt.brand || slug)}</div>
      <div class="detail-sub">Run: ${runId}</div>
    </div>
    <div class="metrics-grid">
      <div class="metric-card"><div class="value">${rpt.pages_crawled ?? '—'}</div><div class="label">Pages</div></div>
      <div class="metric-card"><div class="value">${rpt.images_after_filter ?? '—'}</div><div class="label">Images</div></div>
      <div class="metric-card"><div class="value">${llm.calls ?? '—'}</div><div class="label">LLM Calls</div></div>
      <div class="metric-card"><div class="value">$${(llm.cost_usd || 0).toFixed(3)}</div><div class="label">LLM Cost</div></div>
      <div class="metric-card"><div class="value">${llm.tokens_in ? (llm.tokens_in + llm.tokens_out).toLocaleString() : '—'}</div><div class="label">Tokens</div></div>
    </div>
    ${dossier.one_line_positioning ? `<div style="background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--radius-md);padding:20px;margin-bottom:24px">
      <div style="font-size:.75rem;text-transform:uppercase;color:var(--text-muted);margin-bottom:8px">Positioning</div>
      <div style="font-size:1rem;font-weight:600;font-style:italic;color:var(--color-primary-light)">"${esc(dossier.one_line_positioning)}"</div>
    </div>` : ''}
    ${dossier.executive_summary ? `<div style="background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--radius-md);padding:20px;margin-bottom:24px">
      <div style="font-size:.75rem;text-transform:uppercase;color:var(--text-muted);margin-bottom:8px">Executive Summary</div>
      <div style="font-size:.875rem;line-height:1.6;color:var(--text-secondary)">${esc(dossier.executive_summary)}</div>
    </div>` : ''}
    ${data.has_pdf ? `<a href="/api/brands/${slug}/runs/${runId}/pdf" target="_blank" class="btn btn-primary btn-sm" style="margin-bottom:24px">📄 Download PDF</a>` : ''}
    <div class="section-header"><h2 class="section-title">Pipeline Stages</h2></div>
    <div class="stages-list">
      ${stages.map(s => `<div class="stage-row">
        <span class="stage-name">${esc(s.stage)}</span>
        <div class="stage-bar-wrap"><div class="stage-bar" style="width:${(s.duration_s / maxDur * 100).toFixed(1)}%"></div></div>
        <span class="stage-dur">${s.duration_s.toFixed(2)}s</span>
        <span class="stage-items">${s.items_processed} items</span>
      </div>`).join('')}
    </div>`;
}

// ─── New Brand Form ──────────────────────────────────────────────────────
function renderNewBrand(el) {
  el.innerHTML = `
    <div class="form-page">
      <div class="detail-header">
        <div class="detail-title">Create New Brand</div>
        <div class="detail-sub">Add a fashion brand for AI-powered DNA extraction</div>
      </div>
      <form id="brand-form" onsubmit="handleCreateBrand(event)">
        <div class="form-section">
          <div class="form-section-title">Brand Identity</div>
          <div class="form-group">
            <label class="form-label">Brand Name *</label>
            <input class="form-input" name="name" required placeholder="e.g. Acne Studios">
          </div>
          <div class="form-group">
            <label class="form-label">Website URL *</label>
            <input class="form-input" name="url" type="url" required placeholder="https://www.acnestudios.com">
          </div>
          <div class="form-row">
            <div class="form-group">
              <label class="form-label">Instagram</label>
              <input class="form-input" name="instagram" placeholder="acnestudios">
            </div>
            <div class="form-group">
              <label class="form-label">Pinterest</label>
              <input class="form-input" name="pinterest" placeholder="acnestudios">
            </div>
          </div>
        </div>
        <div class="form-section">
          <div class="form-section-title">Discovery Hints</div>
          <div class="form-group">
            <label class="form-label">Known Categories</label>
            <input class="form-input" name="categories" placeholder="Outerwear, Knitwear, Denim">
            <div class="form-hint">Comma-separated. Biases the garment classifier.</div>
          </div>
          <div class="form-group">
            <label class="form-label">Seed Pages</label>
            <textarea class="form-textarea" name="seed_pages" placeholder="https://www.brand.com/lookbook&#10;https://www.brand.com/collections" rows="3"></textarea>
            <div class="form-hint">One URL per line. Extra pages to prioritise beyond sitemap.</div>
          </div>
          <div class="form-group">
            <label class="form-label">Notes</label>
            <textarea class="form-textarea" name="notes" placeholder="Internal notes about this brand..." rows="2"></textarea>
          </div>
        </div>
        <button type="button" class="collapsible-toggle" onclick="toggleCollapsible(this)">
          <span class="arrow">▶</span> Advanced Settings
        </button>
        <div class="collapsible-content">
          <div class="form-section">
            <div class="form-section-title">Crawl Settings</div>
            <div class="form-row">
              <div class="form-group">
                <label class="form-label">Max Pages</label>
                <input class="form-input" name="max_pages" type="number" value="200" min="10" max="1000">
              </div>
              <div class="form-group">
                <label class="form-label">Delay (ms)</label>
                <input class="form-input" name="delay_ms" type="number" value="500" min="100" max="5000">
              </div>
            </div>
            <div class="form-row">
              <div class="form-group">
                <label class="form-label">Max Concurrency</label>
                <input class="form-input" name="max_concurrency" type="number" value="4" min="1" max="10">
              </div>
              <div class="form-group">
                <label class="form-label">JS Rendering</label>
                <select class="form-select" name="render_js">
                  <option value="false">Off (faster)</option>
                  <option value="true">On (Playwright)</option>
                </select>
              </div>
            </div>
          </div>
          <div class="form-section">
            <div class="form-section-title">Filter Settings</div>
            <div class="form-row">
              <div class="form-group">
                <label class="form-label">Fashion Score Threshold</label>
                <input class="form-input" name="fashion_threshold" type="number" step="0.05" value="0.55" min="0" max="1">
              </div>
              <div class="form-group">
                <label class="form-label">Min Resolution (px)</label>
                <input class="form-input" name="min_shorter_side" type="number" value="512" min="128" max="2048">
              </div>
            </div>
          </div>
          <div class="form-section">
            <div class="form-section-title">Analysis Settings</div>
            <div class="form-row">
              <div class="form-group">
                <label class="form-label">Palette Colors (k)</label>
                <input class="form-input" name="palette_k" type="number" value="8" min="3" max="16">
              </div>
              <div class="form-group">
                <label class="form-label">Cluster Range</label>
                <div style="display:flex;gap:8px;align-items:center">
                  <input class="form-input" name="clusters_min" type="number" value="3" min="2" max="10" style="width:70px">
                  <span style="color:var(--text-muted)">to</span>
                  <input class="form-input" name="clusters_max" type="number" value="6" min="2" max="10" style="width:70px">
                </div>
              </div>
            </div>
          </div>
        </div>
        <div class="form-actions">
          <button type="button" class="btn btn-secondary" onclick="navigate('#/')">Cancel</button>
          <button type="submit" class="btn btn-primary" id="create-btn">Create Brand</button>
        </div>
      </form>
    </div>`;
}

// ─── Actions ─────────────────────────────────────────────────────────────
async function handleCreateBrand(e) {
  e.preventDefault();
  const f = e.target;
  const btn = document.getElementById('create-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Creating...';

  const social = {};
  if (f.instagram.value.trim()) social.instagram = f.instagram.value.trim();
  if (f.pinterest.value.trim()) social.pinterest = f.pinterest.value.trim();

  const body = {
    name: f.name.value.trim(),
    url: f.url.value.trim(),
    social,
    known_categories: f.categories.value.split(',').map(s => s.trim()).filter(Boolean),
    seed_pages: f.seed_pages.value.split('\n').map(s => s.trim()).filter(Boolean),
    notes: f.notes.value.trim(),
    crawl: {
      max_pages: +f.max_pages.value,
      delay_ms: +f.delay_ms.value,
      max_concurrency: +f.max_concurrency.value,
      render_js: f.render_js.value === 'true',
    },
    filter: {
      fashion_score_threshold: +f.fashion_threshold.value,
      min_shorter_side: +f.min_shorter_side.value,
    },
    analysis: {
      palette_k: +f.palette_k.value,
      n_aesthetic_clusters_min: +f.clusters_min.value,
      n_aesthetic_clusters_max: +f.clusters_max.value,
    },
  };

  try {
    const res = await api('/brands', { method: 'POST', body });
    toast('Brand created successfully!', 'success');
    navigate(`#/brand/${res.slug}`);
  } catch (err) {
    toast(err.message, 'error');
    btn.disabled = false;
    btn.textContent = 'Create Brand';
  }
}

async function startRun(slug) {
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Starting...';
  try {
    await api(`/brands/${slug}/run`, { method: 'POST' });
    toast('Run started! This may take 6-10 minutes.', 'success');
    setTimeout(() => route(), 3000);
  } catch (err) {
    toast(err.message, 'error');
    btn.disabled = false;
    btn.textContent = '▶ Run Agent';
  }
}

async function deleteBrand(slug) {
  if (!confirm(`Delete brand "${slug}" and its config?`)) return;
  try {
    await api(`/brands/${slug}`, { method: 'DELETE' });
    toast('Brand deleted', 'info');
    navigate('#/');
  } catch (err) {
    toast(err.message, 'error');
  }
}

function toggleCollapsible(btn) {
  btn.classList.toggle('open');
  btn.nextElementSibling.classList.toggle('open');
}

// ─── Utilities ───────────────────────────────────────────────────────────
function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }

function timeAgo(iso) {
  if (!iso) return '';
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

// ─── Init ────────────────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const s = await api('/status');
    const dot = document.querySelector('.status-dot');
    const txt = document.querySelector('.status-text');
    if (s.api_key_set) {
      dot.classList.add('ok');
      txt.textContent = 'API Connected';
    } else {
      dot.classList.add('err');
      txt.textContent = 'No API Key';
    }
  } catch {
    document.querySelector('.status-dot').classList.add('err');
    document.querySelector('.status-text').textContent = 'Offline';
  }
}

document.getElementById('hamburger').addEventListener('click', () => document.getElementById('sidebar').classList.add('open'));
document.getElementById('sidebar-close').addEventListener('click', () => document.getElementById('sidebar').classList.remove('open'));
window.addEventListener('hashchange', route);
checkStatus();
route();
