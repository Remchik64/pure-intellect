// ================================================================
// PURE INTELLECT ADMIN - JavaScript Core
// ================================================================

'use strict';

const HOST = window.location.host || 'localhost:7860';
const API = `http://${HOST}`;
const WS_URL = `ws://${HOST}/ws`;

// ---- State ----
let ws = null;
let wsConnected = false;
let chatHistory = [];
let isTyping = false;
let allFacts = { coords: [], anchors: [], hot: [] };
let currentSection = 'chat';
let activeSessionId = 'default';
let sessionsList = [];

// ================================================================
// NAVIGATION
// ================================================================

function showSection(name) {
  if (typeof stopLogsAutoRefresh === 'function') stopLogsAutoRefresh();
  document.querySelectorAll('.section').forEach(s => {
    s.style.display = 'none';
    s.classList.remove('active');
  });
  document.querySelectorAll('.header-nav-item').forEach(n => n.classList.remove('active'));

  const sec = document.getElementById('section-' + name);
  if (!sec) return;

  if (name === 'chat') {
    sec.style.display = 'flex';
    sec.style.flexDirection = 'column';
    sec.style.height = '100%';
  } else {
    sec.style.display = 'block';
  }
  sec.classList.add('active');

  const navEl = document.querySelector(`.header-nav-item[data-section="${name}"]`);
  if (navEl) navEl.classList.add('active');

  currentSection = name;

  // Close mobile sidebar
  const sidebar = document.getElementById('sidebar');
  if (sidebar.classList.contains('open')) sidebar.classList.remove('open');

  // Stop models polling if leaving models section
  if (name !== 'models') stopModelsPolling();

  // Load section data
  switch (name) {
    case 'chat': loadChatStats(); break;
    case 'memory': loadMemory(); break;
    case 'models': loadModels(); startModelsPolling(); break;
    case 'settings': loadSettings(); break;
    case 'logs': startLogsAutoRefresh(); break;
  }
}

function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
}

// ================================================================
// SESSION MANAGEMENT
// ================================================================

async function loadSessions() {
  const data = await api('/api/v1/sessions');
  if (!data) return;

  activeSessionId = data.active_session_id || 'default';
  sessionsList = data.sessions || [];

  const listEl = document.getElementById('chat-list');
  if (!listEl) return;

  if (!sessionsList.length) {
    listEl.innerHTML = '<div class="empty-state" style="padding:16px;text-align:center;color:var(--text-dim);font-size:12px;">Нет чатов</div>';
    return;
  }

  listEl.innerHTML = sessionsList.map(s => {
    const id = esc(s.session_id || s.id);
    const name = esc(s.display_name || s.name || 'New Chat');
    const icon = s.icon || '💬';
    const isActive = s.session_id === activeSessionId || s.id === activeSessionId;
    const isDefault = s.session_id === 'default' || s.id === 'default' || s.session_type === 'default';
    return `<div class="chat-item${isActive ? ' active' : ''}" data-session-id="${id}" onclick="switchChat('${id}')">
      <span class="chat-item-icon">${icon}</span>
      <span class="chat-item-name">${name}</span>
      ${!isDefault ? `<button class="chat-item-menu-btn" onclick="showChatMenu('${id}', event)">⋮</button>
      <div class="chat-item-menu" id="menu-${id}">
        <button onclick="startRenameChat('${id}', '${esc(s.display_name || s.name || 'New Chat')}', event)">✏️ Rename</button>
        <button class="menu-danger" onclick="deleteChat('${id}', event)">🗑️ Delete</button>
      </div>` : ''}
    </div>`;
  }).join('');
}

async function createNewChat() {
  const data = await api('/api/v1/sessions', {
    method: 'POST',
    body: JSON.stringify({ display_name: 'New Chat' })
  });
  if (data) {
    toast('Чат создан', 'success');
    await loadSessions();
    showSection('chat');
  } else {
    toast('Ошибка создания чата', 'error');
  }
}

async function switchChat(id) {
  if (id === activeSessionId) return;
  const data = await api(`/api/v1/sessions/${encodeURIComponent(id)}/switch`, {
    method: 'POST'
  });
  if (data !== null) {
    activeSessionId = id;
    // Clear current messages from UI
    document.getElementById('messages').innerHTML = '';
    // Load chat_history from switch response into chat UI
    if (data.chat_history && data.chat_history.length > 0) {
      data.chat_history.forEach(msg => {
        appendMessage(msg.role || 'user', msg.content || '');
      });
    }
    // Update turn/CCI badges
    if (data.turn != null) {
      document.getElementById('chat-turn-badge').textContent = `Turn: ${data.turn}`;
    }
    // Reload stats (CCI, model)
    loadChatStats();
    // Reload memory data for new session (always, not just on memory tab)
    loadMemory();
    await loadSessions();
    showSection('chat');
  } else {
    toast('Ошибка переключения чата', 'error');
  }
}

async function deleteChat(id, event) {
  if (event) { event.stopPropagation(); event.preventDefault(); }
  closeAllMenus();
  if (!confirm('Удалить этот чат?')) return;
  const r = await api(`/api/v1/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' });
  if (r !== null) {
    toast('Чат удалён', 'success');
    document.getElementById('messages').innerHTML = '';
    await loadSessions();
  } else {
    toast('Ошибка удаления чата', 'error');
  }
}

function startRenameChat(id, name, event) {
  if (event) { event.stopPropagation(); event.preventDefault(); }
  closeAllMenus();

  const chatItem = document.querySelector(`.chat-item[data-session-id="${id}"]`);
  if (!chatItem) return;

  const nameSpan = chatItem.querySelector('.chat-item-name');
  if (!nameSpan) return;

  // Replace name span with input
  const input = document.createElement('input');
  input.className = 'chat-item-edit';
  input.value = name;
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      finishRenameChat(id, e);
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      loadSessions(); // Cancel rename
    }
  });
  input.addEventListener('blur', (e) => {
    finishRenameChat(id, e);
  });

  nameSpan.replaceWith(input);
  input.focus();
  input.select();
}

async function finishRenameChat(id, event) {
  if (event) event.stopPropagation();
  const chatItem = document.querySelector(`.chat-item[data-session-id="${id}"]`);
  if (!chatItem) return;

  const input = chatItem.querySelector('.chat-item-edit');
  if (!input) return;

  const newName = input.value.trim();
  if (!newName) {
    loadSessions();
    return;
  }

  const r = await api(`/api/v1/sessions/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    body: JSON.stringify({ display_name: newName })
  });
  if (r !== null) {
    toast('Чат переименован', 'success');
    await loadSessions();
  } else {
    toast('Ошибка переименования', 'error');
    loadSessions();
  }
}

function showChatMenu(id, event) {
  if (event) { event.stopPropagation(); event.preventDefault(); }
  closeAllMenus();
  const menu = document.getElementById('menu-' + id);
  if (menu) menu.classList.add('show');
}

function closeAllMenus() {
  document.querySelectorAll('.chat-item-menu.show').forEach(m => m.classList.remove('show'));
}

// Close menus on outside click
document.addEventListener('click', (e) => {
  if (!e.target.closest('.chat-item-menu') && !e.target.closest('.chat-item-menu-btn')) {
    closeAllMenus();
  }
});

// ================================================================
// UTILITIES
// ================================================================

async function api(path, opts = {}) {
  try {
    const res = await fetch(API + path, {
      headers: { 'Content-Type': 'application/json', ...opts.headers },
      ...opts
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    console.warn('API error', path, e.message);
    return null;
  }
}

async function apiOllama(path) {
  try {
    const res = await fetch(`http://${HOST.split(':')[0]}:11434${path}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    // Try localhost:11434 directly
    try {
      const res2 = await fetch(`http://localhost:11434${path}`);
      if (!res2.ok) throw new Error(`HTTP ${res2.status}`);
      return await res2.json();
    } catch (e2) {
      console.warn('Ollama API error', path, e2.message);
      return null;
    }
  }
}

function toast(msg, type = 'info') {
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  const icons = { info: 'ℹ️', success: '✅', error: '❌', warning: '⚠️' };
  t.className = `toast ${type}`;
  t.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${esc(msg)}</span>`;
  c.appendChild(t);
  setTimeout(() => t.remove(), 2700);
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function fmtSize(bytes) {
  if (!bytes) return '—';
  const gb = bytes / 1e9;
  if (gb >= 1) return gb.toFixed(1) + ' GB';
  const mb = bytes / 1e6;
  return mb.toFixed(0) + ' MB';
}

function fmtDate(s) {
  if (!s) return '—';
  try {
    return new Date(s).toLocaleDateString('ru-RU', { day:'2-digit', month:'2-digit', year:'2-digit', hour:'2-digit', minute:'2-digit' });
  } catch { return s; }
}

function fmtTime() {
  return new Date().toLocaleTimeString('ru-RU', { hour:'2-digit', minute:'2-digit', second:'2-digit' });
}

function copyText(text) {
  navigator.clipboard.writeText(text).then(() => toast('Скопировано!', 'success')).catch(() => {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
    toast('Скопировано!', 'success');
  });
}

function copyJSON(obj) {
  copyText(JSON.stringify(obj, null, 2));
}

function pct(v, max = 100) {
  if (v == null || isNaN(v)) return 0;
  return Math.min(100, Math.max(0, (v / max) * 100));
}

function colorClass(v) {
  if (v < 60) return 'green';
  if (v < 85) return 'yellow';
  return 'red';
}

// Simple markdown → HTML (no library)
function renderMarkdown(text) {
  if (!text) return '';
  let s = esc(text);

  // Code blocks
  s = s.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);
  // Inline code
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  // Italic
  s = s.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  // Headers
  s = s.replace(/^### (.+)$/gm, '<h4>$1</h4>');
  s = s.replace(/^## (.+)$/gm, '<h3>$1</h3>');
  s = s.replace(/^# (.+)$/gm, '<h2>$1</h2>');
  // Lists
  s = s.replace(/^[*-] (.+)$/gm, '<li>$1</li>');
  s = s.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
  // Line breaks (outside pre)
  s = s.replace(/\n/g, '<br>');

  return s;
}

// ================================================================
// WEBSOCKET
// ================================================================

function connectWS() {
  if (ws && ws.readyState < 2) return;
  try {
    ws = new WebSocket(WS_URL);
  } catch (e) {
    setWSStatus(false);
    return;
  }

  ws.onopen = () => {
    wsConnected = true;
    setWSStatus(true);
  };

  ws.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      handleWSMessage(msg);
    } catch {
      // plain text
      appendMessage('assistant', e.data);
    }
  };

  ws.onclose = () => {
    wsConnected = false;
    setWSStatus(false);
    setTimeout(connectWS, 4000);
  };

  ws.onerror = () => {
    wsConnected = false;
    setWSStatus(false);
  };
}

// Текущий streaming bubble (для накопления токенов)
let _streamBubble = null;
let _streamText = '';
let _thinkingBubble = null;
let _thinkingText = '';

function startStreamBubble() {
  hideTyping();
  const el = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.innerHTML = `
    <div class="avatar">🧠</div>
    <div>
      <div class="bubble stream-bubble"><span class="stream-cursor">▋</span></div>
      <div class="time">${fmtTime()}</div>
    </div>`;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
  _streamBubble = div.querySelector('.stream-bubble');
  _streamText = '';
}

function appendStreamToken(token) {
  if (!_streamBubble) startStreamBubble();
  _streamText += token;
  _streamBubble.innerHTML = renderMarkdown(_streamText) + '<span class="stream-cursor">▋</span>';
  const el = document.getElementById('messages');
  el.scrollTop = el.scrollHeight;
}

function finalizeStreamBubble() {
  if (_streamBubble) {
    _streamBubble.innerHTML = renderMarkdown(_streamText);
    _streamBubble = null;
    _streamText = '';
  }
  hideTyping();
}


function createThinkingBubble() {
  const chatMessages = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'message assistant thinking-message';
  div.innerHTML = `
    <div class="thinking-header" onclick="toggleThinking(this)">
      <span class="thinking-icon">🧠</span>
      <span class="thinking-label">🔄 Думаю...</span>
      <span class="thinking-toggle">▼</span>
    </div>
    <div class="thinking-content"></div>
  `;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return div.querySelector('.thinking-content');
}

function appendThinkingToken(token) {
  if (!_thinkingBubble) return;
  _thinkingText += token;
  _thinkingBubble.textContent = _thinkingText;
  const chatMessages = document.getElementById('messages');
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function finalizeThinkingBubble() {
  if (_thinkingBubble) {
    const header = _thinkingBubble.parentElement.querySelector('.thinking-label');
    if (header) header.innerHTML = '💭 Мысли <small style="opacity:0.6">(нажми чтобы развернуть)</small>';
    const toggle = _thinkingBubble.parentElement.querySelector('.thinking-toggle');
    if (toggle) toggle.textContent = '▶';
    _thinkingBubble.style.display = 'none';
    _thinkingBubble = null;
    _thinkingText = '';
  }
}

function toggleThinking(header) {
  const content = header.nextElementSibling;
  const toggle = header.querySelector('.thinking-toggle');
  const label = header.querySelector('.thinking-label');
  if (!content) return;
  if (content.style.display === 'none' || content.style.display === '') {
    content.style.display = 'block';
    if (toggle) toggle.textContent = '▼';
    if (label) label.innerHTML = '💭 Мысли <small style="opacity:0.6">(свернуть)</small>';
  } else {
    content.style.display = 'none';
    if (toggle) toggle.textContent = '▶';
    if (label) label.innerHTML = '💭 Мысли <small style="opacity:0.6">(развернуть)</small>';
  }
}

function handleWSMessage(msg) {
  if (msg.type === 'start') {
    showTyping();
    _streamBubble = null;
    _streamText = '';
  } else if (msg.type === 'token') {
    appendStreamToken(msg.content || '');
  } else if (msg.type === 'end') {
    finalizeStreamBubble();
    updateChatStats(msg);
    // Обновляем статистику из API (turn, cci могут быть точнее)
    loadChatStats();
  } else if (msg.type === 'reset_marker') {
    appendResetDivider(msg.turn || 0);
  } else if (msg.type === 'response' || msg.type === 'message') {
    hideTyping();
    appendMessage('assistant', msg.content || msg.message || '');
  } else if (msg.type === 'status') {
    updateChatStats(msg);
  } else if (msg.type === 'thinking_start') {
    hideTyping();
    _thinkingBubble = createThinkingBubble();
    _thinkingText = '';
  } else if (msg.type === 'thinking') {
    if (_thinkingBubble) appendThinkingToken(msg.content || '');
  } else if (msg.type === 'thinking_end') {
    if (_thinkingBubble) finalizeThinkingBubble();
  } else if (msg.type === 'error') {
    finalizeStreamBubble();
    hideTyping();
    appendMessage('assistant', '⚠️ Ошибка: ' + (msg.message || 'Unknown'));
  }
}

function setWSStatus(online) {
  const dot = document.getElementById('ws-dot');
  const label = document.getElementById('ws-label');
  if (online) {
    dot.className = 'dot green';
    label.textContent = 'Online';
  } else {
    dot.className = 'dot red';
    label.textContent = 'Offline';
  }
}


function showToast(msg) {
  let t = document.getElementById('pi-toast');
  if (!t) { t = document.createElement('div'); t.id = 'pi-toast'; t.style.cssText = 'position:fixed;bottom:24px;right:24px;background:var(--accent);color:#000;padding:10px 18px;border-radius:8px;font-size:14px;z-index:9999;transition:opacity 0.3s'; document.body.appendChild(t); }
  t.textContent = msg; t.style.opacity = '1';
  setTimeout(() => { t.style.opacity = '0'; }, 2500);
}


// ================================================================
// CHAT
// ================================================================

function appendMessage(role, content) {
  const el = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = `message ${role}`;
  const isUser = role === 'user';
  const avatar = isUser ? '👤' : '🧠';
  const time = fmtTime();
  div.innerHTML = `
    <div class="avatar">${avatar}</div>
    <div>
      <div class="bubble">${isUser ? esc(content) : renderMarkdown(content)}</div>
      <div class="time">${time}</div>
    </div>`;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

function appendResetDivider(turn) {
  const el = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'reset-divider';
  div.innerHTML = `<span class="reset-icon">🔄</span> Контекст обновлён (Turn ${turn}) — факты сохранены в памяти`;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

function showTyping() {
  document.getElementById('typing-area').style.display = 'block';
  isTyping = true;
  const el = document.getElementById('messages');
  el.scrollTop = el.scrollHeight;
}

function hideTyping() {
  document.getElementById('typing-area').style.display = 'none';
  isTyping = false;
}

function sendMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;
  if (!wsConnected) {
    toast('WebSocket не подключён', 'error');
    connectWS();
    return;
  }
  appendMessage('user', text);
  input.value = '';
  input.style.height = 'auto';
  showTyping();
  try {
    const sf = document.getElementById('force-web-search');
    const forceWebSearch = sf ? sf.checked : false;
    if (sf) sf.checked = false; // Auto-disable
    ws.send(JSON.stringify({ type: 'chat', message: text, content: text, force_web_search: forceWebSearch }));
  } catch (e) {
    hideTyping();
    toast('Ошибка отправки: ' + e.message, 'error');
  }

  // Auto-name: if current chat is named "New Chat", rename with first 30 chars
  const activeItem = document.querySelector('.chat-item.active .chat-item-name');
  if (activeItem && activeItem.textContent === 'New Chat' && activeSessionId !== 'default') {
    const autoName = text.substring(0, 30).replace(/[\n\r]/g, ' ').trim();
    if (autoName) {
      api(`/api/v1/sessions/${encodeURIComponent(activeSessionId)}`, {
        method: 'PATCH',
        body: JSON.stringify({ display_name: autoName })
      }).then(() => {
        loadSessions();
      });
    }
  }
}

function updateChatStats(msg) {
  if (msg.turn != null) document.getElementById('chat-turn-badge').textContent = `Turn: ${msg.turn}`;
  if (msg.cci != null) document.getElementById('chat-cci-badge').textContent = `CCI: ${(msg.cci * 100).toFixed(0)}%`;
  if (msg.model) document.getElementById('chat-model-badge').textContent = `Model: ${msg.model}`;
}

async function loadChatStats() {
  const [cciData, sessData, dualData] = await Promise.all([
    api('/api/v1/cci/stats'),
    api('/api/v1/session/info'),
    api('/api/v1/dual-model/stats')
  ]);
  if (cciData) {
    const score = cciData.current_score ?? cciData.score ?? cciData.cci ?? 0;
    document.getElementById('chat-cci-badge').textContent = `CCI: ${(score * 100).toFixed(0)}%`;
  }
  if (sessData) {
    const turns = sessData.turn_count ?? sessData.turns ?? 0;
    document.getElementById('chat-turn-badge').textContent = `Turn: ${turns}`;
  }
  if (dualData) {
    const gen = dualData.generator ?? dualData.gen ?? {};
    const name = gen.model ?? gen.name ?? '—';
    document.getElementById('chat-model-badge').textContent = `Model: ${name}`;
  }
}

async function clearSession() {
  if (!confirm('Очистить текущую сессию?')) return;
  const r = await api('/api/v1/session', { method: 'DELETE' });
  if (r !== null) {
    document.getElementById('messages').innerHTML = '';
    toast('Сессия очищена', 'success');
  } else {
    toast('Ошибка очистки', 'error');
  }
}

// Auto-resize textarea & version badge
document.addEventListener('DOMContentLoaded', () => {
  const ta = document.getElementById('chat-input');
  ta.addEventListener('input', () => {
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 140) + 'px';
  });
  ta.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Fetch version from API and update badge
  fetch('/api/v1/version')
    .then(r => r.json())
    .then(data => {
      const ver = data.version || 'dev';
      const badge = document.getElementById('version-badge');
      if (badge) badge.textContent = 'v' + ver.replace(/^v/, '');
    })
    .catch(() => {});  // silently ignore — keep default v0.2
});

// ================================================================
// MEMORY
// ================================================================

async function loadMemory() {
  const stats = await api('/api/v1/memory/stats');
  const factsData = await fetch('/api/v1/memory/facts').then(r => r.json()).catch(e => null);
  allFacts = { coords: [], anchors: [], hot: [] };

  if (factsData && factsData.facts) {
    const facts = factsData.facts;
    facts.forEach(f => {
      const src = f.source ?? '';
      const isAnchor = f.is_anchor ?? false;
      const isCoord = src === 'soft_reset' || src === 'coordinate' || f.type === 'coordinate';
      if (isCoord) allFacts.coords.push(f);
      else if (isAnchor) allFacts.anchors.push(f);
      else allFacts.hot.push(f);
    });
  }

  renderMemory();
}

function renderMemory(filter = '') {
  const q = filter.toLowerCase();
  const renderList = (facts, listId, countId) => {
    const el = document.getElementById(listId);
    const cnt = document.getElementById(countId);
    const filtered = q ? facts.filter(f => (f.content || f.text || '').toLowerCase().includes(q)) : facts;
    cnt.textContent = filtered.length;
    if (!filtered.length) {
      el.innerHTML = '<div class="empty-state">Нет данных</div>';
      return;
    }
    el.innerHTML = filtered.map((f, i) => {
      const text = f.content ?? f.text ?? JSON.stringify(f);
      const imp = f.importance ?? f.score ?? 0.5;
      const impClass = imp > 0.7 ? 'high' : imp > 0.4 ? 'med' : 'low';
      const impLabel = imp > 0.7 ? '🔴' : imp > 0.4 ? '🟡' : '⚪';
      return `<div class="fact-item" id="fact-${listId}-${i}">
        <div class="fact-text">${esc(text)}</div>
        <div class="fact-meta">
          <span class="importance-badge ${impClass}">${impLabel} ${imp.toFixed ? imp.toFixed(2) : imp}</span>
          <button class="del-fact-btn" onclick="deleteFact('${listId}', ${i})">×</button>
        </div>
      </div>`;
    }).join('');
  };

  renderList(allFacts.coords, 'list-coords', 'count-coords');
  renderList(allFacts.anchors, 'list-anchors', 'count-anchors');
  renderList(allFacts.hot, 'list-hot', 'count-hot');
}

function filterMemory() {
  const q = document.getElementById('memory-search').value;
  renderMemory(q);
}

async function deleteFact(listId, idx) {
  // Remove locally (API for delete by content/id would need backend support)
  const map = { 'list-coords': 'coords', 'list-anchors': 'anchors', 'list-hot': 'hot' };
  const key = map[listId];
  if (key) {
    allFacts[key].splice(idx, 1);
    renderMemory(document.getElementById('memory-search').value);
    toast('Факт удалён (локально)', 'info');
  }
}

async function clearMemory() {
  if (!confirm('Очистить всю рабочую память?')) return;
  const r = await api('/api/v1/memory/clear', { method: 'POST' });
  if (r !== null) {
    toast('Память очищена', 'success');
    loadMemory();
  } else {
    toast('Ошибка очистки памяти', 'error');
  }
}

// ================================================================
// MODELS
// ================================================================

function updateDualModelUI(stats, prefix) {
  const coord = stats.coordinator ?? stats.coord ?? {};
  const gen = stats.generator ?? stats.gen ?? {};

  const coordName = coord.model ?? coord.name ?? '—';
  const genName = gen.model ?? gen.name ?? '—';

  function resolveStatus(role) {
    if (role.status === 'active')  return 'active';
    if (role.status === 'ready')   return 'ready';
    if (role.status === 'offline') return 'offline';
    const ok = role.ready ?? role.loaded ?? false;
    return ok ? 'ready' : 'offline';
  }

  const STATUS_LABEL = {
    active:  '🔥 В памяти',
    ready:   '✅ Готова',
    offline: '❌ Не скачана',
  };

  const coordStatus = resolveStatus(coord);
  const genStatus   = resolveStatus(gen);

  const cn = document.getElementById(`${prefix}-coord-name`);
  const gn = document.getElementById(`${prefix}-gen-name`);
  const cd = document.getElementById(`${prefix}-coord-dot`);
  const gd = document.getElementById(`${prefix}-gen-dot`);
  const cs = document.getElementById(`${prefix}-coord-status`);
  const gs = document.getElementById(`${prefix}-gen-status`);

  if (cn) cn.textContent = coordName;
  if (gn) gn.textContent = genName;
  if (cd) cd.className = 'status-dot ' + coordStatus;
  if (gd) gd.className = 'status-dot ' + genStatus;
  if (cs) cs.textContent = STATUS_LABEL[coordStatus] ?? coordStatus;
  if (gs) gs.textContent = STATUS_LABEL[genStatus]   ?? genStatus;
}

let _modelsStatusInterval = null;

async function loadModels() {
  // 1. Получаем полный статус моделей (скачанные + VRAM + назначенные)
  const statusData = await api('/api/v1/models/status');
  if (statusData && !statusData.error) {
    updateDualModelUI(statusData, 'm');
    window._modelsActiveInVram = new Set(statusData.active_in_vram ?? []);
    window._modelsDownloaded   = new Set(statusData.downloaded ?? []);
    window._modelsCoordinator  = statusData.coordinator?.model ?? '';
    window._modelsGenerator    = statusData.generator?.model ?? '';
  } else {
    const dualData = await api('/api/v1/dual-model/stats');
    if (dualData) updateDualModelUI(dualData, 'm');
  }

  // 2. Список скачанных моделей Ollama (через наш прокси)
  const ollamaData = await api('/api/v1/ollama/models');
  const tbody = document.getElementById('ollama-tbody');

  if (!ollamaData) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-state"><div class="empty-icon">⚠️</div>Ollama недоступен</td></tr>';
    return;
  }

  const models = ollamaData.models ?? [];
  if (!models.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-state">Нет моделей</td></tr>';
    return;
  }

  const active  = window._modelsActiveInVram ?? new Set();
  const coord   = window._modelsCoordinator ?? '';
  const gen     = window._modelsGenerator ?? '';

  function modelStatusBadge(name) {
    if (active.has(name))         return '<span style="color:#ff8c00" title="Загружена в VRAM">🔥 В памяти</span>';
    if (name === coord || name === gen) return '<span style="color:var(--accent)" title="Скачана, назначена">✅ Готова</span>';
    return '<span style="color:var(--text-dim)" title="Скачана, не назначена">💾 Скачана</span>';
  }

  function roleTag(name) {
    const tags = [];
    if (name === coord) tags.push('<span style="font-size:10px;background:rgba(88,166,255,.2);color:var(--accent);padding:1px 6px;border-radius:4px;">🎯 Координатор</span>');
    if (name === gen)   tags.push('<span style="font-size:10px;background:rgba(188,140,255,.2);color:#bc8cff;padding:1px 6px;border-radius:4px;">⚡ Генератор</span>');
    return tags.join(' ');
  }
  tbody.innerHTML = models.map(m => {
    const name     = m.name ?? m.model ?? '—';
    const size     = fmtSize(m.size);
    const modified = fmtDate(m.modified_at ?? m.modified);
    return `<tr class="ollama-model-row">
      <td><strong>${esc(name)}</strong><br><small>${roleTag(name)}</small></td>
      <td class="text-dim">${size}</td>
      <td class="text-dim">${modified}</td>
      <td>${modelStatusBadge(name)}</td>
      <td>
        <div class="btn-group">
          <button class="btn" onclick="setModel('coordinator','${esc(name)}')">🎯 Координатор</button>
          <button class="btn" onclick="setModel('generator','${esc(name)}')">⚡ Генератор</button>
          <button class="btn" style="background:rgba(255,80,80,.15);color:#ff5050;border-color:rgba(255,80,80,.3)" onclick="deleteModel('${esc(name)}')">🗑️ Удалить</button>
        </div>
      </td>
    </tr>`;
  }).join('');
}


// Запускаем/останавливаем polling статусов при переключении секций
function startModelsPolling() {
  if (_modelsStatusInterval) return;
  _modelsStatusInterval = setInterval(loadModels, 10000);
}
function stopModelsPolling() {
  if (_modelsStatusInterval) { clearInterval(_modelsStatusInterval); _modelsStatusInterval = null; }
}

async function setModel(role, name) {
  // POST /api/v1/models/switch — прямое переключение без перезапуска
  const r = await api('/api/v1/models/switch', {
    method: 'POST',
    body: JSON.stringify({ role, model: name })
  });
  if (r !== null && r.status === 'switched') {
    toast(`${role === 'coordinator' ? '🎯 Координатор' : '⚡ Генератор'} → ${name}`, 'success');
    loadModels();  // обновим статусы сразу после смены
  } else {
    // Fallback: обновляем поле в настройках
    if (role === 'coordinator') { const el = document.getElementById('cfg-coord'); if (el) el.value = name; }
    else                        { const el = document.getElementById('cfg-gen');   if (el) el.value = name; }
    toast(`Модель выбрана: ${name}. Нажмите «Сохранить настройки».`, 'info');
    showSection('settings');
  }
}

async function deleteModel(name) {
  if (!confirm(`Удалить модель "${name}" с диска?\n\nЭто действие нельзя отменить!`)) return;
  const r = await api(`/api/v1/models/${encodeURIComponent(name)}`, { method: 'DELETE' });
  if (r && r.status === 'deleted') {
    toast(`🗑️ Модель "${name}" удалена`, 'success');
    loadModels();
  } else {
    toast(`Ошибка удаления: ${r?.detail ?? 'неизвестная ошибка'}`, 'error');
  }
}


// ================================================================
// HARDWARE DETECTION
// ================================================================

let _hwData = null;  // stores last hardware detection result

async function detectHardware() {
  const btn = document.querySelector('button[onclick="detectHardware()"]');
  if (btn) { btn.disabled = true; btn.textContent = '⏳ Определяем...'; }

  try {
    const data = await api('/api/v1/hardware/detect');
    if (!data) {
      toast('Не удалось получить данные о железе', 'error');
      return;
    }
    _hwData = data;

    // API response structure:
    // { hardware: {gpu, vram_gb, ram_gb, os, cpu, ...},
    //   recommendation: {coordinator, generator, mode, speed_estimate, status, ...} }
    console.log('[detectHardware] raw response:', JSON.stringify(data, null, 2));

    const hw  = data.hardware  ?? {};
    const rec = data.recommendation ?? data.rec ?? {};

    // GPU info — hw.gpu is a string e.g. "NVIDIA RTX 3080"
    const gpuName = hw.gpu ?? hw.gpu_name ?? hw.gpu_model ?? 'Net GPU';
    const vram    = hw.vram_gb ?? hw.gpu_vram_gb ?? null;
    document.getElementById('hw-gpu').textContent  = gpuName;
    document.getElementById('hw-vram').textContent = vram != null ? vram + ' GB' : '';

    // RAM + OS
    const ram = hw.ram_gb ?? hw.ram ?? null;
    const os  = hw.os ?? hw.platform ?? '—';
    document.getElementById('hw-ram').textContent = ram != null ? Number(ram).toFixed(1) + ' GB' : '—';
    document.getElementById('hw-os').textContent  = os;

    // Recommendation — rec.mode e.g. "GPU FULL", rec.status e.g. "✅ Отлично"
    const recLabel = rec.status ?? rec.mode ?? rec.tier ?? rec.label ?? '—';
    document.getElementById('hw-rec-label').textContent = recLabel;
    document.getElementById('hw-rec-coord').textContent = rec.coordinator ?? rec.coordinator_model ?? '—';
    document.getElementById('hw-rec-gen').textContent   = rec.generator   ?? rec.generator_model   ?? '—';
    const speed = rec.speed_estimate ?? rec.speed ?? rec.tokens_per_sec ?? null;
    document.getElementById('hw-rec-speed').textContent = speed ? speed : '—';

    document.getElementById('hw-result-card').style.display = '';
    toast('Железо определено!', 'success');
  } catch (e) {
    toast('Ошибка определения железа: ' + e.message, 'error');
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = '🔍 Определить железо'; }
  }
}

async function applyRecommendedModels() {
  if (!_hwData) { toast('Сначала определите железо', 'warning'); return; }
  const rec = _hwData.recommendation ?? _hwData.rec ?? {};
  const coord = rec.coordinator ?? rec.coordinator_model;
  const gen   = rec.generator   ?? rec.generator_model;

  if (!coord && !gen) {
    toast('Нет рекомендаций для применения', 'warning');
    return;
  }

  const btn = document.getElementById('hw-apply-btn');
  if (btn) { btn.disabled = true; btn.textContent = '⏳ Применяем...'; }

  let ok = 0;
  try {
    if (coord) {
      const r1 = await api('/api/v1/models/switch', {
        method: 'POST',
        body: JSON.stringify({ role: 'coordinator', model: coord })
      });
      if (r1 !== null) ok++;
    }
    if (gen) {
      const r2 = await api('/api/v1/models/switch', {
        method: 'POST',
        body: JSON.stringify({ role: 'generator', model: gen })
      });
      if (r2 !== null) ok++;
    }
    if (ok > 0) {
      toast(`✅ Применены рекомендованные модели (${ok})`, 'success');
      loadModels();
    } else {
      toast('Не удалось применить модели — проверьте API', 'error');
    }
  } catch (e) {
    toast('Ошибка: ' + e.message, 'error');
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = '✅ Применить рекомендованные модели'; }
  }
}

// ================================================================
// MODEL DOWNLOAD
// ================================================================

let _dlPollTimer = null;

async function downloadModel() {
  const input = document.getElementById('dl-model-input');
  const modelName = (input?.value ?? '').trim();
  if (!modelName) { toast('Введите название модели', 'warning'); return; }

  const progressWrap = document.getElementById('dl-progress-wrap');
  const progressBar  = document.getElementById('dl-progress-bar');
  const statusText   = document.getElementById('dl-status-text');
  const successWrap  = document.getElementById('dl-success-wrap');
  const dlBtn        = document.querySelector('button[onclick="downloadModel()"]');

  // Reset UI
  progressWrap.style.display = '';
  successWrap.style.display  = 'none';
  progressBar.style.width    = '5%';
  statusText.textContent     = `Скачивание ${modelName}...`;
  if (dlBtn) { dlBtn.disabled = true; }

  // Stop any existing poll
  if (_dlPollTimer) { clearInterval(_dlPollTimer); _dlPollTimer = null; }

  try {
    const r = await api('/api/v1/models/download', {
      method: 'POST',
      body: JSON.stringify({ model: modelName })
    });
    if (r === null) {
      statusText.textContent = '❌ Ошибка запуска скачивания';
      if (dlBtn) dlBtn.disabled = false;
      return;
    }
    toast(`Скачивание ${modelName} запущено...`, 'info');
  } catch (e) {
    statusText.textContent = '❌ ' + e.message;
    if (dlBtn) dlBtn.disabled = false;
    return;
  }

  // Poll progress every 3 seconds
  let elapsed = 0;
  _dlPollTimer = setInterval(async () => {
    elapsed += 3;
    try {
      const check = await api(`/api/v1/models/download/check/${encodeURIComponent(modelName)}`);
      if (!check) return;

      // Update progress bar — check.percent is real progress from Ollama streaming
      const pct = check.percent ?? check.progress ?? null;
      if (pct !== null && pct > 0) {
        progressBar.style.width = Math.min(Math.max(pct, 5), 99) + '%';
      } else {
        // Indeterminate animation until first real percent arrives
        const pseudo = Math.min(5 + elapsed * 1.2, 60);
        progressBar.style.width = pseudo + '%';
      }

      // Status text: show speed + percent when available
      const status = check.status ?? '';
      const speedStr = check.speed ? ` — ${check.speed}` : '';
      const pctStr   = pct != null && pct > 0 ? ` (${pct}%)` : '';
      const msgStr   = check.status_msg || check.message || `Скачиваем ${modelName}...`;
      statusText.textContent = msgStr + pctStr + speedStr;

      // Stop polling on error
      if (status === 'error') {
        clearInterval(_dlPollTimer);
        _dlPollTimer = null;
        statusText.textContent = '❌ Ошибка: ' + (check.error || 'неизвестно');
        if (dlBtn) dlBtn.disabled = false;
        return;
      }

      // Stop polling when done
      if (check.ready === true || status === 'done' || status === 'ready') {
        clearInterval(_dlPollTimer);
        _dlPollTimer = null;
        progressBar.style.width = '100%';
        statusText.textContent = '✅ Завершено!';
        setTimeout(() => { progressWrap.style.display = 'none'; }, 1500);
        successWrap.style.display = '';
        if (dlBtn) dlBtn.disabled = false;
        if (input) input.value = '';
        toast(`✅ Модель ${modelName} готова!`, 'success');
        loadModels();
      }
    } catch (e) {
      // Non-fatal poll error — keep trying
      statusText.textContent = `Ожидаем... (${elapsed}с)`;
    }
  }, 3000);
}


// ================================================================


// ================================================================
// SETTINGS
// ================================================================

async function loadSettings() {
  const data = await api('/api/v1/config');
  if (!data) { toast('Не удалось загрузить настройки', 'error'); return; }

  // Try various config shapes
  const coord = data.coordinator_model ?? data.coordinator?.model ?? data.models?.coordinator ?? '';
  const gen = data.generator_model ?? data.generator?.model ?? data.models?.generator ?? '';
  const cciThr = data.cci_threshold ?? data.cci?.threshold ?? data.threshold ?? 0.7;
  const maxTurns = data.max_turns_without_reset ?? data.adaptive_reset?.max_turns ?? data.max_turns ?? 20;
  const minTurns = data.min_turns_between_resets ?? data.adaptive_reset?.min_turns ?? data.min_turns ?? 5;
  const hotMax = data.hot_facts_max ?? data.memory?.hot_facts_max ?? data.max_hot_facts ?? 30;

  document.getElementById('cfg-coord').value = coord;
  document.getElementById('cfg-gen').value = gen;
  document.getElementById('cfg-cci').value = cciThr;
  document.getElementById('cfg-cci-val').textContent = cciThr;
  document.getElementById('cfg-max-turns').value = maxTurns;
  document.getElementById('cfg-min-turns').value = minTurns;
  document.getElementById('cfg-hot-max').value = hotMax;
}

async function saveSettings() {
  const payload = {
    coordinator_model: document.getElementById('cfg-coord').value,
    generator_model: document.getElementById('cfg-gen').value,
    cci_threshold: parseFloat(document.getElementById('cfg-cci').value),
    max_turns_without_reset: parseInt(document.getElementById('cfg-max-turns').value),
    min_turns_between_resets: parseInt(document.getElementById('cfg-min-turns').value),
    hot_facts_max: parseInt(document.getElementById('cfg-hot-max').value)
  };

  const r = await api('/api/v1/config/reload', { method: 'POST', body: JSON.stringify(payload) });
  if (r !== null) toast('Настройки сохранены', 'success');
  else {
    // Fallback: try PATCH /api/v1/config
    const r2 = await api('/api/v1/config', { method: 'PATCH', body: JSON.stringify(payload) });
    if (r2 !== null) toast('Настройки сохранены', 'success');
    else toast('Ошибка сохранения', 'error');
  }
}

// ================================================================
// INIT
// ================================================================

// Load chat history for active session (on F5 / page init)
async function loadChatHistory() {
  const messagesEl = document.getElementById('messages');
  if (!messagesEl || !activeSessionId) return;
  try {
    const data = await api(`/api/v1/sessions/${encodeURIComponent(activeSessionId)}/history`);
    if (data && data.chat_history && data.chat_history.length > 0) {
      data.chat_history.forEach(msg => {
        appendMessage(msg.role || 'user', msg.content || '');
      });
    }
  } catch (e) {
    console.warn('Failed to load chat history:', e);
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  connectWS();
  await loadSessions();
  await loadChatHistory();
  loadChatStats();
  loadMemory();  // Pre-load memory data for active session
  showSection('chat');
});

// Keyboard shortcut: Alt+1..7 → sections
document.addEventListener('keydown', e => {
  if (e.altKey) {
    const sections = ['chat','memory','models','settings','logs'];
    const idx = parseInt(e.key) - 1;
    if (idx >= 0 && idx < sections.length) showSection(sections[idx]);
  }
});


// ── LOGS ──────────────────────────────────────────────────────────────────────
let _logsAutoRefresh = null;
let _logsAutoScroll = true;
let _logsLastCount = 0;

async function loadLogs() {
  const level = document.getElementById('log-level-filter')?.value || 'ALL';
  const limit = parseInt(document.getElementById('log-limit')?.value || '500');
  try {
    const r = await fetch(`/api/v1/logs?level=${level}&limit=${limit}`);
    const data = await r.json();
    const terminal = document.getElementById('log-terminal');
    const counter = document.getElementById('log-counter');
    if (!terminal) return;
    const newCount = data.total;
    if (newCount !== _logsLastCount) {
      const lines = data.lines || [];
      terminal.innerHTML = lines.map(e => {
        const cls = 'log-line-' + (e.level || 'INFO');
        const escaped = e.line.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        return `<span class="${cls}">${escaped}</span>`;
      }).join('\n');
      if (_logsAutoScroll) terminal.scrollTop = terminal.scrollHeight;
      _logsLastCount = newCount;
    }
    if (counter) counter.textContent = `Строк: ${data.count} / Буфер: ${data.total}`;
  } catch(e) {
    console.error('Logs fetch error:', e);
  }
}

function startLogsAutoRefresh() {
  stopLogsAutoRefresh();
  loadLogs();
  _logsAutoRefresh = setInterval(loadLogs, 2000);
}

function stopLogsAutoRefresh() {
  if (_logsAutoRefresh) { clearInterval(_logsAutoRefresh); _logsAutoRefresh = null; }
}

function copyLogs() {
  const terminal = document.getElementById('log-terminal');
  if (!terminal) return;
  navigator.clipboard.writeText(terminal.innerText).then(() => {
    showToast('📋 Логи скопированы в буфер обмена', 'success');
  }).catch(() => {
    showToast('❌ Ошибка копирования', 'error');
  });
}

async function clearServerLogs() {
  try {
    await fetch('/api/v1/logs', {method: 'DELETE'});
    _logsLastCount = 0;
    document.getElementById('log-terminal').innerHTML = '';
    document.getElementById('log-counter').textContent = 'Строк: 0 / Буфер: 0';
    showToast('🗑️ Логи очищены', 'info');
  } catch(e) {
    showToast('❌ Ошибка очистки: ' + e.message, 'error');
  }
}

function toggleLogsAutoScroll() {
  _logsAutoScroll = !_logsAutoScroll;
  const btn = document.getElementById('log-autoscroll-btn');
  if (btn) btn.textContent = _logsAutoScroll ? '⬇️ Авто-скролл: ВКЛ' : '⬇️ Авто-скролл: ВЫКЛ';
}

