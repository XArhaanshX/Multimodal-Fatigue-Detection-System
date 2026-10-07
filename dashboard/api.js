// Shared helpers for the dashboard pages (backend/main.py serves them at /dashboard)
const API_BASE = location.protocol.startsWith('http') ? location.origin : 'http://localhost:8000';
const WS_BASE = API_BASE.replace(/^http/, 'ws');

function saveDraft(key, value) {
  try { sessionStorage.setItem(key, JSON.stringify(value)); } catch (e) {}
}

function loadDraft(key) {
  try { return JSON.parse(sessionStorage.getItem(key)) || null; } catch (e) { return null; }
}

async function apiPost(path, body) {
  const res = await fetch(API_BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function apiGet(path) {
  const res = await fetch(API_BASE + path);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function fullName(person) {
  return [person.firstName, person.lastName].filter(Boolean).join(' ').trim();
}

// Person form shared by login.html and contact.html: fills from the draft, validates, saves, moves on
function bindPersonForm(draftKey, nextPage) {
  const form = document.querySelector('form');
  const fields = ['firstName', 'lastName', 'phone'].map((id) => document.getElementById(id));
  const saved = loadDraft(draftKey);
  if (saved) fields.forEach((f) => { f.value = saved[f.id] || ''; });

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const person = Object.fromEntries(fields.map((f) => [f.id, f.value.trim()]));
    const missing = fields.filter((f) => f.required && !person[f.id]);
    fields.forEach((f) => f.classList.toggle('invalid', missing.includes(f)));
    if (missing.length) { missing[0].focus(); return; }
    saveDraft(draftKey, person);
    location.href = nextPage;
  });
}
