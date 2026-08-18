"use strict";

const N = 11;
const THRONE = 60;
const CORNERS = [0, N - 1, N * N - N, N * N - 1];

let meta = null;
let state = null;
let sel = null;          // selected square (int) or null
let gen = 0;             // bumped on new game / undo to cancel stale AI timers
let aiPending = false;

const $ = (id) => document.getElementById(id);
const boardEl = $("board");
const squares = [];

// --- helpers ---------------------------------------------------------------

function sqName(sq) {
  const r = Math.floor(sq / N), c = sq % N;
  return "abcdefghijk"[c] + (N - r);
}

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (e) { /* keep */ }
    throw new Error(detail);
  }
  return r.json();
}

function setError(msg) { $("error").textContent = msg || ""; }

// --- board construction ----------------------------------------------------

function buildBoard() {
  for (let sq = 0; sq < N * N; sq++) {
    const d = document.createElement("div");
    const r = Math.floor(sq / N), c = sq % N;
    d.className = "sq" + ((r + c) % 2 ? " alt" : "");
    if (sq === THRONE) d.classList.add("throne");
    if (CORNERS.includes(sq)) d.classList.add("corner");
    d.dataset.sq = sq;
    d.title = sqName(sq);
    d.addEventListener("click", () => onSquareClick(sq));
    boardEl.appendChild(d);
    squares.push(d);
  }
  for (let r = 0; r < N; r++) {
    const d = document.createElement("div");
    d.textContent = N - r;
    $("rank-labels").appendChild(d);
  }
  for (let c = 0; c < N; c++) {
    const d = document.createElement("div");
    d.textContent = "abcdefghijk"[c];
    $("file-labels").appendChild(d);
  }
}

// --- rendering -------------------------------------------------------------

function agentLabel(id) {
  const o = meta.agents.find((a) => a.id === id);
  return o ? o.label : id;
}

function humanToMove() {
  return state && !state.result && state.human[state.to_move];
}

function render() {
  const b = state.board;
  const lm = state.last_move;
  for (let sq = 0; sq < N * N; sq++) {
    const d = squares[sq];
    d.classList.remove("sel", "dest", "occupied", "last-from", "last-to",
                       "captured", "mine");
    const p = d.querySelector(".piece");
    if (p) p.remove();
    const ch = b[sq];
    if (ch !== ".") {
      const piece = document.createElement("div");
      piece.className = "piece " +
        (ch === "A" ? "att" : ch === "K" ? "king" : "def");
      d.appendChild(piece);
    }
  }
  if (lm) {
    squares[lm.from].classList.add("last-from");
    squares[lm.to].classList.add("last-to");
    for (const sq of lm.captured) squares[sq].classList.add("captured");
  }
  if (humanToMove()) {
    for (const frm of Object.keys(state.legal)) {
      squares[+frm].classList.add("mine");
    }
    if (sel !== null && state.legal[sel]) {
      squares[sel].classList.add("sel");
      for (const to of state.legal[sel]) {
        squares[to].classList.add("dest");
        if (state.board[to] !== ".") squares[to].classList.add("occupied");
      }
    }
  }
  renderPanel();
}

function renderPanel() {
  const st = $("status");
  st.classList.toggle("over", !!state.result);
  if (state.result) {
    const w = state.result.winner;
    st.textContent = (w ? (w === "attacker" ? "Attackers win" : "Defenders win")
                        : "Draw") + " — " + state.result.reason;
  } else {
    const who = state.to_move === "attacker" ? "Attackers" : "Defenders";
    const label = state.human[state.to_move]
      ? "your move" : agentLabel(state.agents[state.to_move]) + " thinking…";
    st.textContent = `${who} to move (${label}) — ply ${state.ply}/${state.move_limit}`;
  }
  let att = 0, def = 0;
  for (const ch of state.board) {
    if (ch === "A") att++;
    else if (ch === "D" || ch === "K") def++;
  }
  $("counts").textContent =
    `attackers ${att} · defenders ${def} (incl. king)`;
  $("undo").disabled = !state || state.ply === 0;
  $("ai-step").disabled = !state || !!state.result
    || state.human[state.to_move];

  const log = $("move-log");
  log.innerHTML = "";
  state.moves.forEach((m, i) => {
    const li = document.createElement("li");
    const cap = m.captured.length
      ? ` ×${m.captured.length} (${m.captured.map(sqName).join(", ")})` : "";
    li.innerHTML = `<span class="n">${i + 1}.</span>` +
      `<span class="who">${m.side === "attacker" ? "●" : "○"}</span>` +
      `<span>${sqName(m.from)}–${sqName(m.to)}` +
      `<span class="cap">${cap}</span></span>`;
    log.appendChild(li);
  });
  log.scrollTop = log.scrollHeight;
}

// --- interaction -----------------------------------------------------------

function onSquareClick(sq) {
  if (!humanToMove()) return;
  setError("");
  if (sel !== null && state.legal[sel] && state.legal[sel].includes(sq)) {
    const frm = sel;
    sel = null;
    move(frm, sq);
    return;
  }
  sel = (state.legal[sq] && sq !== sel) ? sq : null;
  render();
}

async function move(frm, to) {
  try {
    state = await api(`/api/game/${state.game_id}/move`, { frm, to });
    render();
    scheduleAI();
  } catch (e) {
    setError(e.message);
    render();
  }
}

function scheduleAI() {
  if (!state || state.result || state.human[state.to_move]) return;
  if (!$("auto-ai").checked || aiPending) return;
  const g = gen;
  aiPending = true;
  setTimeout(async () => {
    if (g !== gen) { aiPending = false; return; }
    try {
      const s = await api(`/api/game/${state.game_id}/ai_move`, {});
      if (g !== gen) { aiPending = false; return; }
      state = s;
      render();
    } catch (e) {
      if (g === gen) setError(e.message);
    }
    aiPending = false;
    if (g === gen) scheduleAI();
  }, 220);
}

async function newGame() {
  setError("");
  gen++;
  aiPending = false;
  sel = null;
  try {
    state = await api("/api/game", {
      attacker: $("att-select").value,
      defender: $("def-select").value,
    });
    render();
    scheduleAI();
  } catch (e) {
    setError(e.message);
  }
}

async function undo() {
  if (!state) return;
  setError("");
  gen++;                 // cancel any queued AI move
  aiPending = false;
  sel = null;
  try {
    state = await api(`/api/game/${state.game_id}/undo`, {});
    render();
    scheduleAI();
  } catch (e) {
    setError(e.message);
  }
}

// --- init ------------------------------------------------------------------

async function init() {
  buildBoard();
  meta = await api("/api/meta");
  for (const selEl of [$("att-select"), $("def-select")]) {
    for (const a of meta.agents) {
      const o = document.createElement("option");
      o.value = a.id;
      o.textContent = a.label;
      if (a.detail) {
        const side = selEl.id === "att-select" ? "attacker" : "defender";
        o.title = a.detail[side];
      }
      selEl.appendChild(o);
    }
  }
  $("att-select").value = "random";   // rule-checking default: you vs random
  $("def-select").value = "human";
  $("new-game").addEventListener("click", newGame);
  $("undo").addEventListener("click", undo);
  $("ai-step").addEventListener("click", () => {
    if (!aiPending) scheduleAIStep();
  });
  await newGame();
}

async function scheduleAIStep() {
  if (!state || state.result || state.human[state.to_move]) return;
  try {
    state = await api(`/api/game/${state.game_id}/ai_move`, {});
    render();
  } catch (e) {
    setError(e.message);
  }
}

init().catch((e) => setError(e.message));
