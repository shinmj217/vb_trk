// vb_trk 뷰어: 랠리 복기(영상 + 겹침 + 코트 지도) / 경기 통계. 서버 API(app.py)만 쓴다.
const $ = (s) => document.querySelector(s);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const TEAM_COLOR = () => [css("--left"), css("--right")];          // 0 = LEFT, 1 = RIGHT
const ACTION = { pass: "리시브", set: "토스", spike: "스파이크" };
const TAIL = 12;                                                    // 공 꼬리 (직전 프레임 수)
const MAX_SPAN = 6000;                                              // 서버 /frames 상한

const v = $("#v"), ov = $("#ov"), court = $("#court");
let matches = [], M = null, rallies = [], cur = null, curIdx = -1;
let players = new Map(), balls = new Map(), gaps = [];              // 현재 랠리 구간 데이터

const mmss = (f) => { const s = Math.round(f / M.fps); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
const teamSpan = (t) => (t ? `<span class="${t === "LEFT" ? "L" : "R"}">${t}</span>` : "?");
const getJSON = async (url) => { const r = await fetch(url); if (!r.ok) throw new Error(`${url} → ${r.status}`); return r.json(); };

function showMsg(text) { const m = $("#msg"); m.hidden = !text; m.textContent = text || ""; }

async function init() {
  matches = await getJSON("/api/matches");
  const sel = $("#match");
  sel.innerHTML = matches.map((m) => `<option value="${m.stem}">${m.stem}</option>`).join("");
  sel.onchange = () => selectMatch(sel.value);
  document.querySelectorAll("nav button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
  document.addEventListener("keydown", onKey);
  new ResizeObserver(fitOverlay).observe(v);
  if ("requestVideoFrameCallback" in v) v.requestVideoFrameCallback(onFrame);
  else requestAnimationFrame(rafLoop);
  if (!matches.length) return showMsg("경기가 없습니다. outputs/ 에 <이름>_rallies.csv 가 있어야 합니다.");
  selectMatch(matches[0].stem);
}

function showTab(tab) {
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  $("#replay").hidden = tab !== "replay";
  $("#stats").hidden = tab !== "stats";
}

async function selectMatch(stem) {
  M = matches.find((m) => m.stem === stem);
  v.pause();
  if (!M.has_video) { v.removeAttribute("src"); showMsg("이 경기의 영상(data/" + stem + ".mp4)이 없습니다. 결과 표와 통계만 볼 수 있습니다."); }
  else if (!M.playable) { v.removeAttribute("src"); showMsg("이 영상 형식은 브라우저에서 재생할 수 없습니다 (VP9·H.264 필요)."); }
  else { showMsg(""); v.src = `/video/${stem}`; }
  rallies = await getJSON(`/api/matches/${stem}/rallies`);
  renderList();
  renderStats(stem, await getJSON(`/api/matches/${stem}/summary`));
  if (rallies.length) selectRally(0);
}

function renderList() {
  $("#rallies tbody").innerHTML = rallies.map((r, i) => `
    <tr data-i="${i}"><td>${r.n}</td><td>${mmss(r.start)}</td><td>${teamSpan(r.serve_team)} → ${teamSpan(r.winner)}</td>
    <td>${r.contacts}</td><td>${r.seconds.toFixed(0)}초</td>
    <td class="warn" title="${r.warn.join(", ")}">${r.warn.length ? "⚠" : ""}</td></tr>`).join("");
  document.querySelectorAll("#rallies tbody tr").forEach((tr) => (tr.onclick = () => selectRally(+tr.dataset.i)));
}

async function selectRally(i) {
  if (i < 0 || i >= rallies.length) return;
  curIdx = i; cur = rallies[i];
  document.querySelectorAll("#rallies tbody tr").forEach((tr) => tr.classList.toggle("sel", +tr.dataset.i === i));
  document.querySelector(`#rallies tbody tr[data-i="${i}"]`)?.scrollIntoView({ block: "nearest" });
  const [a, b0] = cur.window, b = Math.min(b0, a + MAX_SPAN);
  const d = await getJSON(`/api/matches/${M.stem}/frames?start=${a}&end=${b}`);
  if (cur !== rallies[i]) return;                                   // 그 사이 다른 랠리를 눌렀으면 버린다
  players = new Map(); balls = new Map(); gaps = d.gaps;
  for (const p of d.players) { if (!players.has(p[0])) players.set(p[0], []); players.get(p[0]).push(p); }
  for (const bl of d.ball) balls.set(bl[0], bl);
  renderTimeline(); renderInfo();
  if (M.has_video && M.playable) { v.currentTime = a / M.fps; v.play().catch(() => {}); }
  draw(a);
}

function renderTimeline() {
  const [a, b] = cur.window, pct = (f) => (100 * (f - a)) / (b - a);
  const seg = (cls, f0, f1, title = "") => `<div class="seg ${cls}" style="left:${pct(f0)}%;width:${Math.max(0.3, pct(f1) - pct(f0))}%" title="${title}"></div>`;
  $("#timeline").innerHTML = seg("rally", cur.start, Math.min(cur.end, b), "랠리 구간") +
    gaps.map((g) => seg("gap", g[0], g[1], "코트 인식 실패 (클로즈업·리플레이 등)")).join("") +
    cur.events.map((e) => `<div class="ev" style="left:${pct(e.start)}%" title="${ACTION[e.action] || e.action} #${e.actor ?? "?"} (${e.team})"></div>`).join("") +
    `<div class="head" id="head"></div>`;
}

function renderInfo() {
  const warn = cur.warn.length ? `<p class="warn">⚠ ${cur.warn.join(" · ")}</p>` : "";
  $("#info").innerHTML = `<h4>랠리 ${cur.n} / ${rallies.length}</h4>
    <p>서브 ${teamSpan(cur.serve_team)} → 득점 ${teamSpan(cur.winner)}</p>
    <p>접촉 ${cur.contacts}회 · ${cur.seconds.toFixed(1)}초 · ${mmss(cur.start)}</p>${warn}
    <p>지도에 표시한 접촉은 코트 안 좌표만 (코트 밖 좌표는 코트 맞춤 오류라 제외)</p>
    <p>${gaps.length ? `코트 인식 실패 구간 ${gaps.length}곳 (타임라인 회색)` : "구간 전체에서 코트 인식됨"}</p>
    <p id="now"></p>`;
}

function renderStats(stem, s) {
  const pct = (r, n, d) => (r === null ? "-" : `${(100 * r).toFixed(1)}% (${n}/${d})`);
  $("#cards").innerHTML = ["LEFT", "RIGHT"].map((t) => `<div class="card"><h3>${teamSpan(t)}</h3>
    <div class="row"><span>득점</span><b>${s[t].points}</b></div>
    <div class="row"><span>서브 득점률</span><b>${pct(s[t].serve_rate, s[t].serve_wins, s[t].serves)}</b></div>
    <div class="row"><span>사이드아웃률</span><b>${pct(s[t].sideout_rate, s[t].receive_wins, s[t].receives)}</b></div></div>`).join("") +
    `<div class="card"><h3>전체</h3><div class="row"><span>랠리</span><b>${s.rallies}</b></div>
     <p class="warn">득점 팀은 추정값 (판정 약 85%)이라 ±2~3 오차가 있습니다.</p></div>`;
  const img = $("#figure"); img.hidden = false; img.onerror = () => (img.hidden = true); img.src = `/api/matches/${stem}/figure`;
}

// ---- 겹쳐 그리기 ----
function fitOverlay() { ov.width = v.clientWidth; ov.height = v.clientHeight; if (cur) draw(frameNow()); }
const frameNow = () => Math.round(v.currentTime * M.fps);

function onFrame(now, meta) {
  if (M && cur) {
    const f = Math.round(meta.mediaTime * M.fps);
    draw(f);
    if (f >= cur.window[1]) v.pause();
  }
  v.requestVideoFrameCallback(onFrame);
}
function rafLoop() { if (M && cur) draw(frameNow()); requestAnimationFrame(rafLoop); }  // requestVideoFrameCallback 이 없는 브라우저용

function draw(f) {
  const ctx = ov.getContext("2d"), colors = TEAM_COLOR();
  ctx.clearRect(0, 0, ov.width, ov.height);
  if (!M?.width) return;
  const k = ov.width / M.width;
  const ps = players.get(f);
  if (ps) {
    ctx.lineWidth = 2; ctx.font = "12px sans-serif";
    for (const p of ps) {
      ctx.strokeStyle = ctx.fillStyle = colors[p[2]] || "#888";
      ctx.strokeRect(p[3] * k, p[4] * k, (p[5] - p[3]) * k, (p[6] - p[4]) * k);
      ctx.fillText(`#${p[1]}${p[9] ? " ↑" : ""}`, p[3] * k, p[4] * k - 3);
    }
  } else if (gaps.some((g) => f >= g[0] && f <= g[1])) {
    ctx.fillStyle = "rgba(0,0,0,.55)"; ctx.fillRect(8, 8, 190, 24);
    ctx.fillStyle = "#fff"; ctx.font = "13px sans-serif"; ctx.fillText("코트 인식 실패 구간", 14, 25);
  }
  const tail = [];
  for (let g = f - TAIL; g <= f; g++) if (balls.has(g)) tail.push(balls.get(g));
  ctx.strokeStyle = "#ffd400"; ctx.lineWidth = 2; ctx.beginPath();
  tail.forEach((b, i) => (i ? ctx.lineTo(b[1] * k, b[2] * k) : ctx.moveTo(b[1] * k, b[2] * k)));
  ctx.stroke();
  if (balls.has(f)) { ctx.beginPath(); ctx.arc(balls.get(f)[1] * k, balls.get(f)[2] * k, 12, 0, 7); ctx.stroke(); }
  const ev = cur?.events.find((e) => f >= e.start && f <= e.end);
  if (ev) {
    ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(8, ov.height - 34, 260, 26);
    ctx.fillStyle = "#fff"; ctx.font = "14px sans-serif"; ctx.fillText(`${ACTION[ev.action] || ev.action} #${ev.actor ?? "?"} (${ev.team})`, 14, ov.height - 16);
  }
  drawCourt(f, ps || []); movePlayhead(f);
}

function movePlayhead(f) {
  const h = $("#head"); if (!h || !cur) return;
  const [a, b] = cur.window; h.style.left = `${Math.min(100, Math.max(0, (100 * (f - a)) / (b - a)))}%`;
  const n = $("#now"); if (n) n.textContent = `프레임 ${f} · ${mmss(f)}`;
}

// 코트 지도: 가로 = court_y (0~18 m, 작은 쪽이 LEFT), 세로 = court_x (0~9 m). 접촉은 시간 순서로 번호를 붙여 잇는다
function drawCourt(f, ps) {
  const c = court.getContext("2d"), colors = TEAM_COLOR(), W = court.width, H = court.height;
  const X0 = -3, X1 = 21, Y0 = -1.5, Y1 = 10.5;                      // 표시 범위 (m), 코트 밖 여유 포함
  const px = (cy) => ((cy - X0) / (X1 - X0)) * W, py = (cx) => ((cx - Y0) / (Y1 - Y0)) * H;
  c.clearRect(0, 0, W, H);
  c.fillStyle = "#f2e4c9"; c.fillRect(px(0), py(0), px(18) - px(0), py(9) - py(0));
  c.strokeStyle = "#fff"; c.lineWidth = 2; c.strokeRect(px(0), py(0), px(18) - px(0), py(9) - py(0));
  c.beginPath(); c.moveTo(px(6), py(0)); c.lineTo(px(6), py(9)); c.moveTo(px(12), py(0)); c.lineTo(px(12), py(9)); c.stroke();
  c.strokeStyle = "#1d2330"; c.lineWidth = 3; c.beginPath(); c.moveTo(px(9), py(-0.5)); c.lineTo(px(9), py(9.5)); c.stroke();
  c.font = "12px sans-serif"; c.fillStyle = colors[0]; c.fillText("LEFT", px(0), 12); c.fillStyle = colors[1]; c.fillText("RIGHT", px(18) - 38, 12);
  for (const p of ps) { c.fillStyle = colors[p[2]] || "#888"; c.beginPath(); c.arc(px(p[8]), py(p[7]), 5, 0, 7); c.fill(); }
  // 코트 밖 멀리 찍힌 접촉(코트 맞춤이 틀린 프레임)은 지도에서 뺀다
  const inMap = (e) => e.court_x !== null && e.court_y !== null && e.court_y > X0 && e.court_y < X1 && e.court_x > Y0 && e.court_x < Y1;
  const pts = (cur?.events || []).filter(inMap);
  c.strokeStyle = "#555"; c.lineWidth = 1.5; c.beginPath();
  pts.forEach((e, i) => (i ? c.lineTo(px(e.court_y), py(e.court_x)) : c.moveTo(px(e.court_y), py(e.court_x))));
  c.stroke();
  pts.forEach((e, i) => {
    const done = e.start <= f; c.beginPath(); c.arc(px(e.court_y), py(e.court_x), 9, 0, 7);
    c.fillStyle = done ? "#1d2330" : "#fff"; c.fill(); c.strokeStyle = "#1d2330"; c.stroke();
    c.fillStyle = done ? "#fff" : "#1d2330"; c.font = "11px sans-serif"; c.textAlign = "center"; c.fillText(i + 1, px(e.court_y), py(e.court_x) + 4); c.textAlign = "start";
  });
}

function onKey(e) {
  if (e.target.tagName === "SELECT" || $("#replay").hidden) return;
  if (e.key === "ArrowDown") { e.preventDefault(); selectRally(curIdx + 1); }
  if (e.key === "ArrowUp") { e.preventDefault(); selectRally(curIdx - 1); }
}

init().catch((e) => showMsg("불러오기 실패: " + e.message));
