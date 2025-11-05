const panels = [...document.querySelectorAll(".panel")];
const tabs = [...document.querySelectorAll(".tab")];
tabs.forEach((t) =>
  t.addEventListener("click", () => {
    tabs.forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    const id = t.getAttribute("data-panel");
    panels.forEach((p) => p.classList.toggle("active", p.id === id));
  })
);

const out = document.getElementById("out");
const q = document.getElementById("q");
const view = document.getElementById("view");
const btn = document.getElementById("btn");
const selIndex = document.getElementById("sel-index");
const rebtn = document.getElementById("rebtn");
const hybtn = document.getElementById("hybtn");
const alpha = document.getElementById("alpha");
const benchbtn = document.getElementById("benchbtn");
const rerankbtn = document.getElementById("rerankbtn");
const selDs = document.getElementById("sel-ds");
const switchbtn = document.getElementById("switchbtn");
const fileInput = document.getElementById("file");
const uploadbtn = document.getElementById("uploadbtn");
const saveix = document.getElementById("saveix");
const loadix = document.getElementById("loadix");
const listix = document.getElementById("listix");
const expbtn = document.getElementById("expbtn");

const nlist = document.getElementById("nlist");
const nprobe = document.getElementById("nprobe");
const graphM = document.getElementById("graphM");

const qAns = document.getElementById("q-answer");
const ansBtn = document.getElementById("answer-btn");
const outAns = document.getElementById("out-answer");
const ansMode = document.getElementById("ans-mode");
const ansAlpha = document.getElementById("ans-alpha");
const ansPool = document.getElementById("ans-pool");
const sentK = document.getElementById("sent-k");

const idSelect = document.getElementById("id-select");
const textSelect = document.getElementById("text-select");
const dsName = document.getElementById("ds-name");
const lowerCb = document.getElementById("lower-cb");
const dedupCb = document.getElementById("dedup-cb");
const previewbtn = document.getElementById("previewbtn");
const importbtn = document.getElementById("importbtn");
const outDataset = document.getElementById("out-dataset");

const apikey = document.getElementById("apikey");
const setkey = document.getElementById("setkey");
const embSel = document.getElementById("emb-sel");
const setEmb = document.getElementById("set-emb");
const cacheInfo = document.getElementById("cache-info");
const cacheClear = document.getElementById("cache-clear");
const outAdmin = document.getElementById("out-admin");
const reembToggle = document.getElementById("reemb-toggle");
const reembNow = document.getElementById("reemb-now");

let APIKEY = "";
["naive", "ivf", "graph"].forEach((k) => {
  const o = document.createElement("option");
  o.value = k;
  o.textContent = k;
  selIndex.appendChild(o);
});
["sample", "uploaded"].forEach((k) => {
  const o = document.createElement("option");
  o.value = k;
  o.textContent = k;
  selDs.appendChild(o);
});

function headersJSON() {
  const h = { "Content-Type": "application/json" };
  if (APIKEY) h["X-Auth"] = APIKEY;
  return h;
}
function headersRaw() {
  const h = {};
  if (APIKEY) h["X-Auth"] = APIKEY;
  return h;
}
function clearOutput(el) {
  el.innerHTML = "";
}
function showJSON(el, x) {
  clearOutput(el);
  const pre = document.createElement("pre");
  pre.className = "codebox";
  pre.textContent = typeof x === "string" ? x : JSON.stringify(x, null, 2);
  el.appendChild(pre);
}
function showJSONMain(x) {
  showJSON(out, x);
}
function showCards(target, data) {
  clearOutput(target);
  const arr = data && data.results ? data.results : [];
  const grid = document.createElement("div");
  grid.className = "cards";
  arr.forEach((r) => {
    const it = document.createElement("div");
    it.className = "card-item";
    const t = document.createElement("div");
    t.className = "title";
    t.textContent = r.text || "(no text)";
    const badges = document.createElement("div");
    badges.className = "badges";
    const b1 = document.createElement("span");
    b1.className = "badge";
    b1.textContent = "id: " + (r.id ?? "-");
    const b2 = document.createElement("span");
    b2.className = "badge";
    b2.textContent =
      "score: " + (r.score != null ? Number(r.score).toFixed(4) : "-");
    const b3 = document.createElement("span");
    b3.className = "badge";
    b3.textContent =
      "rerank: " +
      (r.rerank_score != null ? Number(r.rerank_score).toFixed(4) : "-");
    badges.append(b1, b2, b3);
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent =
      (data.index || data.mode || "") + " • " + (data.dataset || "");
    it.append(t, badges, meta);
    grid.appendChild(it);
  });
  if (arr.length === 0) {
    showJSON(target, data);
    return;
  }
  target.appendChild(grid);
}
function mark(text, query) {
  const words = (query || "").toLowerCase().match(/\w+/g) || [];
  let t = text || "";
  words.forEach((w) => {
    const re = new RegExp(`\\b(${w})\\b`, "gi");
    t = t.replace(re, "<mark>$1</mark>");
  });
  return t;
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: headersJSON(),
    body: JSON.stringify(body),
  });
  return r.json();
}

async function search() {
  const text = q.value.trim();
  if (!text) {
    showJSONMain("Type something first.");
    return;
  }
  const data = await postJSON("/api/search", { text, k: 5 });
  view.value === "cards" ? showCards(out, data) : showJSONMain(data);
}
async function hybrid() {
  const text = q.value.trim();
  if (!text) {
    showJSONMain("Type something first.");
    return;
  }
  const r = await fetch(
    "/api/hybrid?alpha=" + encodeURIComponent(alpha.value || "0.5"),
    {
      method: "POST",
      headers: headersJSON(),
      body: JSON.stringify({ text, k: 5 }),
    }
  );
  const data = await r.json();
  view.value === "cards" ? showCards(out, data) : showJSONMain(data);
}
async function rerank(mode = "hybrid") {
  const text = q.value.trim();
  if (!text) {
    showJSONMain("Type something first.");
    return;
  }
  const params = new URLSearchParams({
    mode,
    alpha: String(alpha.value || "0.5"),
    pool: "25",
  });
  const r = await fetch("/api/rerank?" + params.toString(), {
    method: "POST",
    headers: headersJSON(),
    body: JSON.stringify({ text, k: 5 }),
  });
  const data = await r.json();
  view.value === "cards" ? showCards(out, data) : showJSONMain(data);
}
async function reindex() {
  const params = new URLSearchParams({
    kind: selIndex.value,
    nlist: String(nlist.value || "32"),
    nprobe: String(nprobe.value || "4"),
    M: String(graphM.value || "10"),
  });
  const r = await fetch("/api/reindex?" + params.toString(), {
    method: "POST",
    headers: headersRaw(),
  });
  showJSONMain(await r.json());
}
async function benchmark() {
  const r = await fetch(
    "/api/benchmark?k=5&alpha=" + encodeURIComponent(alpha.value || "0.5")
  );
  showJSONMain(await r.json());
}
async function switchDs() {
  const r = await fetch(
    "/api/switch_dataset?name=" +
      encodeURIComponent(selDs.value) +
      "&kind=" +
      encodeURIComponent(selIndex.value),
    { method: "POST" }
  );
  showJSONMain(await r.json());
}
async function listIx() {
  const r = await fetch("/api/list_indexes");
  showJSONMain(await r.json());
}
async function saveIx() {
  const r = await fetch("/api/save_index", {
    method: "POST",
    headers: headersRaw(),
  });
  showJSONMain(await r.json());
}
async function loadIx() {
  const r = await fetch(
    "/api/load_index?kind=" + encodeURIComponent(selIndex.value),
    { method: "POST", headers: headersRaw() }
  );
  showJSONMain(await r.json());
}
async function exportLast() {
  const r = await fetch("/api/export_last");
  const data = await r.json();
  if (!data.ok) {
    showJSONMain(data);
    return;
  }
  const blob = new Blob([data.content], {
    type: "text/markdown;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = data.filename || "export.md";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  showJSONMain({ ok: true, downloaded: a.download });
}

let uploadToken = null;
async function inspectUpload() {
  const f = fileInput.files[0];
  if (!f) {
    showJSON(outDataset, "Choose a file");
    return;
  }
  const form = new FormData();
  form.append("file", f);
  const r = await fetch("/api/inspect_file", { method: "POST", body: form });
  const data = await r.json();
  if (!data.ok) {
    showJSON(outDataset, data);
    return;
  }
  uploadToken = data.token;
  idSelect.innerHTML = "";
  textSelect.innerHTML = "";
  data.columns.forEach((c) => {
    const o1 = document.createElement("option");
    o1.value = c;
    o1.textContent = c;
    idSelect.appendChild(o1);
    const o2 = document.createElement("option");
    o2.value = c;
    o2.textContent = c;
    textSelect.appendChild(o2);
  });
  renderPreviewTable(data.sample);
}
function renderPreviewTable(rows) {
  clearOutput(outDataset);
  if (!rows || rows.length === 0) {
    showJSON(outDataset, { ok: true, sample: [] });
    return;
  }
  const keys = Object.keys(rows[0] || {});
  const table = document.createElement("table");
  table.className = "table";
  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  keys.forEach((k) => {
    const th = document.createElement("th");
    th.textContent = k;
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  const tbody = document.createElement("tbody");
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    keys.forEach((k) => {
      const td = document.createElement("td");
      td.textContent = String(r[k] ?? "");
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);
  outDataset.appendChild(table);
}
async function importMapped() {
  if (!uploadToken) {
    showJSON(outDataset, "Inspect a file first.");
    return;
  }
  const body = new FormData();
  body.append("token", uploadToken);
  body.append("id_col", idSelect.value);
  body.append("text_col", textSelect.value);
  body.append("name", dsName.value || "uploaded");
  body.append("lower_text", String(lowerCb.checked));
  body.append("dedup_text", String(dedupCb.checked));
  const r = await fetch("/api/import_uploaded", {
    method: "POST",
    headers: headersRaw(),
    body,
  });
  const data = await r.json();
  showJSON(outDataset, data);
  if (data.ok) {
    selDs.value = data.dataset;
    await switchDs();
  }
}

async function answer() {
  const text = qAns.value.trim();
  if (!text) {
    showJSON(outAns, "Type a question first.");
    return;
  }
  const params = new URLSearchParams({
    mode: ansMode.value,
    alpha: String(ansAlpha.value || "0.3"),
    pool: String(ansPool.value || "25"),
    sent_k: String(sentK.value || "3"),
  });
  const r = await fetch("/api/answer?" + params.toString(), {
    method: "POST",
    headers: headersJSON(),
    body: JSON.stringify({ text, k: 5 }),
  });
  const data = await r.json();
  clearOutput(outAns);
  const wrap = document.createElement("div");
  wrap.className = "cards";
  const ans = document.createElement("div");
  ans.className = "card-item";
  ans.innerHTML =
    '<div class="title">Answer</div><div>' +
    (data.result && data.result.answer
      ? mark(data.result.answer, text)
      : "(no answer)") +
    "</div>";
  const src = document.createElement("div");
  src.className = "card-item";
  src.innerHTML = '<div class="title">Citations</div>';
  const ul = document.createElement("ul");
  const cits = (data.result && data.result.citations) || [];
  cits.forEach((c) => {
    const li = document.createElement("li");
    li.className = "meta";
    li.innerHTML = `<strong>${c.id}</strong>: ${mark(c.snippet || "", text)}`;
    ul.appendChild(li);
  });
  const copyBtn = document.createElement("button");
  copyBtn.className = "btn";
  copyBtn.textContent = "Copy Citations";
  copyBtn.addEventListener("click", () => {
    const lines = cits.map((c) => `- ${c.id}: ${c.snippet || ""}`).join("\n");
    navigator.clipboard
      .writeText(lines)
      .then(() => showJSON(outAns, { ok: true, copied: cits.length }));
  });
  src.appendChild(ul);
  src.appendChild(copyBtn);
  wrap.append(ans, src);
  outAns.appendChild(wrap);
}

btn.addEventListener("click", search);
q.addEventListener("keydown", (e) => {
  if (e.key === "Enter") search();
});
hybtn.addEventListener("click", hybrid);
rerankbtn.addEventListener("click", () => rerank("hybrid"));
rebtn.addEventListener("click", reindex);
benchbtn.addEventListener("click", benchmark);
switchbtn.addEventListener("click", switchDs);
listix.addEventListener("click", listIx);
saveix.addEventListener("click", saveIx);
loadix.addEventListener("click", loadIx);
expbtn.addEventListener("click", exportLast);

uploadbtn.addEventListener("click", inspectUpload);
previewbtn.addEventListener("click", inspectUpload);
importbtn.addEventListener("click", importMapped);

ansBtn.addEventListener("click", answer);

setkey.addEventListener("click", () => {
  APIKEY = apikey.value.trim();
  showJSON(outAdmin, { ok: true, key_set: !!APIKEY });
});
setEmb.addEventListener("click", async () => {
  const re = reembToggle && reembToggle.checked ? "true" : "false";
  const url =
    "/api/set_embedder?name=" +
    encodeURIComponent(embSel.value) +
    "&reembed=" +
    re +
    "&kind=" +
    encodeURIComponent(selIndex.value);
  const r = await fetch(url, { method: "POST", headers: headersRaw() });
  showJSON(outAdmin, await r.json());
});

reembNow.addEventListener("click", async () => {
  const url = "/api/reembed?kind=" + encodeURIComponent(selIndex.value);
  const r = await fetch(url, { method: "POST", headers: headersRaw() });
  showJSON(outAdmin, await r.json());
});

cacheInfo.addEventListener("click", async () => {
  const r = await fetch("/api/cache_info");
  showJSON(outAdmin, await r.json());
});
cacheClear.addEventListener("click", async () => {
  const r = await fetch("/api/cache_clear", {
    method: "POST",
    headers: headersRaw(),
  });
  showJSON(outAdmin, await r.json());
});
