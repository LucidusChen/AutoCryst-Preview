"""A self-contained 3-D structure viewer.

The viewer is one HTML document placed in a sandboxed ``<iframe srcdoc>``: a small canvas renderer
written here, no external script, font or image. Its Content-Security-Policy forbids every network
fetch (``default-src 'none'``), and the sandbox withholds same-origin access, so the structure data
cannot leave the page and the page cannot reach the application.

What it draws is exactly what the core's viewer payload describes, and it says so on screen:

* the asymmetric unit as written in the file, without symmetry expansion;
* bonds only within one cell translation -- bonds across a cell boundary are listed, not drawn;
* elements coloured by convention, unknown sites in a distinct dashed style, Q peaks as rings;
* occupancy and displacement as their own numbers on click; **no confidence colouring**, because
  no calibrated confidence exists -- every atom is drawn opaque.

Every string that came from a file (labels, element symbols) is rendered with ``fillText`` or
``textContent``, never as HTML.
"""

from __future__ import annotations

import html
import json
from typing import Any

from crystalx_core.geometry.model import frac_to_cart, orthogonalisation_matrix

__all__ = ["viewer_document", "viewer_html", "empty_viewer", "EMPTY_VIEWER", "VIEWER_TEXT"]

#: Viewer strings per interface language. Placeholders in braces are filled in the page.
VIEWER_TEXT: dict[str, dict[str, str]] = {
    "zh": {
        "html_lang": "zh-CN", "title": "晶体结构三维视图", "empty": "暂无可显示的模型。",
        "controls": "视图控制", "bl": "标签", "bq": "Q 峰", "bc": "晶胞", "br": "重置",
        "mock": "MOCK DEMO · 非真实结果", "unknown": "未知", "aniso": "（各向异性）",
        "q_info": "差值电子密度峰（不是原子）· 高度 {h} e/Å³ · 第 {rank} 高 · 分数坐标 ({xyz})",
        "unknown_site": "未知位点：SFAC 符号 “{el}” 不是元素",
        "element": "元素 {el}（程序按峰高指定，未经独立验证）",
        "frac": "分数坐标 ({xyz})", "occ": "占有率", "fixed": "（固定）",
        "help": "点击原子查看数值；拖动或方向键旋转，滚轮或 +/− 缩放。",
        "aria": "晶体结构三维视图：{atoms} 个原子，{q} 个 Q 峰，{unk} 个未知位点。下方表格列出全部位点。",
        "legend_q": "Q 峰（差值电子密度，不是原子）", "legend_unknown": "未知位点（符号不是元素）",
        "confidence": "置信度：不可用。原子一律不透明，不以颜色或透明度表示可信度。",
        "display": "显示：不对称单元（文件中的位点，未做对称扩展）。", "source": "来源：{s}。",
        "bonds": "化学键：同一晶胞平移内绘制 {n} 条。", "boundary": "跨晶胞边界、未绘制的键 {n} 条：",
        "translation": "（平移 {t}）", "more": "……其余 {n} 条见结果记录",
    },
    "ja": {
        "html_lang": "ja", "title": "結晶構造の 3D 表示", "empty": "表示できるモデルはまだありません。",
        "controls": "表示の操作", "bl": "ラベル", "bq": "Q ピーク", "bc": "単位格子", "br": "リセット",
        "mock": "MOCK DEMO · 実際の結果ではありません", "unknown": "不明", "aniso": "（異方性）",
        "q_info": "差電子密度ピーク（原子ではない）· 高さ {h} e/Å³ · {rank} 番目 · 分率座標 ({xyz})",
        "unknown_site": "未知サイト：SFAC 記号 “{el}” は元素ではない",
        "element": "元素 {el}（プログラムがピーク高から割り当て、独立には未検証）",
        "frac": "分率座標 ({xyz})", "occ": "占有率", "fixed": "（固定）",
        "help": "原子をクリックすると値を表示。ドラッグまたは矢印キーで回転、ホイールまたは +/− で拡大縮小。",
        "aria": "結晶構造の 3D 表示：原子 {atoms} 個、Q ピーク {q} 個、未知サイト {unk} 個。下の表にすべてのサイトがあります。",
        "legend_q": "Q ピーク（差電子密度、原子ではない）", "legend_unknown": "未知サイト（記号が元素ではない）",
        "confidence": "信頼度：利用不可。原子はすべて不透明で、色や透明度で信頼度を表しません。",
        "display": "表示：非対称単位（ファイル中のサイト、対称展開なし）。", "source": "出典：{s}。",
        "bonds": "結合：同じ格子並進内で {n} 本を描画。", "boundary": "格子境界をまたぐため描画しない結合 {n} 本：",
        "translation": "（並進 {t}）", "more": "……残り {n} 本は結果記録を参照",
    },
    "en": {
        "html_lang": "en", "title": "3-D structure view", "empty": "No model to display yet.",
        "controls": "View controls", "bl": "Labels", "bq": "Q peaks", "bc": "Cell", "br": "Reset",
        "mock": "MOCK DEMO · not a real result", "unknown": "unknown", "aniso": " (anisotropic)",
        "q_info": "difference-density peak (not an atom) · height {h} e/Å³ · rank {rank} · fractional ({xyz})",
        "unknown_site": "unknown site: SFAC symbol “{el}” is not an element",
        "element": "element {el} (assigned by the program from peak height; not independently verified)",
        "frac": "fractional ({xyz})", "occ": "occupancy", "fixed": " (fixed)",
        "help": "Click an atom for its values; drag or use the arrow keys to rotate, wheel or +/− to zoom.",
        "aria": "3-D structure view: {atoms} atoms, {q} Q peaks, {unk} unknown sites. The table below lists every site.",
        "legend_q": "Q peak (difference density, not an atom)", "legend_unknown": "unknown site (symbol is not an element)",
        "confidence": "Confidence: unavailable. Every atom is drawn opaque; no colour or transparency stands for confidence.",
        "display": "Shown: the asymmetric unit (sites as written; no symmetry expansion).", "source": "Source: {s}.",
        "bonds": "Bonds drawn within one cell translation: {n}.", "boundary": "Bonds across a cell boundary, not drawn: {n}:",
        "translation": "(translation {t})", "more": "… {n} more in the run record",
    },
}


def empty_viewer(lang: str = "zh") -> str:
    text = VIEWER_TEXT.get(lang, VIEWER_TEXT["zh"])["empty"]
    return (
        '<div style="padding:48px;text-align:center;border:1px dashed #c9c9cf;border-radius:10px;'
        f'color:#6e6e73">{html.escape(text)}</div>'
    )


EMPTY_VIEWER = empty_viewer("zh")

_TEMPLATE = r"""<!doctype html>
<html lang="__LANG__"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'">
<style>
:root{color-scheme:light dark}
html,body{margin:0;height:100%;font:13px system-ui,-apple-system,"PingFang SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif;background:#fbfbfd;color:#1d1d1f}
#wrap{position:relative;height:100%;overflow:hidden}
canvas{width:100%;height:100%;display:block;cursor:grab;touch-action:none;outline:none}
canvas:focus-visible{box-shadow:inset 0 0 0 2px #0071e3}
.panel{position:absolute;background:rgba(255,255,255,.93);border:1px solid #d7d7dc;border-radius:8px;padding:6px 8px;font-size:12px;line-height:1.5}
#controls{top:8px;left:8px}#legend{top:8px;right:8px;max-width:230px}
#info{bottom:8px;left:8px;max-width:58%}#note{bottom:8px;right:8px;max-width:40%}
button{font:inherit;margin:0 2px;padding:2px 9px;border-radius:6px;border:1px solid #b9b9c0;background:transparent;color:inherit;cursor:pointer}
button[aria-pressed=false]{opacity:.5}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:-1px;border:1px solid rgba(0,0,0,.35)}
.ring{background:transparent;border:2px solid #D946EF}.unk{background:#FF1493;border:1px dashed #000}
#mock{position:absolute;top:42%;left:0;right:0;text-align:center;font-size:40px;font-weight:700;letter-spacing:.08em;color:rgba(217,70,239,.18);pointer-events:none}
@media (prefers-color-scheme:dark){html,body{background:#16171a;color:#e8e8ea}.panel{background:rgba(30,31,35,.93);border-color:#3a3b40}}
</style></head><body>
<div id="wrap">
<canvas id="c" tabindex="0"></canvas>
<div id="controls" class="panel" role="toolbar">
<button id="bl" aria-pressed="true"></button><button id="bq" aria-pressed="true"></button><button id="bc" aria-pressed="true"></button><button id="br"></button>
</div>
<div id="legend" class="panel"></div>
<div id="info" class="panel" aria-live="polite"></div>
<div id="note" class="panel"></div>
<div id="mock" hidden></div>
</div>
<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
(function () {
  "use strict";
  var P = JSON.parse(document.getElementById("payload").textContent);
  var T = P.text || {};
  function f(s, o) { return String(s).replace(/\{(\w+)\}/g, function (m, k) { return o[k] == null ? m : o[k]; }); }
  document.getElementById("controls").setAttribute("aria-label", T.controls);
  ["bl", "bq", "bc", "br"].forEach(function (id) { document.getElementById(id).textContent = T[id]; });
  document.getElementById("mock").textContent = T.mock;
  var atoms = P.atoms || [], bonds = P.bonds || [], edges = P.cell_edges || [], axes = P.axes || {};
  var show = { labels: true, q: true, cell: true };
  var canvas = document.getElementById("c"), ctx = canvas.getContext("2d");
  var info = document.getElementById("info"), legend = document.getElementById("legend"), note = document.getElementById("note");
  var real = atoms.filter(function (a) { return a.kind !== "q_peak"; });
  var corners = [].concat.apply([], edges);
  var cx = 0, cy = 0, cz = 0, extent = 2;
  function fit() {
    // Centre on the bounding box of everything drawn, and size the view to its farthest point.
    var pts = atoms.map(function (a) { return a.cart; }).concat(show.cell ? corners : []);
    if (!pts.length) { cx = cy = cz = 0; extent = 2; return; }
    var lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    pts.forEach(function (p) { for (var k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); } });
    cx = (lo[0] + hi[0]) / 2; cy = (lo[1] + hi[1]) / 2; cz = (lo[2] + hi[2]) / 2;
    extent = 2;
    pts.forEach(function (p) { var dx = p[0] - cx, dy = p[1] - cy, dz = p[2] - cz; extent = Math.max(extent, Math.sqrt(dx * dx + dy * dy + dz * dz)); });
  }
  var rotX = -0.45, rotY = 0.65, zoom = 1, W = 0, H = 0, scale = 1, selected = -1, hits = [];

  function project(p) {
    var x = p[0] - cx, y = p[1] - cy, z = p[2] - cz;
    var cY = Math.cos(rotY), sY = Math.sin(rotY), cX = Math.cos(rotX), sX = Math.sin(rotX);
    var x1 = x * cY + z * sY, z1 = -x * sY + z * cY;
    var y2 = y * cX - z1 * sX, z2 = y * sX + z1 * cX;
    return [W / 2 + x1 * scale, H / 2 - y2 * scale, z2];
  }
  function tint(hex, f) {
    var v = [1, 3, 5].map(function (i) { return parseInt(hex.slice(i, i + 2), 16); });
    return "rgb(" + v.map(function (c) { return Math.max(0, Math.min(255, Math.round(f >= 0 ? c + (255 - c) * f : c * (1 + f)))); }).join(",") + ")";
  }
  function textColour() { return getComputedStyle(document.body).color; }
  function resize() {
    var dpr = window.devicePixelRatio || 1;
    W = canvas.clientWidth; H = canvas.clientHeight;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }
  function draw() {
    scale = Math.min(W, H) / (2.1 * extent) * zoom;
    ctx.clearRect(0, 0, W, H);
    var fg = textColour(), items = [];
    if (show.cell) {
      edges.forEach(function (e) { items.push({ z: -1e9, kind: "edge", a: project(e[0]), b: project(e[1]) }); });
    }
    bonds.forEach(function (b) {
      var A = atoms[b[0]], B = atoms[b[1]], a = project(A.cart), c = project(B.cart);
      items.push({ z: (a[2] + c[2]) / 2 - 0.01, kind: "bond", a: a, b: c, A: A, B: B });
    });
    atoms.forEach(function (A, i) {
      if (A.kind === "q_peak" && !show.q) return;
      items.push({ z: project(A.cart)[2], kind: "atom", p: project(A.cart), A: A, i: i });
    });
    items.sort(function (u, v) { return u.z - v.z; });
    hits = [];
    ctx.lineCap = "round";
    items.forEach(function (it) {
      if (it.kind === "edge") {
        ctx.setLineDash([4, 4]); ctx.strokeStyle = "rgba(128,128,140,.75)"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(it.a[0], it.a[1]); ctx.lineTo(it.b[0], it.b[1]); ctx.stroke(); ctx.setLineDash([]);
      } else if (it.kind === "bond") {
        var mx = (it.a[0] + it.b[0]) / 2, my = (it.a[1] + it.b[1]) / 2;
        ctx.lineWidth = Math.max(2, 0.13 * scale);
        ctx.strokeStyle = tint(it.A.colour, -0.15); ctx.beginPath(); ctx.moveTo(it.a[0], it.a[1]); ctx.lineTo(mx, my); ctx.stroke();
        ctx.strokeStyle = tint(it.B.colour, -0.15); ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(it.b[0], it.b[1]); ctx.stroke();
      } else {
        drawAtom(it.A, it.p, it.i, fg);
      }
    });
    if (show.cell) {
      ctx.fillStyle = fg; ctx.font = "italic 13px serif";
      ["a", "b", "c"].forEach(function (k) {
        if (!axes[k]) return;
        var p = project(axes[k]); ctx.fillText(k, p[0] + 4, p[1] - 4);
      });
      var o = project([0, 0, 0]); ctx.fillText("O", o[0] - 12, o[1] + 12);
    }
  }
  function drawAtom(A, p, i, fg) {
    var r;
    if (A.kind === "q_peak") {
      r = Math.max(2.5, 0.16 * scale);
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(217,70,239,.14)"; ctx.fill();
      ctx.lineWidth = 1.6; ctx.strokeStyle = "#D946EF"; ctx.stroke();
    } else if (A.kind === "unknown") {
      r = Math.max(4, 0.32 * scale);
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 2 * Math.PI);
      ctx.fillStyle = "#FF1493"; ctx.fill();
      ctx.setLineDash([3, 2]); ctx.lineWidth = 1.5; ctx.strokeStyle = "#000"; ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = "#fff"; ctx.font = "bold 11px system-ui"; ctx.fillText("?", p[0] - 3, p[1] + 4);
    } else {
      r = Math.max(3.5, (0.22 + 0.24 * (A.radius || 1)) * scale);
      var g = ctx.createRadialGradient(p[0] - r / 3, p[1] - r / 3, r * 0.15, p[0], p[1], r);
      g.addColorStop(0, tint(A.colour, 0.55)); g.addColorStop(0.55, A.colour); g.addColorStop(1, tint(A.colour, -0.45));
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 2 * Math.PI);
      ctx.fillStyle = g; ctx.fill();
      ctx.lineWidth = 0.8; ctx.strokeStyle = "rgba(0,0,0,.35)"; ctx.stroke();
    }
    if (i === selected) {
      ctx.beginPath(); ctx.arc(p[0], p[1], r + 4, 0, 2 * Math.PI);
      ctx.lineWidth = 2; ctx.strokeStyle = "#0071e3"; ctx.stroke();
    }
    if (show.labels) {
      ctx.fillStyle = A.kind === "q_peak" ? "#B026C9" : fg;
      ctx.font = A.kind === "q_peak" ? "10px system-ui,sans-serif" : "12px system-ui,sans-serif";
      var text = A.kind === "q_peak" && A.height != null ? A.label + " " + Number(A.height).toFixed(2) : A.label;
      ctx.fillText(text, p[0] + r + 2, p[1] - r + 2);
    }
    hits.push({ i: i, x: p[0], y: p[1], r: r + 3, z: p[2] });
  }
  function fmt(v, d) { return v == null ? T.unknown : Number(v).toFixed(d); }
  function describe(A) {
    var fr = A.frac.map(function (v) { return Number(v).toFixed(4); }).join(", ");
    if (A.kind === "q_peak") {
      return A.label + " · " + f(T.q_info, { h: fmt(A.height, 2), rank: A.rank, xyz: fr });
    }
    var u = A.u_iso != null ? "Uiso " + fmt(A.u_iso, 4) : (A.u_eq != null ? "Ueq " + fmt(A.u_eq, 4) + T.aniso : "U " + T.unknown);
    var head = A.kind === "unknown"
      ? A.label + " · " + f(T.unknown_site, { el: A.element })
      : A.label + " · " + f(T.element, { el: A.element });
    return head + " · " + f(T.frac, { xyz: fr }) + " · " + T.occ + " " + fmt(A.occupancy, 3) + (A.occupancy_fixed ? T.fixed : "") + " · " + u + " Å²";
  }
  var dragging = false, moved = 0, lastX = 0, lastY = 0;
  canvas.addEventListener("pointerdown", function (e) { dragging = true; moved = 0; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); canvas.style.cursor = "grabbing"; });
  canvas.addEventListener("pointermove", function (e) {
    if (!dragging) return;
    var dx = e.clientX - lastX, dy = e.clientY - lastY; moved += Math.abs(dx) + Math.abs(dy);
    rotY += dx * 0.01; rotX += dy * 0.01; lastX = e.clientX; lastY = e.clientY; draw();
  });
  canvas.addEventListener("pointerup", function (e) {
    dragging = false; canvas.style.cursor = "grab";
    if (moved > 4) return;
    var rect = canvas.getBoundingClientRect(), x = e.clientX - rect.left, y = e.clientY - rect.top, best = null;
    hits.forEach(function (h) { var d = Math.hypot(h.x - x, h.y - y); if (d <= h.r && (!best || h.z > best.z)) best = h; });
    selected = best ? best.i : -1;
    info.textContent = best ? describe(atoms[best.i]) : defaultInfo;
    draw();
  });
  canvas.addEventListener("wheel", function (e) { e.preventDefault(); zoom = Math.min(8, Math.max(0.2, zoom * Math.exp(-e.deltaY * 0.0015))); draw(); }, { passive: false });
  canvas.addEventListener("keydown", function (e) {
    var k = e.key;
    if (k === "ArrowLeft") rotY -= 0.1; else if (k === "ArrowRight") rotY += 0.1;
    else if (k === "ArrowUp") rotX -= 0.1; else if (k === "ArrowDown") rotX += 0.1;
    else if (k === "+" || k === "=") zoom = Math.min(8, zoom * 1.15); else if (k === "-") zoom = Math.max(0.2, zoom / 1.15);
    else return;
    e.preventDefault(); draw();
  });
  function toggle(id, key) {
    var b = document.getElementById(id);
    b.addEventListener("click", function () { show[key] = !show[key]; b.setAttribute("aria-pressed", String(show[key])); fit(); draw(); });
  }
  toggle("bl", "labels"); toggle("bq", "q"); toggle("bc", "cell");
  document.getElementById("br").addEventListener("click", function () { rotX = -0.45; rotY = 0.65; zoom = 1; selected = -1; info.textContent = defaultInfo; draw(); });

  var counts = P.counts || {};
  var defaultInfo = T.help;
  info.textContent = defaultInfo;
  canvas.setAttribute("aria-label", f(T.aria, { atoms: counts.atoms || 0, q: counts.q_peaks || 0, unk: counts.unknown_sites || 0 }));
  var elements = [];
  real.forEach(function (a) { if (a.kind === "atom" && elements.indexOf(a.element) < 0) elements.push(a.element); });
  function line(parent, swClass, colour, text) {
    var row = document.createElement("div"), sw = document.createElement("span");
    sw.className = "sw " + (swClass || ""); if (colour) sw.style.background = colour;
    row.appendChild(sw); row.appendChild(document.createTextNode(text)); parent.appendChild(row);
  }
  elements.forEach(function (el) {
    var a = real.filter(function (x) { return x.element === el; })[0];
    line(legend, "", a.colour, el);
  });
  if (counts.q_peaks) line(legend, "ring", null, T.legend_q);
  if (counts.unknown_sites) line(legend, "unk", null, T.legend_unknown);
  var conf = document.createElement("div");
  conf.textContent = T.confidence;
  conf.style.marginTop = "4px"; legend.appendChild(conf);

  var boundary = P.boundary_bonds || [];
  var lines = [T.display + (P.source ? " " + f(T.source, { s: P.source }) : ""),
               f(T.bonds, { n: counts.bonds_drawn || 0 })];
  if (boundary.length) {
    lines.push(f(T.boundary, { n: boundary.length }));
    boundary.slice(0, 8).forEach(function (b) {
      lines.push("  " + b.a + "–" + b.b + "  " + Number(b.distance).toFixed(3) + " Å " + f(T.translation, { t: b.translation.join(",") }));
    });
    if (boundary.length > 8) lines.push("  " + f(T.more, { n: boundary.length - 8 }));
  }
  lines.forEach(function (t) { var d = document.createElement("div"); d.textContent = t; d.style.whiteSpace = "pre-wrap"; note.appendChild(d); });
  if (P.mock) document.getElementById("mock").hidden = false;
  window.addEventListener("resize", resize);
  fit();
  resize();
})();
</script></body></html>"""


def viewer_document(payload: dict[str, Any], lang: str = "zh") -> str:
    """The complete HTML document for one model, with its words in ``lang``."""
    texts = VIEWER_TEXT.get(lang, VIEWER_TEXT["zh"])
    data = dict(payload)
    data["text"] = texts
    cell = data.get("cell") or {}
    try:
        m = orthogonalisation_matrix(
            (cell["a"], cell["b"], cell["c"], cell["alpha"], cell["beta"], cell["gamma"])
        )
        data["axes"] = {
            "a": list(frac_to_cart(m, (1.0, 0.0, 0.0))),
            "b": list(frac_to_cart(m, (0.0, 1.0, 0.0))),
            "c": list(frac_to_cart(m, (0.0, 0.0, 1.0))),
        }
    except (KeyError, ValueError):
        data["axes"] = {}
    # Every "<", ">" and "&" becomes a JSON unicode escape, so no file-supplied text can end the
    # script element or open an HTML comment inside it; JSON.parse restores the characters.
    text = (
        json.dumps(data, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return _TEMPLATE.replace("__LANG__", texts["html_lang"]).replace("__PAYLOAD__", text)


def viewer_html(payload: dict[str, Any] | None, *, lang: str = "zh", height: int = 560) -> str:
    """The sandboxed iframe that hosts :func:`viewer_document`."""
    if not payload:
        return empty_viewer(lang)
    texts = VIEWER_TEXT.get(lang, VIEWER_TEXT["zh"])
    document = html.escape(viewer_document(payload, lang), quote=True)
    return (
        f'<iframe title="{html.escape(texts["title"])}" sandbox="allow-scripts" referrerpolicy="no-referrer" '
        f'loading="eager" style="width:100%;height:{int(height)}px;border:1px solid #d7d7dc;'
        f'border-radius:10px;background:#fbfbfd" srcdoc="{document}"></iframe>'
    )
