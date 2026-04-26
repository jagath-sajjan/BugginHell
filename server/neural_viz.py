from __future__ import annotations

import html
import json
from typing import Sequence
from uuid import uuid4


ACTION_NAMES = ["read_file", "run_test", "search_symbol", "trace_caller", "commit_location"]
FEATURE_NAMES = [
    "steps_left",
    "failed",
    "assert",
    "zerodivision",
    "error",
    "none",
    "bool_logic",
    "has_src_file",
    "explored",
    "explored_deep",
    "symbol_hit",
    "symbol_miss",
    "caller_hit",
]


def _vector(values: Sequence[float] | None, fallback: int) -> list[float]:
    if not values:
        return [0.0] * fallback
    return [float(value) for value in values]


def _matrix(values: Sequence[Sequence[float]] | None, rows: int, cols: int) -> list[list[float]]:
    if not values:
        return [[0.0] * cols for _ in range(rows)]
    return [[float(cell) for cell in row] for row in values]


def render_neural_network_svg(
    *,
    features: Sequence[float],
    h1: Sequence[float],
    h2: Sequence[float],
    logits: Sequence[float],
    probs: Sequence[float],
    c1: Sequence[Sequence[float]],
    c2: Sequence[Sequence[float]],
    c3: Sequence[Sequence[float]],
    value_estimate: float = 0.0,
    step_index: int = 0,
    episode_index: int = 0,
    feature_names: Sequence[str] | None = None,
    action_id: int | None = None,
    action_name: str | None = None,
    reward: float | None = None,
    title: str = "Actor-Critic Trace",
) -> str:
    viz_id = f"nn_{uuid4().hex}"
    payload = {
        "features": _vector(features, len(FEATURE_NAMES)),
        "h1": _vector(h1, 64),
        "h2": _vector(h2, 64),
        "logits": _vector(logits, len(ACTION_NAMES)),
        "probs": _vector(probs, len(ACTION_NAMES)),
        "c1": _matrix(c1, len(FEATURE_NAMES), 64),
        "c2": _matrix(c2, 64, 64),
        "c3": _matrix(c3, 64, len(ACTION_NAMES)),
        "value": float(value_estimate),
        "step": int(step_index),
        "episode": int(episode_index),
        "feature_names": list(feature_names or FEATURE_NAMES),
        "action_names": ACTION_NAMES,
        "action_id": None if action_id is None else int(action_id),
        "action_name": action_name or "",
        "reward": None if reward is None else float(reward),
        "title": title,
    }

    inner_doc = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body style="margin:0;background:transparent;">
<div id="{viz_id}_wrap" style="padding:0.75rem 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;">
  <style>
    #{viz_id}_wrap * {{ box-sizing:border-box; }}
    #{viz_id}_canvas {{ display:block; width:100%; height:520px; border-radius:20px; border:1px solid rgba(255,255,255,0.08); background:#08111b; }}
    #{viz_id}_meta {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }}
    #{viz_id}_meta .pill {{ font-size:11px; padding:4px 10px; border-radius:999px; background:rgba(255,255,255,0.05); color:#e9edf3; }}
  </style>
  <canvas id="{viz_id}_canvas" height="520"></canvas>
  <div id="{viz_id}_meta">
    <span class="pill">episode {payload["episode"]}</span>
    <span class="pill">step {payload["step"]}</span>
    <span class="pill">action {payload["action_name"] or "unknown"}</span>
    <span class="pill">V(s) {payload["value"]:.3f}</span>
    <span class="pill">reward {"pending" if payload["reward"] is None else f"{payload['reward']:.2f}"}</span>
  </div>
</div>

<script>
(() => {{
  const root = document.getElementById({json.dumps(viz_id + "_wrap")});
  const canvas = document.getElementById({json.dumps(viz_id + "_canvas")});
  if (!root || !canvas) return;
  const ctx = canvas.getContext('2d');
  const payload = {json.dumps(payload)};

  const COLORS = {{
    bg: '#08111b',
    ink: '#e9edf3',
    blue: '#4f8cff',
    orange: '#ff8a3d',
    yellow: '#ffd166',
  }};

  const state = {{ pulses: [], startedAt: null, rafId: null }};

  function selectedActionIndex() {{
    if (Number.isInteger(payload.action_id)) return payload.action_id;
    if (payload.action_name) {{
      const idx = payload.action_names.indexOf(payload.action_name);
      if (idx >= 0) return idx;
    }}
    return payload.probs.indexOf(Math.max(...payload.probs));
  }}

  function clamp(value, low, high) {{
    return Math.max(low, Math.min(high, value));
  }}

  function nodeColor(value, scale=1) {{
    const t = clamp(value / Math.max(scale, 1e-6), -1, 1);
    if (t > 0.08) return COLORS.orange;
    if (t < -0.08) return COLORS.blue;
    return COLORS.ink;
  }}

  function edgeColor(value, alpha) {{
    return value >= 0
      ? `rgba(255,138,61,${{alpha}})`
      : `rgba(79,140,255,${{alpha}})`;
  }}

  function topEntries(matrix, limit) {{
    const entries = [];
    for (let i = 0; i < matrix.length; i++) {{
      for (let j = 0; j < matrix[i].length; j++) {{
        const value = matrix[i][j] || 0;
        entries.push({{ fromIdx: i, toIdx: j, value, mag: Math.abs(value) }});
      }}
    }}
    entries.sort((a, b) => b.mag - a.mag);
    return entries.slice(0, limit);
  }}

  function topNodeIndices(values, limit) {{
    return values
      .map((value, idx) => ({{ idx, value, mag: Math.abs(value || 0) }}))
      .sort((a, b) => b.mag - a.mag)
      .slice(0, limit)
      .map(item => item.idx);
  }}

  function topEdgesToAction(matrix, actionIdx, limit) {{
    return matrix
      .map((row, fromIdx) => {{
        const value = row[actionIdx] || 0;
        return {{ fromIdx, toIdx: actionIdx, value, mag: Math.abs(value) }};
      }})
      .sort((a, b) => b.mag - a.mag)
      .slice(0, limit);
  }}

  function layout(width, height) {{
    const margin = {{ top: 74, bottom: 38, left: 40, right: 48 }};
    const x = [
      margin.left + width * 0.10,
      margin.left + width * 0.33,
      margin.left + width * 0.59,
      width - margin.right - width * 0.11,
    ];
    const sizes = [payload.features.length, payload.h1.length, payload.h2.length, payload.logits.length];
    const layers = sizes.map((n, layerIdx) => {{
      const usable = height - margin.top - margin.bottom;
      const gap = usable / Math.max(1, n - 1);
      return Array.from({{ length: n }}, (_, i) => ({{
        x: x[layerIdx],
        y: margin.top + (n === 1 ? usable / 2 : i * gap),
      }}));
    }});
    return {{ x, layers }};
  }}

  function buildPulsePlan() {{
    const actionIdx = selectedActionIndex();
    const edges1 = topEntries(payload.c1, 18).map((edge, idx) => ({{
      ...edge, layer: 0, startAt: idx * 55, duration: 1250
    }}));
    const edges2 = topEntries(payload.c2, 20).map((edge, idx) => ({{
      ...edge, layer: 1, startAt: 1200 + idx * 48, duration: 1350
    }}));
    const edges3 = topEdgesToAction(payload.c3, actionIdx, 12).map((edge, idx) => ({{
      ...edge, layer: 2, startAt: 2550 + idx * 60, duration: 1450
    }}));
    return [...edges1, ...edges2, ...edges3];
  }}

  function drawText(text, x, y, size=12, color=COLORS.ink, align='left') {{
    ctx.font = `${{size}}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.fillStyle = color;
    ctx.textAlign = align;
    ctx.fillText(text, x, y);
  }}

  function drawNode(x, y, radius, fill, strokeWidth=1.2) {{
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = 'rgba(233,237,243,0.16)';
    ctx.lineWidth = strokeWidth;
    ctx.stroke();
  }}

  function drawBase(width, height, layers, xCols) {{
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, width, height);

    drawText(payload.title, 24, 34, 24, COLORS.ink);
    drawText('Input', xCols[0], 56, 14, COLORS.ink, 'center');
    drawText('Hidden 1', xCols[1], 56, 14, COLORS.ink, 'center');
    drawText('Hidden 2', xCols[2], 56, 14, COLORS.ink, 'center');
    drawText('Output', xCols[3], 56, 14, COLORS.ink, 'center');

    const edgesByLayer = [
      topEntries(payload.c1, 18),
      topEntries(payload.c2, 20),
      topEdgesToAction(payload.c3, selectedActionIndex(), 12),
    ];

    edgesByLayer.forEach((edges, layerIdx) => {{
      edges.forEach(edge => {{
        const from = layers[layerIdx][edge.fromIdx];
        const to = layers[layerIdx + 1][edge.toIdx];
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.strokeStyle = edgeColor(edge.value, Math.min(0.34, 0.08 + edge.mag * 0.75));
        ctx.lineWidth = Math.min(2.2, 0.4 + edge.mag * 2.2);
        ctx.stroke();
      }});
    }});

    payload.features.forEach((value, idx) => {{
      const node = layers[0][idx];
      drawNode(node.x, node.y, 8.5, nodeColor(value));
      drawText(payload.feature_names[idx], node.x - 16, node.y + 3, 11, COLORS.ink, 'right');
      drawText(value.toFixed(2), node.x + 15, node.y + 3, 11, COLORS.ink);
    }});

    const showH1 = new Set(topNodeIndices(payload.h1, 10));
    payload.h1.forEach((value, idx) => {{
      const node = layers[1][idx];
      drawNode(node.x, node.y, showH1.has(idx) ? 5.2 : 3.6, nodeColor(value));
    }});

    const showH2 = new Set(topNodeIndices(payload.h2, 10));
    payload.h2.forEach((value, idx) => {{
      const node = layers[2][idx];
      drawNode(node.x, node.y, showH2.has(idx) ? 5.2 : 3.6, nodeColor(value));
    }});

    const chosen = selectedActionIndex();
    payload.logits.forEach((value, idx) => {{
      const node = layers[3][idx];
      const prob = (payload.probs[idx] || 0) * 100;
      drawNode(node.x, node.y, idx === chosen ? 15.5 : 13.5, nodeColor(value, 3.0), idx === chosen ? 2.2 : 1.2);
      if (idx === chosen) {{
        ctx.beginPath();
        ctx.arc(node.x, node.y, 19, 0, Math.PI * 2);
        ctx.strokeStyle = COLORS.yellow;
        ctx.lineWidth = 2;
        ctx.stroke();
      }}
      drawText(payload.action_names[idx], node.x + 22, node.y - 2, 12, COLORS.ink);
      drawText(`${{prob.toFixed(1)}}%`, node.x + 22, node.y + 12, 11, COLORS.ink);
    }});

    drawText(`V(s) = ${{payload.value.toFixed(3)}}`, width - 20, height - 12, 12, COLORS.ink, 'right');
    drawText('blue = negative  orange = positive  yellow = live signal', 24, height - 12, 11, COLORS.ink);
  }}

  function drawPulses(layers, elapsed) {{
    state.pulses.forEach(pulse => {{
      const local = (elapsed - pulse.startAt) / pulse.duration;
      if (local < 0 || local > 1) return;
      const from = layers[pulse.layer][pulse.fromIdx];
      const to = layers[pulse.layer + 1][pulse.toIdx];
      const px = from.x + (to.x - from.x) * local;
      const py = from.y + (to.y - from.y) * local;
      const alpha = 0.35 + 0.65 * Math.sin(local * Math.PI);
      ctx.beginPath();
      ctx.arc(px, py, 4 + pulse.mag * 11, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255,209,102,${{Math.max(0.35, alpha)}})`;
      ctx.fill();
    }});
  }}

  function frame(now) {{
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(760, rect.width || 760);
    const height = 520;
    if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {{
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.scale(dpr, dpr);
    }}

    if (state.startedAt === null) {{
      state.startedAt = now;
      state.pulses = buildPulsePlan();
    }}

    const {{ x, layers }} = layout(width, height);
    drawBase(width, height, layers, x);
    drawPulses(layers, now - state.startedAt);

    const endTime = state.pulses.reduce((best, pulse) => Math.max(best, pulse.startAt + pulse.duration), 0);
    if (now - state.startedAt < endTime + 260) {{
      state.rafId = requestAnimationFrame(frame);
    }} else {{
      state.rafId = null;
    }}
  }}

  state.rafId = requestAnimationFrame(frame);
  new ResizeObserver(() => {{
    if (state.rafId !== null) cancelAnimationFrame(state.rafId);
    state.startedAt = null;
    state.pulses = [];
    state.rafId = requestAnimationFrame(frame);
  }}).observe(root);
}})();
</script>
</body>
</html>
"""

    return (
        '<iframe '
        'style="width:100%;height:610px;border:0;display:block;background:transparent;" '
        'sandbox="allow-scripts" '
        f'srcdoc="{html.escape(inner_doc, quote=True)}"></iframe>'
    )
