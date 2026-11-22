"""Lightweight HTML5 video gallery server for inspecting collected D1 simulation episodes."""
import argparse
import json
from pathlib import Path
from aiohttp import web

HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>D1 仿真采集数据 · 可视化审查</title>
  <style>
    :root { color-scheme: dark; font: 15px system-ui, sans-serif; background: #0f1520; color: #eef3fb; }
    body { max-width: 1120px; margin: 28px auto; padding: 0 18px 36px; }
    h1 { font-size: 1.85rem; margin-bottom: .35rem; }
    p { color: #b5c5d8; line-height: 1.5; }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 18px 0 12px; }
    select, button { font: inherit; color: inherit; background: #1e2a3d; border: 1px solid #4c627c; border-radius: 6px; padding: 8px 10px; }
    button { cursor: pointer; }
    video { display: block; width: 100%; aspect-ratio: 16/9; background: #06090f; border-radius: 8px; }
    .meta { margin: 12px 0 22px; padding: 12px 14px; background: #162030; border: 1px solid #2e4057; border-radius: 8px; }
    .table { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; white-space: nowrap; font-size: 14px; }
    th, td { border-bottom: 1px solid #2e4057; text-align: left; padding: 9px 10px; }
    tr.active { background: #20314a; }
    tr:hover { background: #1a273b; cursor: pointer; }
    .ok { color: #6ee7b7; font-weight: 600; }
    .bad { color: #fca5a5; font-weight: 600; }
  </style>
</head>
<body>
  <main>
    <h1>D1 仿真采集数据 · 避障触碰按钮可视化审查</h1>
    <p id="summary">正在加载仿真采集数据清单…</p>
    <div class="controls">
      <label for="picker">选择样本</label>
      <select id="picker"></select>
      <button id="prev">上一条</button>
      <button id="next">下一条</button>
      <label><input id="slow" type="checkbox"> 0.5× 慢放</label>
    </div>
    <video id="player" controls autoplay loop muted playsinline></video>
    <div class="meta" id="detail"></div>
    <div class="table">
      <table>
        <thead>
          <tr><th>Seed</th><th>任务</th><th>指令</th><th>绕障侧距 (m)</th><th>障碍碰撞帧</th><th>按键行程 (mm)</th><th>目标入框</th><th>手物同框</th><th>状态</th></tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
  </main>
  <script type="module">
    const picker = document.getElementById("picker");
    const player = document.getElementById("player");
    const detail = document.getElementById("detail");
    const rowsEl = document.getElementById("rows");
    const slow = document.getElementById("slow");
    const data = await fetch("api/manifest").then(r => r.json());
    const items = data.episodes || [];
    const okCount = items.filter(x => x.accepted).length;
    document.getElementById("summary").textContent =
      `共展示 ${items.length} 条仿真样本（门禁通过 ${okCount} 条）· 第三视角全景 + 左上角 Q1 头相机 RGB + 五点目标标记。`;

    let current = 0;
    function select(idx) {
      if (!items.length) return;
      current = (idx + items.length) % items.length;
      picker.selectedIndex = current;
      const it = items[current];
      player.src = "videos/" + it.video;
      player.playbackRate = slow.checked ? 0.5 : 1.0;
      player.play().catch(() => {});
      detail.innerHTML =
        `<b>Seed ${it.seed}</b> · 任务：<code>${it.task}</code> · 状态：<span class="${it.accepted ? 'ok' : 'bad'}">${it.accepted ? '通过门禁' : '未通过'}</span><br>` +
        `指令：${it.instruction}<br>` +
        `绕障侧距：<b>${(it.max_lateral_detour_m ?? 0).toFixed(3)} m</b> · 障碍碰撞帧：<b>${it.obstacle_collision_steps ?? 0}</b> · ` +
        `按钮最大行程：<b>${((it.button_max_travel_m ?? 0) * 1000).toFixed(1)} mm</b> · ` +
        `目标入框：<b>${((it.target_in_frustum_fraction ?? 0) * 100).toFixed(0)}%</b> · 手物同框：<b>${((it.target_and_wrist_in_frustum_fraction ?? 0) * 100).toFixed(0)}%</b>`;
      [...rowsEl.children].forEach((tr, i) => tr.classList.toggle("active", i === current));
    }

    items.forEach((it, i) => {
      picker.add(new Option(`Seed ${it.seed} · ${it.task} · ${it.accepted ? "通过" : "未通过"}`, i));
      const tr = rowsEl.insertRow();
      tr.onclick = () => select(i);
      [
        it.seed, it.task, it.instruction,
        (it.max_lateral_detour_m ?? 0).toFixed(3),
        it.obstacle_collision_steps ?? 0,
        ((it.button_max_travel_m ?? 0) * 1000).toFixed(1),
        ((it.target_in_frustum_fraction ?? 0) * 100).toFixed(0) + "%",
        ((it.target_and_wrist_in_frustum_fraction ?? 0) * 100).toFixed(0) + "%",
        it.accepted ? "通过" : "未通过",
      ].forEach((val, ci) => {
        const td = tr.insertCell();
        td.textContent = val;
        if (ci === 8) td.className = it.accepted ? "ok" : "bad";
      });
    });

    picker.onchange = () => select(Number(picker.value));
    document.getElementById("prev").onclick = () => select(current - 1);
    document.getElementById("next").onclick = () => select(current + 1);
    slow.onchange = () => { player.playbackRate = slow.checked ? 0.5 : 1.0; };
    select(0);
  </script>
</body>
</html>
"""


def create_app(directory, access_key):
    directory = Path(directory)

    async def page(_):
        return web.Response(text=HTML, content_type="text/html")

    async def manifest(_):
        return web.json_response(json.loads((directory / "manifest.json").read_text()))

    async def video(request):
        name = request.match_info["name"]
        path = directory / name
        if Path(name).name != name or path.suffix != ".mp4" or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    app = web.Application()
    root = "/" + access_key + "/"
    app.router.add_get(root, page)
    app.router.add_get(root + "api/manifest", manifest)
    app.router.add_get(root + "videos/{name}", video)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--access-key-file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=31796)
    args = parser.parse_args()
    app = create_app(args.directory, args.access_key_file.read_text().strip())
    web.run_app(app, host="0.0.0.0", port=args.port, access_log=None)
