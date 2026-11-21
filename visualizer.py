# -*- coding: utf-8 -*-
"""visualizer.py - lianghua 量化类脑可视化：Spike Raster Plot + 放电活动图。

# 生物依据：神经元放电图（Spike Raster Plot）是 SNN 最直观的诊断/展示工具——
#   每行一个神经元，每列一个时间步，黑点 = 该神经元在该 tick 放电。
#   训练后不同标签的放电模式在"神经元空间"里会分开（类分离），
#   一张栅格图就能看出"脑"是否真的学到了类别差异。
# 简化说明：优先用 PIL（Pillow）出图（零重依赖，跨平台）；matplotlib 有就用作增强。
#   spike 数据从 lianghua.QuantizedBrain 的 fire()/propagate() 返回值里收集，
#   不需要额外存盘。PIL 不可用时打印提示（不崩，不影响训练）。
# 参数来源：图尺寸 / 颜色 / 字体 = 工程简化，保证 README 里看着清爽。
#
# 用法：
#   训练时自动出图：
#     python cli.py train --brain brain.npz --data samples.tsv --plot spikes.png
#   或单独调 visualizer：
#     python visualizer.py --brain brain.npz --text 猫 狗 汽车 --out raster.png
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 绘图后端：PIL 优先（零重依赖），matplotlib 可选增强
try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def collect_raster(brain, texts, ticks=6, salt=0, top_sub=None):
    """收集一组文本在指定子脑上的 spike 栅格。

    返回 (raster, meta)：
      raster : 2D bool 数组 [n_neurons, texts*ticks]（每个 tick 哪些神经元放电）
      meta   : {"texts": [...], "ticks": int, "n_neurons": int, "sub": int}
    """
    import encoder
    sub = top_sub if top_sub is not None else 0
    seeds_all, wts_all = [], []
    for t in texts:
        s, w = encoder.text_to_seeds(t, brain.N, salt=salt)
        seeds_all.append(s)
        wts_all.append(w)

    n_cols = len(texts) * ticks
    raster = np.zeros((brain.N, n_cols), dtype=bool)
    for ti, (s, w) in enumerate(zip(seeds_all, wts_all)):
        cur_s, cur_w = s, w
        for t in range(ticks):
            f, sync, fired = brain.fire(sub, cur_s, cur_w, tick=ti * 1000 + t)
            col = ti * ticks + t
            if len(fired):
                raster[fired, col] = True
            if f == 0:
                cur_s, cur_w = s, w
            else:
                nb = brain.neighbors[sub][fired].reshape(-1)
                ww = brain.weights[sub][fired].reshape(-1)
                pos = ww > 0
                cur_s, cur_w = nb[pos], ww[pos]
                if len(cur_s) == 0:
                    cur_s, cur_w = s, w
    meta = {"texts": list(texts), "ticks": ticks, "n_neurons": brain.N, "sub": int(sub)}
    return raster, meta


def raster_plot(raster, out_path, title=None, max_neurons=200, ticks=6, texts=None,
                use_mpl=False):
    """画 Spike Raster Plot（栅格图）：每行神经元，每列时间步，黑点=放电。

    用 PIL 出 PNG（零重依赖）；use_mpl=True 且 matplotlib 可用时用 matplotlib。
    """
    if use_mpl and _HAS_MPL:
        return _raster_mpl(raster, out_path, title, max_neurons, ticks, texts)
    if not _HAS_PIL:
        print("[visualizer] PIL/matplotlib 都不可用，跳过出图（不影响训练）")
        return None
    # 按放电次数排序（活跃神经元在上）
    counts = raster.sum(axis=1)
    order = np.argsort(-counts)
    order = order[:max_neurons]
    order = np.sort(order)
    show = raster[order]
    n_rows, n_cols = show.shape
    n_texts = len(texts) if texts else max(1, n_cols // max(1, ticks))

    # 尺寸：每 tick 一列，每神经元一行；放大保证看着清楚
    cell_w = 14
    cell_h = 6
    margin_l, margin_r, margin_t, margin_b = 120, 30, 50, 60
    W = margin_l + n_cols * cell_w + margin_r
    H = margin_t + n_rows * cell_h + margin_b
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)

    # 标题
    dr.text((margin_l, 10), (title or "Spike Raster Plot"), fill="black")

    # 栅格：每个 tick 一列，放电=黑点
    for c in range(n_cols):
        x = margin_l + c * cell_w + cell_w // 2
        col = show[:, c]
        if col.any():
            for r in np.where(col)[0]:
                y = margin_t + int(r) * cell_h
                dr.line([(x, y), (x, y + cell_h - 1)], fill="black", width=1)
        # 列分隔线（每个 text 之间）
        if c % ticks == ticks - 1 and c < n_cols - 1:
            dr.line([(x + 3, margin_t), (x + 3, margin_t + n_rows * cell_h)],
                    fill="lightgray", width=1)

    # 左边：神经元编号（每 10 个标一次）
    for r in range(0, n_rows, max(1, n_rows // 10)):
        y = margin_t + r * cell_h
        dr.text((4, y), str(int(order[r])), fill="gray")

    # 底部：tick 标签
    for c in range(n_cols):
        x = margin_l + c * cell_w
        if c % ticks == 0 and texts:
            dr.text((x - 4, H - 30), "T%d" % (c // ticks), fill="gray")
    # 坐标轴
    dr.line([(margin_l, margin_t - 4), (margin_l, margin_t + n_rows * cell_h)],
            fill="black", width=1)
    dr.line([(margin_l, margin_t + n_rows * cell_h),
             (margin_l + n_cols * cell_w, margin_t + n_rows * cell_h)],
            fill="black", width=1)
    # 左下标注
    dr.text((margin_l, H - 12), "%d neurons (top %d by activity) x %d ticks, sub %s"
            % (n_rows, max_neurons, n_cols, ""), fill="gray")

    img.save(out_path)
    return out_path


def _raster_mpl(raster, out_path, title, max_neurons, ticks, texts):
    """matplotlib 版栅格图（可选增强）。"""
    counts = raster.sum(axis=1)
    order = np.argsort(-counts)[:max_neurons]
    order = np.sort(order)
    show = raster[order]
    n_rows, n_cols = show.shape
    n_texts = len(texts) if texts else max(1, n_cols // max(1, ticks))
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.35), 8), dpi=120)
    for c in range(n_cols):
        col = show[:, c]
        if col.any():
            ax.scatter(np.where(col)[0], np.full(int(col.sum()), c),
                       s=3, c="black", marker="|", linewidths=0.6)
    for i in range(1, n_texts):
        ax.axvline(x=i * ticks, color="lightgray", linewidth=0.8, linestyle="--")
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_yticks(order[::max(1, len(order) // 10)])
    ax.set_xlabel("tick (%d texts x %d ticks)" % (n_texts, ticks))
    ax.set_ylabel("neuron (top %d by activity)" % max_neurons)
    ax.set_title(title or "Spike Raster Plot", fontsize=11)
    ax.grid(axis="x", color="lightgray", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def activity_plot(brain, texts, out_path, ticks=6, salt=0, title=None):
    """画每个子脑的放电活动（堆叠柱状图）：展示"不同类别激活不同子脑"的类分离。"""
    if not _HAS_PIL:
        print("[visualizer] PIL 不可用，跳过活动图")
        return None
    import encoder
    n_sub = brain.NSUB
    counts = np.zeros((len(texts), n_sub), dtype=int)
    for i, t in enumerate(texts):
        s, w = encoder.text_to_seeds(t, brain.N, salt=salt)
        for sub in range(n_sub):
            r = brain.propagate(sub, s, w, ticks=ticks)
            counts[i, sub] = len(r["fired"])

    # 简单堆叠柱状图（PIL）
    cell_w, cell_h = 60, 20
    margin_l, margin_r, margin_t, margin_b = 120, 40, 50, 80
    max_count = int(counts.max()) if counts.size else 1
    W = margin_l + len(texts) * cell_w + margin_r
    H = margin_t + 200 + margin_b
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)
    dr.text((margin_l, 10), (title or "Firing Activity per Sub-brain"), fill="black")

    colors = ["#222222", "#444444", "#666666", "#888888",
              "#aaaaaa", "#cccccc", "#996633", "#6699cc"]
    for i, t in enumerate(texts):
        x0 = margin_l + i * cell_w + 8
        y_top = margin_t + 20
        cum = 0
        for sub in range(n_sub):
            h = int(counts[i, sub] / max(1, max_count) * 180)
            if h > 0:
                c = colors[sub % len(colors)]
                # 堆叠柱：从底往上堆，y 轴向下增长
                y0 = y_top + cum
                y1 = y0 + h
                dr.rectangle([x0, y0, x0 + cell_w - 16, y1],
                             fill=c, outline="white", width=1)
            cum += h
        # x 标签
        dr.text((x0, H - 40), t[:8], fill="gray")
    # y 轴
    dr.line([(margin_l, margin_t), (margin_l, margin_t + 200)], fill="black", width=1)
    dr.line([(margin_l, margin_t + 200), (W - margin_r, margin_t + 200)], fill="black", width=1)
    # 图例
    lx = margin_l + 10
    for sub in range(n_sub):
        c = colors[sub % len(colors)]
        dr.rectangle([lx, margin_t + 210 + sub * 14, lx + 10, margin_t + 218 + sub * 14], fill=c)
        dr.text((lx + 14, margin_t + 210 + sub * 14), "sub%d" % sub, fill="black")
    img.save(out_path)
    return out_path


def selftest():
    """自检：收集 spike → 出栅格图 + 活动图，验证出图通路。"""
    import lianghua
    b = lianghua.QuantizedBrain.build(nsub=4, n=512, k=16, seed=20261120)
    texts = ["猫", "狗", "汽车", "飞机", "鱼"]
    b.train_samples([(t, t) for t in texts], epochs=1, ticks=4)
    if not _HAS_PIL and not _HAS_MPL:
        print("visualizer 自检：PIL/matplotlib 都不可用，跳过出图")
        return 0
    out_dir = os.path.join(HERE, "viz")
    os.makedirs(out_dir, exist_ok=True)
    p1 = os.path.join(out_dir, "raster_selftest.png")
    p2 = os.path.join(out_dir, "activity_selftest.png")
    raster, meta = collect_raster(b, texts, ticks=4)
    raster_plot(raster, p1, ticks=4, texts=texts,
                title="Self-test raster (4 sub x 512 neurons)")
    activity_plot(b, texts, p2, ticks=4, title="Self-test activity")
    print("visualizer 自检：栅格图 → %s（%d KB）" % (p1, os.path.getsize(p1) // 1024))
    print("visualizer 自检：活动图 → %s（%d KB）" % (p2, os.path.getsize(p2) // 1024))
    print("raster 形状 %s，总放电 %d 个" % (str(raster.shape), int(raster.sum())))
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="lianghua 可视化（栅格图/活动图，PIL 零重依赖）")
    ap.add_argument("--brain", help="QuantizedBrain .npz 路径")
    ap.add_argument("--text", nargs="*", default=["猫", "狗", "汽车"])
    ap.add_argument("--out", default=None, help="输出 PNG（默认 viz/raster.png）")
    ap.add_argument("--ticks", type=int, default=6)
    ap.add_argument("--salt", type=int, default=0)
    ap.add_argument("--sub", type=int, default=None, help="栅格图只看某个子脑（默认 0）")
    ap.add_argument("--max-neurons", type=int, default=200)
    ap.add_argument("--activity", action="store_true", help="也出活动图")
    a = ap.parse_args()
    if a.brain:
        import lianghua
        b = lianghua.QuantizedBrain.load(a.brain)
        out_dir = os.path.join(HERE, "viz")
        os.makedirs(out_dir, exist_ok=True)
        out = a.out or os.path.join(out_dir, "raster.png")
        raster, meta = collect_raster(b, a.text, ticks=a.ticks, salt=a.salt, top_sub=a.sub)
        p = raster_plot(raster, out, ticks=a.ticks, texts=a.text, max_neurons=a.max_neurons,
                        title="Raster: %d neurons x %d texts (sub %s)"
                              % (a.max_neurons, len(a.text), a.sub or 0))
        print("栅格图 → %s" % p)
        if a.activity:
            p2 = out.rsplit(".", 1)[0] + "_activity.png"
            p2 = activity_plot(b, a.text, p2, ticks=a.ticks, salt=a.salt,
                               title="Activity: per sub-brain")
            print("活动图 → %s" % p2)
    else:
        sys.exit(selftest())
