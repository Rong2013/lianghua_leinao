# -*- coding: utf-8 -*-
"""cli.py - lianghua 量化类脑的命令端。

# 生物依据：把"发育（编译拓扑）→ 学习（STDP）→ 回想（放电匹配）→
#   离线巩固（睡眠重放 / 稳态 / 修剪）"暴露成明确命令，对应神经系统的
#   发育期、可塑性期、记忆提取与睡眠巩固。
# 简化说明：单文件 argparse，无第三方 CLI 框架；只依赖 numpy。
# 参数来源：命令行默认值 = 开源骨架默认规模（8 子脑 × 2048），工程简化。

常用：
    python cli.py build  --out brain.npz
    python cli.py train  --brain brain.npz --data samples.tsv --epochs 2
    python cli.py think  --brain brain.npz --text "hello world"
    python cli.py eval   --brain brain.npz --data samples_test.tsv
    python cli.py arch   --brain brain.npz          # 类脑结构演示
    python cli.py sleep  --brain brain.npz --data samples_test.tsv
    python cli.py prune  --brain brain.npz --threshold 1
    python cli.py rplus  --brain brain.npz          # R+ 简化版一致性
    python cli.py stats  --brain brain.npz
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import encoder      # noqa: E402
import lianghua     # noqa: E402


# ---------------- 数据读取 ----------------

def load_samples(path):
    """读训练样本：支持 TSV(label\\ttext) / JSONL / 纯文本（label=行号）。"""
    samples = []
    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n\r")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.lstrip().startswith("{"):
                try:
                    o = json.loads(line)
                    samples.append((str(o.get("label", "样本%d" % i)), str(o.get("text", ""))))
                    continue
                except Exception:
                    pass
            if "\t" in line:
                label, text = line.split("\t", 1)
            elif "|" in line and line.count("|") == 1:
                label, text = line.split("|", 1)
            else:
                label, text = "样本%d" % i, line
            if text.strip():
                samples.append((label.strip(), text.strip()))
    return samples


def _eval(b, samples, top_k=4, ticks=6, salt=0, inhibit=False):
    """在内存里的脑上评估（不读盘），供 eval / arch / sleep / prune 复用。"""
    ok, total, score_sum = 0, 0, 0.0
    bad = []
    for label, text in samples:
        r = b.think(text, top_k=top_k, ticks=ticks, salt=salt, inhibit=inhibit)
        total += 1
        score_sum += r["score"]
        if r["label"] == label:
            ok += 1
        elif len(bad) < 12:
            bad.append((label, r["label"], round(r["score"], 3), text[:30]))
    return {"samples": total, "correct": ok,
            "accuracy": round(ok / max(1, total), 4),
            "avg_score": round(score_sum / max(1, total), 4), "errors": bad}


def _fmt_layers(h):
    return "感觉 %s / 联合 %s / 决策 %s" % (
        h["sensory"], h["association"], h["decision"])


# ---------------- 子命令 ----------------

def cmd_build(a):
    b = lianghua.QuantizedBrain.build(nsub=a.nsub, n=a.n, k=a.k, seed=a.seed)
    b.save(a.out)
    st = b.stats()
    print("已编译神经元 → %s" % a.out)
    print("  子脑 %d × 神经元 %d × 邻居 %d = %d 个神经元 / %d 个突触"
          % (st["subs"], b.N, b.K, st["neurons"], st["synapses"]))
    print("  内存 %.3f MB（state %.3f / weights %.3f / neighbors %.3f）"
          % (st["total_mb"], st["state_mb"], st["weights_mb"], st["neighbors_mb"]))
    print("  权重范围 [%d, %d]，均值 %.3f" % (st["weight_min"], st["weight_max"], st["weight_mean"]))
    print("  皮层分级：%s" % _fmt_layers(b.hierarchy()))
    return 0


def cmd_train(a):
    b = lianghua.QuantizedBrain.load(a.brain)
    samples = load_samples(a.data)
    if not samples:
        print("训练集为空：%s" % a.data)
        return 2
    print("加载 %s：%d 神经元，已有原型 %d 个" % (a.brain, b.NSUB * b.N, len(b.protos)))
    print("训练样本 %d 条，epochs=%d ticks=%d reward=%d%%"
          % (len(samples), a.epochs, a.ticks, a.reward))
    import time
    import metrics
    t0 = time.perf_counter()
    log = b.train_samples(samples, epochs=a.epochs, ticks=a.ticks,
                          salt=a.salt, reward=a.reward)
    dt = time.perf_counter() - t0
    # 指标可观测性（Prometheus 采集，零重依赖）
    metrics.tick_seconds(dt / max(1, a.epochs * a.ticks))
    metrics.saturation_rate(float(b.quant_stats()["saturated_cells"]) / max(1, b.NSUB * b.N))
    metrics.train_epoch()
    out = a.out or a.brain
    b.save(out)
    print("已保存 → %s（原型 %d 个，最后加强突触 %d）"
          % (out, len(b.protos), log[-1]["reinforced"] if log else 0))
    if a.plot:
        import visualizer
        import os
        os.makedirs(os.path.dirname(a.plot) if os.path.dirname(a.plot) else ".", exist_ok=True)
        # 用训练样本的前几条做栅格图（展示类分离）
        demo_texts = [t for _, t in samples[:6]]
        raster, meta = visualizer.collect_raster(b, demo_texts, ticks=a.ticks, salt=a.salt)
        p = visualizer.raster_plot(raster, a.plot, ticks=a.ticks, texts=demo_texts,
                                    title="训练栅格图：%d 样本 × %d ticks" % (len(demo_texts), a.ticks))
        print("栅格图 → %s" % p)
    return 0


def cmd_think(a):
    b = lianghua.QuantizedBrain.load(a.brain)
    import metrics
    metrics.think_call()
    r = b.think(a.text, top_k=a.top_k, ticks=a.ticks, salt=a.salt, inhibit=a.inhibit)
    metrics.accuracy(float(r.get("score", 0.0)))
    if a.json:
        print(json.dumps(r, ensure_ascii=False))
    else:
        print("输入: %s" % r["text"])
        print("门控子脑: %s（%s）" % (
            r["subs"], ", ".join("%d=%s" % (s, b.layer_of(s)) for s in r["subs"])))
        print("总放电: %d" % r["spikes"])
        if r["label"] is not None:
            print("识别: %s（Jaccard %.3f）" % (r["label"], r["score"]))
        else:
            print("识别: 无匹配（还没训练？先跑 cli.py train）")
        if r.get("inhibit") and r.get("gains"):
            print("侧抑制增益: %s" % r["gains"])
        print("top 神经元: %s" % r["top"][:12])
    return 0


def cmd_stats(a):
    b = lianghua.QuantizedBrain.load(a.brain)
    st = b.stats()
    q = b.quant_stats()
    st.update(q)
    st["hierarchy"] = b.hierarchy()
    if a.json:
        print(json.dumps(st, ensure_ascii=False))
        return 0
    print("=== %s ===" % a.brain)
    print("规模: %d 子脑 × %d 神经元 × %d 邻居 = %d 神经元 / %d 突触"
          % (st["subs"], b.N, b.K, st["neurons"], st["synapses"]))
    print("内存: %.3f MB（state %.3f / weights %.3f / neighbors %.3f）"
          % (st["total_mb"], st["state_mb"], st["weights_mb"], st["neighbors_mb"]))
    print("权重: [%d, %d] 均值 %.3f" % (st["weight_min"], st["weight_max"], st["weight_mean"]))
    print("稀疏度: %.1f%%（非零突触 %d / %d）"
          % (st["sparsity"] * 100, st["weight_nonzero"], st["synapses"]))
    print("皮层分级: %s" % _fmt_layers(b.hierarchy()))
    print("量化: %s / %s，饱和单元 %d" % (st["dtype_state"], st["dtype_weights"], q["saturated_cells"]))
    print("原型: %d 个标签 / %d 个放电模式" % (st["protos"], st["patterns"]))
    return 0


def cmd_eval(a):
    b = lianghua.QuantizedBrain.load(a.brain)
    samples = load_samples(a.data)
    if not samples:
        print("数据集为空：%s" % a.data)
        return 2
    out = _eval(b, samples, top_k=a.top_k, ticks=a.ticks, salt=a.salt, inhibit=a.inhibit)
    import metrics
    metrics.accuracy(float(out["accuracy"]))
    if a.json:
        print(json.dumps(out, ensure_ascii=False))
        return 0
    print("样本 %d 条：正确 %d，准确率 %.1f%%，平均 Jaccard %.3f"
          % (out["samples"], out["correct"], out["accuracy"] * 100, out["avg_score"]))
    if out["errors"]:
        print("错例：")
        for l, p, s, t in out["errors"]:
            print("  真值=%-4s 预测=%-4s 分=%.3f  %s" % (l, p, s, t))
    return 0


def cmd_arch(a):
    """类脑结构演示：分级 / 丘脑中继 / 侧抑制 / DMN 自发活动。"""
    b = lianghua.QuantizedBrain.load(a.brain)
    h = b.hierarchy()
    print("=== 类脑结构（%s）===" % a.brain)
    print("皮层分级: %s" % _fmt_layers(h))
    print("  感觉 = 输入表征；联合 = 跨模态组合；决策 = 记忆匹配与输出")

    text = a.text
    seeds, wts = encoder.text_to_seeds(text, b.N)
    q = np.clip(seeds % 128 - 64, -128, 127).astype(np.int8)
    relay = b.thalamic_relay(q, top_k=a.top_k)
    print("\n丘脑中继（输入：%s）" % text)
    print("  放行子脑: %s（%s）" % (
        relay["subs"], ", ".join("%d=%s" % (s, relay["layers"][s]) for s in relay["subs"])))
    print("  相关度 top-4: %s" % [(s, r) for s, r in relay["rel"][:4]])

    r = b.think(text, top_k=a.top_k, ticks=a.ticks)
    counts = {s: v["spikes"] for s, v in r["per_sub"].items()}
    gains = b.lateral_inhibition(counts)
    ranked = sorted(counts.items(), key=lambda x: -x[1])
    print("\n侧抑制（各子脑放电数 → 增益百分比）")
    for s, c in ranked:
        print("  子脑%d(%s) 放电 %3d → 增益 %d%%"
              % (s, b.layer_of(s), c, gains.get(s, 100)))
    print("  识别: %s（Jaccard %.3f）" % (r["label"], r["score"]))

    d = b.dmn_idle(ticks=a.ticks)
    print("\nDMN 静息（无外部输入，决策层自发活动）")
    for s in d["subs"]:
        v = d["per_sub"][s]
        print("  子脑%d(%s) 每 tick 放电 %s  累计覆盖 %d 个神经元"
              % (s, b.layer_of(s), v["counts"], v["unique"]))
    print("  每 tick 自发率: %.4f%%（%d 次放电 / %d 个神经元·tick）"
          % (d["rate"] * 100, d["spikes"], len(d["subs"]) * b.N * a.ticks))

    if a.data:
        samples = load_samples(a.data)
        if samples:
            before = _eval(b, samples, top_k=a.top_k, ticks=a.ticks)
            sr = b.sleep_replay(epochs=a.epochs, ticks=a.ticks)
            after = _eval(b, samples, top_k=a.top_k, ticks=a.ticks)
            print("\n睡眠重放（%d 条原型 × %d 轮）" % (sr["replayed"], sr["epochs"]))
            print("  |w| 总量 %d → %d（%+.3f%%）"
                  % (sr["weight_abs_before"], sr["weight_abs_after"], sr["delta_pct"]))
            print("  留出/评估 %d 条：%.1f%% → %.1f%%"
                  % (before["samples"], before["accuracy"] * 100, after["accuracy"] * 100))
            if a.out:
                b.save(a.out)
                print("  已保存 → %s" % a.out)
    return 0


def cmd_sleep(a):
    b = lianghua.QuantizedBrain.load(a.brain)
    samples = load_samples(a.data) if a.data else []
    before = _eval(b, samples, top_k=a.top_k, ticks=a.ticks) if samples else None
    print("睡眠重放：%d 个标签 / %d 条原型，%d 轮 × %d ticks"
          % (len(b.protos), sum(len(v) for v in b.protos.values()),
             a.epochs, a.ticks))
    sr = b.sleep_replay(epochs=a.epochs, ticks=a.ticks, strength=a.strength)
    out = a.out or a.brain
    b.save(out)
    print("  重放 %d 条原型；|w| 总量 %d → %d（%+.3f%%）"
          % (sr["replayed"], sr["weight_abs_before"], sr["weight_abs_after"], sr["delta_pct"]))
    if before is not None:
        after = _eval(b, samples, top_k=a.top_k, ticks=a.ticks)
        print("  %s 上准确率：%.1f%% → %.1f%%（平均 Jaccard %.3f → %.3f）"
              % (a.data, before["accuracy"] * 100, after["accuracy"] * 100,
                 before["avg_score"], after["avg_score"]))
    print("已保存 → %s" % out)
    return 0


def cmd_prune(a):
    b = lianghua.QuantizedBrain.load(a.brain)
    samples = load_samples(a.data) if a.data else []
    before = _eval(b, samples, top_k=a.top_k, ticks=a.ticks) if samples else None
    st0 = b.stats()
    r = b.prune(a.threshold)
    st1 = b.stats()
    out = a.out or a.brain
    b.save(out)
    print("突触修剪：|w| <= %d" % a.threshold)
    print("  剪掉 %d / %d 个突触（本次 %.1f%%）；累计稀疏度 %.1f%% → %.1f%%"
          % (r["pruned"], r["total"], r["sparsity"] * 100,
             st0["sparsity"] * 100, st1["sparsity"] * 100))
    print("  权重范围 [%d, %d] → [%d, %d]"
          % (st0["weight_min"], st0["weight_max"], st1["weight_min"], st1["weight_max"]))
    if before is not None:
        after = _eval(b, samples, top_k=a.top_k, ticks=a.ticks)
        print("  %s 上准确率：%.1f%% → %.1f%%"
              % (a.data, before["accuracy"] * 100, after["accuracy"] * 100))
    print("已保存 → %s" % out)
    return 0


def cmd_homeostasis(a):
    b = lianghua.QuantizedBrain.load(a.brain)
    before = b.stats()
    r = b.homeostasis(target=a.target)
    after = b.stats()
    out = a.out or a.brain
    b.save(out)
    print("突触稳态：目标非零 |w| 均值 = %d" % a.target)
    print("  每个子脑非零 |w| 均值：%s → %s"
          % (r["mean_abs_before"], r["mean_abs_after"]))
    print("  全局 |w| 均值 %.3f → %.3f" % (before["weight_mean"], after["weight_mean"]))
    print("已保存 → %s" % out)
    return 0


def cmd_rplus(a):
    b = lianghua.QuantizedBrain.load(a.brain)
    import rplus_simple
    texts = [a.text] if a.text else None
    out = rplus_simple.selftest(brain=b, texts=texts, verbose=not a.json)
    if a.json:
        print(json.dumps(out, ensure_ascii=False))
    return 0 if out["fire_equal"] == out["fire_cases"] and out["prop_equal"] == out["prop_cases"] else 1


def cmd_selftest(a):
    return lianghua.selftest()


def cmd_verify(a):
    import runpy
    path = os.path.join(HERE, "verify_quant.py")
    runpy.run_path(path, run_name="__main__")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="lianghua", description="量化类脑命令端（编译 / 训练 / 推理 / 巩固 / 统计）")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="编译神经元（生成拓扑并保存）")
    b.add_argument("--out", required=True, help="输出 .npz 路径")
    b.add_argument("--nsub", type=int, default=8, help="子脑数（默认 8）")
    b.add_argument("--n", type=int, default=2048, help="每个子脑神经元数（默认 2048）")
    b.add_argument("--k", type=int, default=32, help="每个神经元邻居数（默认 32）")
    b.add_argument("--seed", type=int, default=20261120, help="随机种子")
    b.set_defaults(fn=cmd_build)

    t = sub.add_parser("train", help="INT8 STDP 训练")
    t.add_argument("--brain", required=True, help="build 出来的 .npz")
    t.add_argument("--data", required=True, help="训练集：TSV(label\\ttext) / JSONL / 纯文本")
    t.add_argument("--epochs", type=int, default=3, help="训练轮数（默认 3）")
    t.add_argument("--ticks", type=int, default=6, help="每个样本传播 tick 数（默认 6）")
    t.add_argument("--salt", type=int, default=0, help="编码加盐（默认 0）")
    t.add_argument("--reward", type=int, default=100,
                   help="奖赏调制百分比（100 = 不调制，默认 100）")
    t.add_argument("--out", help="另存路径（默认覆盖 --brain）")
    t.add_argument("--plot", help="可选：输出栅格图 PNG 路径（训练时自动收集 spike 并出图）")
    t.set_defaults(fn=cmd_train)

    k = sub.add_parser("think", help="推理：文本 → 放电 → 匹配记忆")
    k.add_argument("--brain", required=True, help="训练后的 .npz")
    k.add_argument("--text", required=True, help="输入文本")
    k.add_argument("--top-k", type=int, default=4, help="门控放行子脑数（默认 4）")
    k.add_argument("--ticks", type=int, default=6, help="传播 tick 数（默认 6）")
    k.add_argument("--salt", type=int, default=0, help="编码加盐（默认 0）")
    k.add_argument("--inhibit", action="store_true", help="启用侧抑制（默认关）")
    k.add_argument("--json", action="store_true", help="输出 JSON")
    k.set_defaults(fn=cmd_think)

    s = sub.add_parser("stats", help="查看规模/内存/权重/稀疏度/原型")
    s.add_argument("--brain", required=True)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_stats)

    e = sub.add_parser("eval", help="在数据集上评估准确率")
    e.add_argument("--brain", required=True, help="训练后的 .npz")
    e.add_argument("--data", required=True, help="带标签的数据集（TSV/JSONL）")
    e.add_argument("--top-k", type=int, default=4)
    e.add_argument("--ticks", type=int, default=6)
    e.add_argument("--salt", type=int, default=0)
    e.add_argument("--inhibit", action="store_true", help="启用侧抑制（默认关）")
    e.add_argument("--json", action="store_true")
    e.set_defaults(fn=cmd_eval)

    ar = sub.add_parser("arch", help="类脑结构演示（分级/门控/侧抑制/DMN/睡眠）")
    ar.add_argument("--brain", required=True)
    ar.add_argument("--text", default="今天天气怎么样，会下雨吗", help="门控演示输入")
    ar.add_argument("--data", help="可选：评估集，顺带演示睡眠前后准确率")
    ar.add_argument("--out", help="arch 演示睡眠后另存路径（默认不保存）")
    ar.add_argument("--epochs", type=int, default=1, help="睡眠重放轮数（默认 1）")
    ar.add_argument("--ticks", type=int, default=6)
    ar.add_argument("--top-k", type=int, default=4)
    ar.set_defaults(fn=cmd_arch)

    sl = sub.add_parser("sleep", help="睡眠重放（离线再巩固）")
    sl.add_argument("--brain", required=True)
    sl.add_argument("--data", help="可选：评估集，打印睡眠前后准确率")
    sl.add_argument("--epochs", type=int, default=1)
    sl.add_argument("--ticks", type=int, default=4)
    sl.add_argument("--strength", type=int, default=70, help="重放输入强度（默认 70）")
    sl.add_argument("--top-k", type=int, default=4)
    sl.add_argument("--out", help="另存路径（默认覆盖 --brain）")
    sl.set_defaults(fn=cmd_sleep)

    pr = sub.add_parser("prune", help="突触修剪（弱连接置 0）")
    pr.add_argument("--brain", required=True)
    pr.add_argument("--threshold", type=int, default=0, help="|w| <= 阈值 被剪（默认 0）")
    pr.add_argument("--data", help="可选：评估集，打印修剪前后准确率")
    pr.add_argument("--top-k", type=int, default=4)
    pr.add_argument("--ticks", type=int, default=6)
    pr.add_argument("--out", help="另存路径（默认覆盖 --brain）")
    pr.set_defaults(fn=cmd_prune)

    hm = sub.add_parser("homeostasis", help="突触稳态（下调平均 |w|）")
    hm.add_argument("--brain", required=True)
    hm.add_argument("--target", type=int, default=2, help="目标非零 |w| 均值（默认 2）")
    hm.add_argument("--out", help="另存路径（默认覆盖 --brain）")
    hm.set_defaults(fn=cmd_homeostasis)

    rp = sub.add_parser("rplus", help="R+ 简化版：与默认路径一致性自检")
    rp.add_argument("--brain", required=True)
    rp.add_argument("--text", help="只测这一条输入（默认测 4 条）")
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(fn=cmd_rplus)

    sub.add_parser("selftest", help="内置自检（不读写文件）").set_defaults(fn=cmd_selftest)
    sub.add_parser("verify", help="量化 vs float 一致性验证").set_defaults(fn=cmd_verify)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
