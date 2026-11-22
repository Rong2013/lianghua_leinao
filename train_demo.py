# -*- coding: utf-8 -*-
"""train_demo.py - 训练演示脑（绕过 cli.py TSV 解析器，直接喂数据）。

# 生物依据：学习（STDP）暴露成显式训练流程。
# 简化说明：直接用 QuantizedBrain.train_samples，不经过 cli.py 的 load_samples。
#   训练数据 32 条（8 类 × 4），测试数据 16 条（8 类 × 2，与训练集不重叠）。
# 参数来源：epochs=4（比默认 2 多，让演示脑学得扎实），工程简化。
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lianghua
import visualizer


def parse_tsv(path):
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[1].strip():
                out.append((parts[0].strip(), parts[1].strip()))
    return out


def main():
    train_path = os.path.join(HERE, "demo_samples.tsv")
    test_path = os.path.join(HERE, "demo_samples_test.tsv")
    brain_path = os.path.join(HERE, "demo_brain.npz")
    raster_path = os.path.join(HERE, "viz", "demo_raster.png")

    train = parse_tsv(train_path)
    test = parse_tsv(test_path)
    print("训练集 %d 条（标签: %s）" % (len(train), sorted(set(l for l, _ in train))))
    print("测试集 %d 条（标签: %s）" % (len(test), sorted(set(l for l, _ in test))))

    # 编译脑
    b = lianghua.QuantizedBrain.build(nsub=8, n=2048, k=32, seed=20261120)
    print("已编译神经元：8 子脑 × 2048 神经元 = 16,384 神经元 / 524,288 突触")

    # 训练（epochs=4）
    b.train_samples(train, epochs=4, ticks=6, verbose=True)
    b.save(brain_path)
    print("已保存演示脑 → %s" % brain_path)

    # 测试集评估
    correct = 0
    scores = []
    errors = []
    for label, text in test:
        r = b.think(text, top_k=4, ticks=6)
        pred = r["label"]
        score = r["score"]
        scores.append(score)
        if pred == label:
            correct += 1
        else:
            errors.append({"true": label, "pred": pred, "score": round(score, 3), "text": text})
    acc = correct / len(test)
    print("\n=== 测试集评估（16 条，与训练集不重叠）===")
    print("正确 %d / %d，准确率 %.1f%%" % (correct, len(test), acc * 100))
    if errors:
        print("错例：")
        for e in errors:
            print("  真值=%-4s 预测=%-4s 分=%.3f  %s" % (e["true"], e["pred"], e["score"], e["text"]))

    # 栅格图（取前 6 条测试样本）
    texts = [t for _, t in test[:6]]
    import encoder as enc
    # 逐条 think 收集栅格（visualizer.raster_plot 接受 raster 矩阵，不是 brain）
    import numpy as np
    all_raster = []
    for _, text in test[:6]:
        r = b.think(text, top_k=4, ticks=6)
        # 从 per_sub 重建栅格（简化：用 fired 集合）
        sub = 0
        seeds, wts = enc.text_to_seeds(text, b.N)
        cur_s, cur_w = seeds, wts
        col_data = np.zeros(b.N, dtype=bool)
        for t in range(6):
            f, sync, fired = b.fire(sub, cur_s, cur_w, tick=t)
            if len(fired):
                col_data[fired] = True
            if f == 0:
                cur_s, cur_w = seeds, wts
            else:
                nb = b.neighbors[sub][fired].reshape(-1)
                ww = b.weights[sub][fired].reshape(-1)
                pos = ww > 0
                cur_s, cur_w = nb[pos], ww[pos]
                if len(cur_s) == 0:
                    cur_s, cur_w = seeds, wts
        all_raster.append(col_data)
    raster_matrix = np.stack(all_raster, axis=1)
    visualizer.raster_plot(raster_matrix, out_path=raster_path, title="Demo Brain Raster",
                           ticks=6, texts=[t for _, t in test[:6]])
    print("栅格图 → %s" % raster_path)

    # 存评估结果（GUI 用）
    eval_path = os.path.join(HERE, "demo_eval.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": acc,
            "correct": correct,
            "total": len(test),
            "errors": errors,
            "avg_score": round(sum(scores) / len(scores), 3),
            "train_samples": train,
            "test_samples": test,
        }, f, ensure_ascii=False, indent=2)
    print("评估结果 → %s" % eval_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
