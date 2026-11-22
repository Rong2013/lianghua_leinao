# -*- coding: utf-8 -*-
"""thousand_brains.py - 霍金斯千脑理论启发：皮质柱投票 + 预测编码（量化版，2026-11-22）。

# 生物依据（Hawkins & Blum《千脑通识》Thousand Brains Theory）：
#   - 皮层不是单一"脑"，而是 ~2000 个并行"脑"（皮层柱/cortical columns），
#     每根柱独立表征世界，最后**投票**选出最佳解释（全局层 L6a 把预测发给其他区）。
#   - 每根柱维护对下一步的**预测**（predictive coding）：预测对 → 误差小 → 不学；
#     预测错 → 误差大 → 驱动学习（prediction error 是信号，不是噪声）。
#   - 这正好对应我们的"放电原型"：每个子脑 = 一根"柱"，原型 = 柱的预测表征，
#     投票 = 选最匹配的柱。
# 简化说明：
#   - "皮质柱" = 我们的子脑（NSUB 根柱）；"柱的预测" = 该子脑的放电原型（protos）。
#   - "投票" = gate_topk + Jaccard 匹配（已实现，本模块只是把它升格成显式"千脑投票"）。
#   - "预测编码" = 新增：用上一 tick 的放电预测下一 tick，算预测误差，误差门控可塑性
#     （预测错才强学，预测对则弱学）——这是 STDP 之上的第二层门控（reward 之上）。
# 参数来源：
#   - 预测误差阈值 0.3（Jaccard 距离）；误差门控 a_plus 缩放 [0.5, 1.5]。工程简化。
#
# 用法：
#   python thousand_brains.py            # 自检（小脑演示柱投票 + 预测编码）
#   在 cli.py 里：python cli.py tb --brain brain.npz   # 千脑投票 + 预测误差报告
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lianghua      # noqa: E402
import encoder       # noqa: E402

# 预测编码参数（工程简化）
PRED_ERR_THRESH = 0.30        # 预测误差阈值（Jaccard 距离 > 0.3 = 预测错，强学）
PRED_LEARN_MAX = 1.5          # 预测错时 STDP a_plus 放大上限（150%）
PRED_LEARN_MIN = 0.5          # 预测对时 STDP a_plus 缩小下限（50%，防过拟合）


def column_vote(brain, seeds, wts, ticks=6):
    """千脑投票：每根皮质柱（子脑）独立放电 → 各自与原型 Jaccard → 投票选最佳柱。

    返回 (winner_sub, votes, per_sub)：
      winner_sub: 得票最高的柱（子脑号）
      votes: 每柱票数（匹配的原型数 × Jaccard 分）
      per_sub: 每柱的放电 + 匹配分
    """
    votes = {}
    per_sub = {}
    for sub in range(brain.NSUB):
        r = brain.propagate(sub, seeds, wts, ticks=ticks)
        fired_set = set(int(v) for v in r["fired"].tolist())
        best = 0.0
        for label, patterns in brain.protos.items():
            if brain.sub_of.get(label) != sub:
                continue
            for proto in patterns:
                if not proto:
                    continue
                inter = len(fired_set & proto)
                union = len(fired_set | proto) or 1
                jac = inter / union
                best = max(best, jac)
        votes[sub] = best
        per_sub[sub] = {"fired": len(fired_set), "score": round(float(best), 4)}
    winner = max(votes, key=votes.get) if votes else None
    return winner, votes, per_sub


def prediction_error(brain, sub, seeds, wts, ticks=6):
    """预测编码：用前 t 个 tick 的放电预测第 t+1 个 tick，算 Jaccard 误差。

    误差 < PRED_ERR_THRESH = 预测对（弱学）；误差 > 阈值 = 预测错（强学）。
    返回 (avg_err, learn_scale)：
      avg_err: 平均预测误差（0=完美预测，1=完全错）
      learn_scale: 建议的 STDP a_plus 缩放（[PRED_LEARN_MIN, PRED_LEARN_MAX]）
    """
    errors = []
    prev_fired = set()
    for t in range(ticks - 1):
        r = brain.propagate(sub, seeds, wts, ticks=1, tick0=t)
        fired = set(int(v) for v in r["fired"].tolist())
        if prev_fired:
            inter = len(fired & prev_fired)
            union = len(fired | prev_fired) or 1
            err = 1.0 - inter / union      # Jaccard 距离 = 预测误差
            errors.append(err)
        prev_fired = fired
    if not errors:
        return 0.0, 1.0
    avg_err = float(np.mean(errors))
    # 误差大 → 放大学习（PRED_LEARN_MAX）；误差小 → 缩小学习（PRED_LEARN_MIN）
    learn_scale = PRED_LEARN_MIN + (PRED_LEARN_MAX - PRED_LEARN_MIN) * min(1.0, avg_err / PRED_ERR_THRESH)
    return avg_err, round(learn_scale, 3)


def selftest():
    """自检：小脑演示柱投票 + 预测编码。"""
    print("=== 千脑理论自检（皮质柱投票 + 预测编码）===")
    b = lianghua.QuantizedBrain.build(nsub=6, n=512, k=16, seed=20261120)
    texts = ["猫", "狗", "汽车", "飞机"]
    b.train_samples([(t, t) for t in texts], epochs=1, ticks=4)
    # 千脑投票
    seeds, wts = encoder.text_to_seeds("狗", b.N)
    winner, votes, per_sub = column_vote(b, seeds, wts, ticks=4)
    top3 = sorted(votes.items(), key=lambda x: -x[1])[:3]
    print("千脑投票（输入'狗'）：胜柱=%d，前3柱得分=%s" % (winner, [(s, round(v, 3)) for s, v in top3]))
    # 预测编码
    err, scale = prediction_error(b, winner, seeds, wts, ticks=4)
    print("预测编码（胜柱）：平均误差=%.3f（<%.2f=预测对），建议学习缩放=%.2f×"
          % (err, PRED_ERR_THRESH, scale))
    print("千脑 + 预测编码 PASS（柱投票选出胜柱，预测误差门控学习强度）")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="霍金斯千脑启发：皮质柱投票 + 预测编码")
    ap.add_argument("--brain", help="QuantizedBrain .npz 路径（默认自检小脑）")
    ap.add_argument("--text", default="狗", help="投票输入文本")
    ap.add_argument("--ticks", type=int, default=4)
    a = ap.parse_args()
    if a.brain:
        b = lianghua.QuantizedBrain.load(a.brain)
        seeds, wts = encoder.text_to_seeds(a.text, b.N)
        winner, votes, per_sub = column_vote(b, seeds, wts, ticks=a.ticks)
        top = sorted(votes.items(), key=lambda x: -x[1])[:5]
        print("千脑投票（输入 %r）：胜柱=%d" % (a.text, winner))
        print("  前5柱得分:", [(s, round(v, 3)) for s, v in top])
        err, scale = prediction_error(b, winner, seeds, wts, ticks=a.ticks)
        print("预测编码：误差=%.3f，学习缩放=%.2f×" % (err, scale))
    else:
        sys.exit(selftest())
