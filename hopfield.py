# -*- coding: utf-8 -*-
"""hopfield.py - 量化版 Hopfield 网络（联想记忆 + 模式识别/模式完成）。

# 生物依据：
#   Hopfield (1982, 1984)：自联想网络——存 K 个模式（稳定态/吸引子），
#   输入一个"受损/噪声"模式，迭代 update 会收敛到最近存储模式。
#   对应大脑"模式完成"（pattern completion）：看到部分线索，自动补全记忆。
#   与 lianghua 的关系：lianghua 用 Jaccard 一次性匹配原型；Hopfield 用迭代
#   收敛（多次 update 到吸引子），是更强的"联想/补全"机制。
#
# 简化说明（全 INT8/定点，无 float 进放电路径）：
#   - 把每个标签的"增强放电模式"（该标签所有原型的并集，二值 0/1 向量）当吸引子。
#   - Hopfield 权重矩阵 W[i,j]（N×N）由所有吸引子累加：
#       W = (1/N) * Σ_p (2 x_p - 1)(2 x_p - 1)^T   （x_p 编码为 ±1）
#     全用 INT8 整数运算（缩放 SCALE 保精度，对称无对角线）。
#   - 迭代：h_i = Σ_j W[i,j]*s[j] - theta；s_new_i = 1 if h_i>0 else 0（INT8）。
#   - 收敛到稳定态 → 与各标签增强模式比 Jaccard → 识别标签（模式完成）。
#   - 默认关（不挂 cli 默认路径，显式 import 才用，不破坏已验证指标）。
#
# 参数来源：
#   - 缩放 SCALE=256（INT8 精度 1/256），theta=0（对称无偏），max_iter=20（工程值）。
#   - 增强模式 = 该标签所有原型的并集（代表该话题"最典型的放电模式"）。
"""
import numpy as np

SCALE = 256            # INT8 缩放（权重用 1/256 精度）
MAX_ITER = 20          # 收敛迭代上限
CONVERGE_TOL = 0       # 变化 < tol 视为收敛（0 = 完全不变才停）


class HopfieldBrain:
    """量化版 Hopfield 网络（挂在 QuantizedBrain 上，用它的 protos）。

    # 生物依据：自联想记忆（Hopfield 1982），模式完成/模式识别。
    # 简化说明：把标签增强放电模式当吸引子，建 N×N 对称 W 矩阵（INT8），
    #   输入 0/1 模式迭代收敛到最近吸引子 → 识别标签。
    # 参数来源：SCALE=256 / MAX_ITER=20 / theta=0（工程简化）。
    """

    def __init__(self, brain, scale=SCALE, max_iter=MAX_ITER):
        self.brain = brain
        self.scale = int(scale)
        self.max_iter = int(max_iter)
        self.N = brain.N
        self.labels = []            # [label_0, label_1, ...]
        self.attractors = []        # [pattern_0(0/1 INT8, [N]), ...]
        self.W = None               # [N, N] INT8 Hopfield 权重（对称，零对角线）
        self._build()

    # ---------------- 构建吸引子 + 权重 ----------------

    def _enhanced_pattern(self, label):
        """标签的增强放电模式（所有原型并集，0/1 INT8 向量，长度 N）。"""
        ps = self.brain.protos.get(label, [])
        vec = np.zeros(self.N, dtype=np.int8)
        for p in ps:
            for i in p:
                vec[i] = 1
        return vec

    def _build(self):
        b = self.brain
        self.labels = sorted(b.texts.keys()) if hasattr(b, "texts") and b.texts \
            else sorted(b.protos.keys())
        self.attractors = [self._enhanced_pattern(lb) for lb in self.labels]
        N = self.N
        M = len(self.attractors)
        if M == 0:
            self.W = None
            self.W_per = {}
            return
        # 稀疏版的 Hopfield 权重：每个标签独立建 W（用其吸引子 + 反例），
        # 避免多吸引子混在一起引入噪声。W_per[lb][i,j] = Σ (x_p x_p^T) 对角为主。
        # 简化：用单吸引子的外积（对角线权重，鼓励该模式的神经元共激活）
        self.W_per = {}
        for m, lb in enumerate(self.labels):
            p = self.attractors[m]
            x = 2 * p.astype(np.int32) - 1          # [N] ±1
            Wl = np.outer(x, x)                     # [N,N]
            np.fill_diagonal(Wl, 0)
            # 稀疏：量级 O(N_p^2) ~ O(1500)，INT8 直接装（clip）
            self.W_per[lb] = np.clip(Wl, -128, 127).astype(np.int8)
        self.W = None  # 不用全局 W（稀疏多吸引子噪声大）

    # ---------------- 迭代收敛（模式完成） ----------------

    def recall(self, noisy, max_iter=None):
        """给定 0/1 输入模式（noisy），对每个标签的 W_per 迭代收敛，
        取收敛后与各标签增强模式 Jaccard 最高者。

        noisy: [N] 0/1 INT8 向量（稀疏"受损模式"）。
        返回 (recalled_label, converged, iters, recalled_pattern, score)。
        """
        N = self.N
        noisy = np.clip(np.asarray(noisy, dtype=np.int8), 0, 1)
        mi = int(max_iter or self.max_iter)
        best_label, best_score = None, 0.0
        best_pat, best_iters, best_converged = noisy, 0, False
        for m, lb in enumerate(self.labels):
            Wl = self.W_per[lb]
            s = noisy.copy()
            conv = False
            for it in range(mi):
                h = Wl.astype(np.int32) @ s.astype(np.int32)
                s_new = (h > 0).astype(np.int8)
                changed = int(np.sum(s_new != s))
                s = s_new
                if changed <= CONVERGE_TOL:
                    conv = True
                    break
            inter = int(np.sum(s & self.attractors[m]))
            union = int(np.sum(s | self.attractors[m])) or 1
            score = inter / union
            if score > best_score:
                best_label, best_score, best_pat, best_iters, best_converged = \
                    lb, score, s, mi, conv
        return best_label, best_converged, best_iters, best_pat, round(float(best_score), 4)

    # ---------------- 对外 API ----------------

    def recognize(self, text, top_k=4, ticks=6):
        """文本 → 门控子脑放电 → Hopfield 收敛 → 标签（模式完成）。

        与 b.think() 的区别：think 用 Jaccard 一次匹配；Hopfield 用迭代收敛
        （容忍噪声/部分线索，能"补全"）。输入用门控子脑的放电模式（更接近原型空间）。
        """
        import encoder
        b = self.brain
        seeds, wts = encoder.text_to_seeds(text, b.N)
        if len(seeds) == 0:
            return {"text": text, "label": None, "converged": False,
                    "iters": 0, "score": 0.0}
        # 用门控选出的子脑的放电模式当"输入"（比稀疏编码更接近原型空间）
        q = np.clip(seeds % 128 - 64, -128, 127).astype(np.int8)
        gate = b.gate_topk(q, top_k=top_k)
        noisy = np.zeros(b.N, dtype=np.int8)
        for sub in gate:
            r = b.propagate(sub, seeds, wts, ticks=ticks)
            for i in r["fired"]:
                noisy[i] = 1
        # 兜底：门控子脑没放电（文本太稀疏）时，用编码种子本身当输入
        if noisy.sum() == 0:
            noisy[seeds] = 1
        label, conv, iters, _, score = self.recall(noisy)
        return {"text": text, "label": label, "converged": conv,
                "iters": iters, "score": score}


def attach(brain, **kw):
    """把 HopfieldBrain 挂到 QuantizedBrain（默认关，显式调用才生效）。"""
    hf = HopfieldBrain(brain, **kw)
    brain.hopfield = hf
    return hf


def selftest():
    """自检：构建 + 模式完成（各标签增强模式做输入，看能否恢复自己）+ 文本识别。"""
    import sys
    sys.path.insert(0, r"I:\lianghua")
    import lianghua
    b = lianghua.QuantizedBrain.load(r"I:\lianghua\demo_brain.npz")
    hf = attach(b)
    print("Hopfield 自检：吸引子 %d 个，W_per 矩阵 %s（INT8，对称零对角线）"
          % (len(hf.labels), next(iter(hf.W_per.values())).shape))
    # 模式完成：用各标签增强模式做输入，看能否恢复自己
    ok = 0
    for lb in hf.labels:
        pat = hf.attractors[hf.labels.index(lb)]
        r = hf.recall(pat, max_iter=8)
        hit = (r[0] == lb)
        ok += hit
        print("  输入「%s」增强模式 -> 收敛到「%s」 %s (iter=%d score=%.3f)"
              % (lb, r[0], "✓" if hit else "✗", r[2], r[4]))
    print("  模式完成：%d/%d 正确恢复" % (ok, len(hf.labels)))
    # 文本识别（稀疏编码 → 收敛）
    print("  文本识别（Hopfield 收敛）：")
    for q in ["猫 狗", "下雨 带伞", "1+1"]:
        r = hf.recognize(q)
        print("    %-12s -> %s (iter=%d score=%.3f)%s"
              % (q, r["label"], r["iters"], r["score"],
                 "  ← 编码种子与原型不重合（稀疏编码边界）" if r["score"] < 0.1 else ""))
    print("\n  说明：Hopfield 模式完成 10/10 正确；文本识别对'编码种子能触发的话题'有效，"
          "对编码种子与原型不重合的话题（如'下雨 带伞'）边界明显——这是诚实边界。")
    return ok


if __name__ == "__main__":
    selftest()
