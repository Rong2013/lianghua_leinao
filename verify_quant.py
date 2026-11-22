# -*- coding: utf-8 -*-
"""验证 lianghua 量化版 vs float 版：同一输入，量化与 float 放电行为一致性。

关键：两版膜电位都用 [0,1] 物理尺度。
  - float 版：state 直接存 [0,1] float32
  - 量化版：state 存 INT8（[0,100] = [0,1]×100），fire 内部用 int32 还原 [0,1] 比较
验证：喂同一批 wts（[0,1] float，量化版 ×100 取整），看放电集合重合率。
"""
import sys
import numpy as np
sys.path.insert(0, r"I:\lianghua")
import lianghua


class FloatBrain:
    NSUB, N, K = 8, 2048, 32
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.state = np.zeros((self.NSUB, self.N), dtype=np.float32)
        self.weights = rng.integers(-2, 3, (self.NSUB, self.N, self.K), dtype=np.int32)
    def fire(self, sub, seeds, wts, tick):
        st = self.state[sub]
        seeds = np.asarray(seeds, dtype=np.int32).reshape(-1)
        s_sorted, inv = np.unique(seeds, return_inverse=True)
        input_sum = np.zeros(s_sorted.size, dtype=np.float32)
        np.add.at(input_sum, inv, np.asarray(wts, dtype=np.float32))  # wts 已是 [0,1]
        base = st[s_sorted]
        newv = np.clip(base * 0.9 + input_sum, 0, 1.0)
        st[s_sorted] = newv
        fired = np.flatnonzero(newv >= 0.5)
        fl = s_sorted[fired]
        if len(fl):
            st[fl] = 0.0
        return len(fl), fl


# 量化版：给 fire 喂 [0,100] 定点（= wts×100），内部 int32 还原
qb = lianghua.QuantizedBrain(seed=42)
fb = FloatBrain(seed=42)

rng = np.random.default_rng(7)
rounds, subs, nseed = 6, 3, 128
tot_jac, tot_q, tot_f = 0.0, 0, 0
for r in range(rounds):
    for sub in range(subs):
        s = rng.integers(0, 2048, nseed)
        w01 = rng.uniform(0.2, 0.9, nseed)          # [0,1] 输入
        wq = np.clip((w01 * 100).round(), 0, 127).astype(np.int8)   # 量化版输入（×100）
        fq, sync_q, flq = qb.fire(sub, s, wq, r)
        ff, flf = fb.fire(sub, s, w01, r)
        uq, uf = set(flq.tolist()), set(flf.tolist())
        jac = len(uq & uf) / (len(uq | uf) or 1)
        tot_jac += jac; tot_q += fq; tot_f += ff
        print("  轮%d 子脑%d  量化 %3d | float %3d | Jaccard %.2f" % (r, sub, fq, ff, jac))
avg = tot_jac / (rounds * subs)
print()
print("平均 Jaccard %.2f | 量化总放电 %d | float 总放电 %d | 比值 %.2f"
      % (avg, tot_q, tot_f, tot_q / max(1, tot_f)))
print()
if avg > 0.7:
    print("结论: PASS 量化与 float 高度一致，量化版可用")
elif avg > 0.4:
    print("结论: 基本一致（量化精度损失可接受，阈值附近抖动）")
else:
    print("结论: 偏差大，量化需调参（阈值/leak/输入取整）")
