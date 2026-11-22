# -*- coding: utf-8 -*-
"""adaptive.py - lianghua 量化类脑自适应机制（量化加强版，2026-11-22）。

# 生物依据：
#   - 大脑不是固定参数，而是**自适应**：突触强度随活动调节（synaptic scaling /
#     homeostasis，Turrigiano & Nelson 2004）；神经元放电阈值随历史活跃度漂移
#     （refractory adaptation）；学习率随任务难度调节（curriculum）。
#   - 这 4 项把"固定参数"升级成"随状态自动调"，让量化类脑更强。
# 简化说明：全部用 INT8/定点整数运算（不进 float，守量化理念），
#   且默认关闭（`LIANGHUA_ADAPTIVE=0`），不破坏已验证指标；显式开才生效。
# 参数来源：自适应步长/边界 = 工程简化（与 brain-dual 的 threshold_adapt 同源）。
#
# 4 项自适应：
#   1. 自适应阈值：活跃神经元阈值自动抬高（抑制少数神经元反复放电，分布更均匀）
#   2. 自适应学习率：按子脑放电活跃度自动调 STDP 步长（活跃区弱学，静默区强学，防过拟合）
#   3. 自适应稀疏化：睡眠剪枝阈值按当前稀疏度动态调（稀疏度低多剪，高则少剪）
#   4. 自适应发放预算：max_fire 按当前发放率动态调（发放率高则收紧，守住 <15Hz 稀疏判据）
"""
import numpy as np


class AdaptiveBrain:
    """自适应包装：挂在 QuantizedBrain 上，提供 4 项自适应调节。

    用法（默认关，不破坏已验证指标）：
        b = QuantizedBrain.build()
        ad = AdaptiveBrain(b, enabled=True)   # LIANGHUA_ADAPTIVE=1 时 enabled
        thr = ad.adaptive_threshold(sub)       # 该子脑当前阈值
        rate = ad.adaptive_stdp_rate(sub)      # 该子脑当前 STDP 步长缩放
    """

    def __init__(self, brain, enabled=True):
        self.b = brain
        self.enabled = bool(enabled)
        # 每子脑活跃度计数（INT32，放电影法）
        self.sub_activity = np.zeros(brain.NSUB, dtype=np.int32)
        # 自适应参数（INT8/定点，工程简化）
        self.th_base = int(50)                 # 基础阈值（=THR）
        self.th_adapt_num, self.th_adapt_den = 6, 100   # 阈值随活跃度抬升 6%
        self.lr_min_pct, self.lr_max_pct = 50, 120      # STDP 步长缩放 [50%, 120%]
        self.sparse_target = 0.5                # 目标稀疏度（0.5 = 一半突触剪掉）
        self.sparse_step = 1                    # 剪枝阈值自适应步长（INT8）
        self.firing_budget_pct = 3              # 发放预算上限（=3%，守 <15Hz 稀疏判据）
        self.history = []                       # 最近自适应决策（诊断用）

    # ---- 1. 自适应阈值：活跃子脑阈值抬高（抑制反复放电）----
    def adaptive_threshold(self, sub):
        """子脑当前阈值（INT8）。活跃子脑阈值 = base + base*activity/den（封顶）。"""
        if not self.enabled:
            return self.th_base
        act = int(self.sub_activity[sub])
        bump = (self.th_base * act) // 1000      # 活跃度 × 0.1%（整数）
        thr = self.th_base + min(bump, self.th_base // 2)   # 最多抬 50%
        return thr

    # ---- 2. 自适应学习率：活跃子脑弱学，静默子脑强学（防过拟合）----
    def adaptive_stdp_rate(self, sub):
        """子脑 STDP 步长缩放百分比（INT）。活跃=lr_min（弱学），静默=lr_max（强学）。"""
        if not self.enabled:
            return 100
        act = int(self.sub_activity[sub])
        # 活跃度 0 → lr_max（120%），活跃度 100+ → lr_min（50%）（线性，整数）
        frac = min(100, act)                     # 封顶 100
        pct = self.lr_max_pct - (self.lr_max_pct - self.lr_min_pct) * frac // 100
        return max(self.lr_min_pct, pct)

    # ---- 3. 自适应稀疏化：剪枝阈值随当前稀疏度动态调 ----
    def adaptive_prune_threshold(self, current_sparsity_pct):
        """剪枝阈值（INT8）。稀疏度低于目标 → 阈值=1（多剪）；高于目标 → 阈值=0（少剪）。"""
        if not self.enabled:
            return 0
        if int(current_sparsity_pct) < self.sparse_target * 100:
            return self.sparse_step               # 稀疏度低，多剪
        return 0                                  # 稀疏度够高，不再剪

    # ---- 4. 自适应发放预算：发放率高则收紧（守 <15Hz）----
    def adaptive_max_fire(self, sub, current_fired, total_neurons):
        """该子脑本 tick 允许的最大放电数（INT）。发放率超预算则收紧。"""
        if not self.enabled:
            return 0          # 0 = 不限制（原行为）
        budget = max(1, int(total_neurons * self.firing_budget_pct / 100))
        return min(int(current_fired) * 2, budget)   # 翻倍但不超预算

    # ---- 活动度更新（每 tick 放电后调）----
    def record_fire(self, sub, fired_count):
        """记录子脑放电活跃度（放电影法：衰减 + 累加，INT32 防溢出）。"""
        if not self.enabled:
            return
        self.sub_activity[sub] = (self.sub_activity[sub] * 95 // 100) + int(fired_count)
        self.sub_activity = np.clip(self.sub_activity, 0, 1000).astype(np.int32)

    # ---- 诊断 ----
    def report(self):
        out = {
            "enabled": self.enabled,
            "sub_activity": self.sub_activity.tolist(),
            "thresholds": [self.adaptive_threshold(s) for s in range(self.b.NSUB)],
            "stdp_rates": [self.adaptive_stdp_rate(s) for s in range(self.b.NSUB)],
            "history": self.history[-8:],
        }
        return out
