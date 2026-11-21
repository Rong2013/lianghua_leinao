# -*- coding: utf-8 -*-
"""lianghua - 量化版类脑骨架（开源，路径 I:/lianghua）。

# 生物依据：
#   - 脉冲是"1-bit 事件"（SNN 物理基础），天然量化：膜电位用 INT8，
#     阈值用定点（0.5 阈值 × 100 = 50），权重 INT8。
#   - 突触稳态用 INT8 截断（±127），与 brain-dual 的 STDP 一致。
#   - 邻居拓扑用"局部为主 + 少量长程"（皮层局部连接 + 长程投射）。
# 简化说明：lianghua 把 CHN2 的"皮层分级+丘脑门控+DMN"做成**纯量化**版本，
#   全部张量 INT8/定点，无 float，可移植到 MCU/边缘（比 CHN2 轻 4×）。
#   开源骨架默认 8 个子脑 × 2048 神经元，规模可调（见 cli.py build）。
# 量化方案（固定，不用调）：
#   - 膜电位 state：INT8，范围 [0,100] 表示 [0.0,1.0]，leak 0.9 → ×90/100
#   - 突触权重 weights：INT8 [-127,127]（沿用 brain-dual STDP 截断）
#   - 放电阈值：定点 50（=0.5）
#   - 邻居 neighbors：INT32（神经元索引，不量化）
#   - 门控相关性：INT8 点积 / 255
# 全部在 1 字节/元素（除邻居 INT32），内存是 float32 版的 1/4。
#
# 跨平台：纯 Python + numpy（INT8 数组），mac/linux/windows 通吃。
# 稳定：量化溢出用 np.clip 兜底，阈值非有限回退 50。
#
# 用法：
#   自检            python lianghua.py
#   编译神经元      python cli.py build   --out brain.npz
#   训练            python cli.py train   --brain brain.npz --data samples.tsv
#   推理            python cli.py think   --brain brain.npz --text "你好"
#   查看统计        python cli.py stats   --brain brain.npz
#
# ============================ 类脑架构 ============================
#
#   文本/字节
#      │  稀疏编码（encoder.py：1~3-gram → 少量神经元）
#      ▼
#   丘脑中继 gate_topk（Moruzzi & Magoun 1949：注意的瓶颈）
#      │
#      ▼
#   感觉层 L1 (0..2)  →  联合层 L2 (3..4)  →  决策层 L3 (5..7)
#      │  Hubel & Wiesel 层级：简单特征 → 组合 → 决策
#      │
#      ├── 侧抑制（E/I 平衡：放电多的子脑压低其他子脑）
#      ├── STDP（赫布可塑性）+ 神经调质（reward 放大 a_plus）
#      └── 放电模式 → 与记忆原型做 Jaccard 匹配
#      ▲
#   离线巩固：稳态 homeostasis / 突触修剪 prune / 睡眠重放 sleep_replay
#   DMN（Raichle 2001）：无外部输入时决策层自发活动 dmn_idle
#
# 所有量仍是 INT8/定点：侧抑制用 INT16 中间量、重放用 INT8 输入，
# 没有任何 float 进入放电路径。
# =================================================================
"""
import json

import numpy as np

import encoder

# 量化常数（定点约定）
SCALE = 100          # 膜电位 [0,100] ↔ [0.0,1.0]
THR = 50             # 放电阈值 0.5
LEAK_NUM, LEAK_DEN = 90, 100   # leak 0.9
W_MIN, W_MAX = -127, 127      # 突触权重 INT8 截断

# STDP 参数（INT8 整数步长）
A_PLUS = 6           # 前→后（因果）增强
A_MINUS = 2          # 后→前 抑制（弱，保持稳定）
LOCAL_RATIO = 0.7    # 邻居里 70% 是局部连接


class QuantizedBrain:
    """量化类脑：默认 8 子脑 × 2048 神经元，全 INT8/定点，可移植、可训练。"""

    NSUB = 8
    N = 2048
    K = 32

    def __init__(self, seed=20261120, nsub=None, n=None, k=None):
        self.NSUB = int(nsub or self.NSUB)
        self.N = int(n or self.N)
        self.K = int(k or self.K)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)

        self.state = np.zeros((self.NSUB, self.N), dtype=np.int8)        # 膜电位 INT8
        self.weights = rng.integers(-2, 3, (self.NSUB, self.N, self.K), dtype=np.int8)
        self.neighbors = self._make_neighbors(rng)
        self.last_spike = np.full((self.NSUB, self.N), -1, dtype=np.int32)

        # 训练产物（量化类脑的"记忆"：标签 → 放电原型）
        self.protos = {}     # label -> set(int)
        self.sub_of = {}     # label -> sub

    # ---------------- 编译神经元（拓扑） ----------------

    def _make_neighbors(self, rng):
        """局部为主 + 少量长程：70% 环形近邻（±1..8），30% 随机长程。"""
        nb = np.empty((self.NSUB, self.N, self.K), dtype=np.int32)
        base = np.repeat(np.arange(self.N, dtype=np.int32)[:, None], self.K, axis=1)
        for s in range(self.NSUB):
            local = rng.integers(1, 9, size=(self.N, self.K), dtype=np.int32)
            sign = rng.choice(np.array([-1, 1], dtype=np.int32), size=(self.N, self.K))
            cand = (base + sign * local) % self.N
            longj = rng.integers(0, self.N, size=(self.N, self.K), dtype=np.int32)
            mask = rng.random((self.N, self.K)) < (1.0 - LOCAL_RATIO)
            nb[s] = np.where(mask, longj, cand)
        return nb

    @classmethod
    def build(cls, nsub=8, n=2048, k=32, seed=20261120):
        """编译神经元：分配膜电位/权重/邻居表。产物可用 save() 落盘。"""
        return cls(seed=seed, nsub=nsub, n=n, k=k)

    # ---------------- 放电（原逻辑，未改） ----------------

    def fire(self, sub, seeds, wts, tick, inh_gain=0):
        """INT8 量化放电：膜电位累积 → 过阈 THR → 放电 → 复位。
        全 INT8 算术（无 float），溢出用 clip 兜底。"""
        st = self.state[sub]
        seeds = np.asarray(seeds, dtype=np.int32).reshape(-1)
        if len(seeds) == 0:
            return 0, 0.0, np.array([], dtype=np.int32)
        s_sorted, inv = np.unique(seeds, return_inverse=True)
        input_sum = np.zeros(s_sorted.size, dtype=np.int32)
        np.add.at(input_sum, inv, np.asarray(wts, dtype=np.int32))
        base = st[s_sorted].astype(np.int32)
        # 膜电位累积（INT32 中间量防溢出）：新 = 旧*leak + 输入
        # 旧=0 时 new=input（40）；旧=40 时 new=40*0.9+40=76 过阈 50
        newv = (base * LEAK_NUM) // LEAK_DEN + input_sum
        newv = np.clip(newv, 0, SCALE).astype(np.int8)
        self.state[sub][s_sorted] = newv
        fired = np.flatnonzero(newv >= THR)
        fired_global = s_sorted[fired]
        if len(fired_global):
            self.state[sub][fired_global] = 0          # 复位
            self.last_spike[sub][fired_global] = tick
        sync = len(fired_global) / max(1, len(seeds))
        return len(fired_global), sync, fired_global

    def gate_topk(self, q_vec_int8, top_k=4):
        """丘脑门控：INT8 相关性（点积 / 255），放行 top_k。"""
        q = np.clip(np.asarray(q_vec_int8, dtype=np.int8), -128, 127)
        scored = []
        for s in range(self.NSUB):
            # 子脑"指纹"用前段膜电位（INT8→INT32）；q 对齐到指纹长度
            fp = self.state[s, :self.N // 8].astype(np.int32)
            m = min(len(q), len(fp))
            rel = int(np.dot(q[:m], fp[:m])) / 255.0 / max(1, m)
            scored.append((s, rel))
        scored.sort(key=lambda x: -x[1])
        return [s for s, _ in scored[:top_k]]

    # ---------------- 训练（INT8 STDP） ----------------

    def stdp_update(self, sub, fired, prev_fired=None, a_plus=A_PLUS, a_minus=A_MINUS):
        """赫布式 STDP：放电神经元之间的连接加强（共激活），其余邻居减弱。
        prev_fired 非空时把它也算进"共激活"集合（跨 tick 因果）。
        返回加强的突触数（用于观察学习是否发生）。"""
        if len(fired) == 0:
            return 0
        active = np.zeros(self.N, dtype=bool)
        active[fired] = True
        if prev_fired is not None and len(prev_fired):
            active[prev_fired] = True
        nb = self.neighbors[sub][fired]                       # [F,K]
        hit = active[nb]                                      # [F,K]
        delta = np.where(hit, a_plus, -a_minus).astype(np.int16)
        w = self.weights[sub][fired].astype(np.int16) + delta
        self.weights[sub][fired] = np.clip(w, W_MIN, W_MAX).astype(np.int8)
        return int(hit.sum())

    def propagate(self, sub, seeds, wts, ticks=6, tick0=0):
        """多 tick 传播：外部输入打进去 → 放电结果沿权重传给下一 tick。
        放电为空时重新注入外部输入（保证能量不灭）。"""
        seeds = np.asarray(seeds, dtype=np.int32).reshape(-1)
        wts = np.asarray(wts, dtype=np.int8).reshape(-1)
        counts, syncs, fired_all = [], [], []
        cur_s, cur_w = seeds, wts
        for t in range(ticks):
            f, sync, fired = self.fire(sub, cur_s, cur_w, tick0 + t)
            counts.append(int(f))
            syncs.append(float(sync))
            fired_all.append(fired)
            if f == 0:
                cur_s, cur_w = seeds, wts
            else:
                nb = self.neighbors[sub][fired].reshape(-1)
                w = self.weights[sub][fired].reshape(-1)
                pos = w > 0
                cur_s, cur_w = nb[pos], w[pos]
                if len(cur_s) == 0:
                    cur_s, cur_w = seeds, wts
        ne = [x for x in fired_all if len(x)]
        total = np.unique(np.concatenate(ne)) if ne else np.zeros(0, dtype=np.int32)
        return {"counts": counts, "sync": syncs, "fired": total}

    def _sub_for(self, label):
        """标签 → 子脑：让不同类别落在不同"皮层分区"（可复现）。"""
        return encoder._h(label, salt=self.seed) % self.NSUB

    # 生物依据：三因子可塑性——赫布共激活（前后同时放电）× 神经调质（多巴胺/奖赏）
    #   决定这条连接是否被写进长期记忆（Schultz 1998；Frémaux & Gerstner 2016）。
    # 简化说明：reward 是整数百分比（100 = 不调制），只缩放 STDP 的 a_plus；
    #   a_minus 不动（避免奖赏把整体权重推向负饱和）。没有价值函数、没有 TD 误差。
    # 参数来源：默认 100（等价于无调制），保持既有 81.2% 留出集指标不变；工程简化。
    def train_samples(self, samples, epochs=3, ticks=6, salt=0, reward=100, verbose=True):
        """训练：samples = [(label, text), ...]；INT8 STDP，产物是 protos/sub_of。

        reward: 奖赏调制百分比（100 = 不调制；>100 加强可塑性；<100 减弱）。
        """
        a_plus = max(1, (A_PLUS * int(reward)) // 100)
        log = []
        for ep in range(epochs):
            changed = 0
            for label, text in samples:
                sub = self._sub_for(label)
                seeds, wts = encoder.text_to_seeds(text, self.N, salt=salt)
                if len(seeds) == 0:
                    continue
                prev = np.zeros(0, dtype=np.int32)
                cur_s, cur_w = seeds, wts
                pat = []
                for t in range(ticks):
                    f, sync, fired = self.fire(sub, cur_s, cur_w, tick=ep * ticks + t)
                    changed += self.stdp_update(sub, fired, prev, a_plus=a_plus)
                    pat.append(fired)
                    if f:
                        prev = fired
                    if f == 0:
                        cur_s, cur_w = seeds, wts
                    else:
                        nb = self.neighbors[sub][fired].reshape(-1)
                        w = self.weights[sub][fired].reshape(-1)
                        pos = w > 0
                        cur_s, cur_w = nb[pos], w[pos]
                        if len(cur_s) == 0:
                            cur_s, cur_w = seeds, wts
                proto = set()
                for x in pat:
                    proto |= set(int(v) for v in x.tolist())
                self.protos.setdefault(label, []).append(proto)
                self.sub_of[label] = sub
            log.append({"epoch": ep + 1, "reinforced": changed,
                        "protos": len(self.protos)})
            if verbose:
                print("  第%d轮：加强突触 %d，原型 %d 个" % (ep + 1, changed, len(self.protos)),
                      flush=True)
        return log

    # ---------------- 推理 ----------------

    def think(self, text, top_k=4, ticks=6, salt=0, inhibit=False):
        """文本 → 放电 → 与训练原型比较，返回最可能的标签。
        为可靠起见在所有子脑上传播（门控结果仅作参考输出）。
        inhibit=True 时启用侧抑制（默认 False，不改变已验证指标）。"""
        seeds, wts = encoder.text_to_seeds(text, self.N, salt=salt)
        if len(seeds) == 0:
            return {"text": text, "spikes": 0, "label": None, "score": 0.0,
                    "subs": [], "top": [], "per_sub": {}, "inhibit": False}

        q = np.clip(seeds % 128 - 64, -128, 127).astype(np.int8)
        gate = self.gate_topk(q, top_k=top_k)

        best_label, best_score, best_fired = None, 0.0, np.zeros(0, dtype=np.int32)
        per_sub = {}
        cands = []          # (score, label, fired, sub) —— 供侧抑制重排
        for sub in range(self.NSUB):
            r = self.propagate(sub, seeds, wts, ticks=ticks)
            fired = set(int(v) for v in r["fired"].tolist())
            per_sub[sub] = {"counts": r["counts"], "sync": r["sync"], "spikes": len(fired)}
            for label, patterns in self.protos.items():
                if self.sub_of.get(label) != sub:
                    continue
                for proto in patterns:
                    if not proto:
                        continue
                    inter = len(fired & proto)
                    union = len(fired | proto) or 1
                    score = inter / union
                    cands.append((score, label, r["fired"], sub))
                    if score > best_score:
                        best_label, best_score, best_fired = label, score, r["fired"]

        gains = None
        if inhibit and cands:
            # 侧抑制：放电多的子脑压制其他子脑（E/I 平衡），按 INT 百分比重排
            gains = self.lateral_inhibition(
                {s: v["spikes"] for s, v in per_sub.items()})
            rescored = [(sc * gains.get(s, 100) / 100.0, lb, fr)
                        for sc, lb, fr, s in cands]
            top = max(rescored, key=lambda x: x[0])
            if top[0] > 0:
                best_label, best_score, best_fired = top[1], top[0], top[2]

        return {"text": text,
                "spikes": int(best_fired.size),
                "label": best_label,
                "score": round(float(best_score), 4),
                "subs": gate,
                "per_sub": per_sub,
                "inhibit": bool(inhibit),
                "gains": gains,
                "top": [int(v) for v in best_fired[:16]]}

    # ==================== 类脑结构（组织视图 + 可选机制） ====================
    # 以下方法的默认参数都不进入已验证的默认路径（train/think 默认值不变），
    # 只通过 cli 的 arch / sleep / prune / rplus 命令显式调用。

    # 生物依据：新皮层是分级的——初级感觉区 → 联合区 → 决策/运动区（Hubel & Wiesel 1962；
    #   Felleman & Van Essen 1991 的层级连接模型）。
    # 简化说明：不建真实解剖连接，只按比例把子脑编号切成 感觉 / 联合 / 决策 三层，
    #   作为组织视图和 DMN/侧抑制的作用范围；不改变默认计算路径。
    # 参数来源：3/8 感觉、1/4 联合、其余决策（8 子脑 → 3/2/3），工程简化。
    def hierarchy(self):
        """皮层分级：按比例划分 感觉 / 联合 / 决策 三层（只作组织视图）。"""
        n = self.NSUB
        sensory_n = max(1, (n * 3) // 8)
        assoc_n = max(1, n // 4)
        if sensory_n + assoc_n >= n:
            sensory_n = max(1, n // 3)
            assoc_n = max(1, (n - sensory_n) // 2)
        decision_n = n - sensory_n - assoc_n
        return {"sensory": list(range(0, sensory_n)),
                "association": list(range(sensory_n, sensory_n + assoc_n)),
                "decision": list(range(sensory_n + assoc_n, n))}

    def layer_of(self, sub):
        """子脑 → 层名（sensory / association / decision）。"""
        for name, subs in self.hierarchy().items():
            if int(sub) in subs:
                return name
        return "unknown"

    # 生物依据：丘脑是中继站，把感觉输入按相关性门控到皮层，形成注意瓶颈
    #   （Moruzzi & Magoun 1949 网状激活系统；Halassa & Kastner 2017 丘脑门控）。
    # 简化说明：与 gate_topk 同一套 INT8 相关性公式，但额外返回每个子脑的
    #   相关度和所属层级，供 arch 命令展示；不改动 gate_topk 本身。
    # 参数来源：相关性 = INT8 点积 / 255 / 长度（与 gate_topk 完全一致），工程简化。
    def thalamic_relay(self, q_vec_int8, top_k=4):
        """丘脑中继（只读诊断版）：返回放行子脑 + 相关度 + 层级。"""
        q = np.clip(np.asarray(q_vec_int8, dtype=np.int8), -128, 127)
        scored = []
        for s in range(self.NSUB):
            fp = self.state[s, :self.N // 8].astype(np.int32)
            m = min(len(q), len(fp))
            rel = int(np.dot(q[:m], fp[:m])) / 255.0 / max(1, m)
            scored.append((s, rel))
        scored.sort(key=lambda x: -x[1])
        subs = [s for s, _ in scored[:top_k]]
        return {"subs": subs,
                "layers": {s: self.layer_of(s) for s in subs},
                "rel": [(s, round(float(r), 6)) for s, r in scored]}

    # 生物依据：皮层兴奋/抑制平衡——一个集群强烈放电会通过抑制性中间神经元
    #   压低邻近集群（侧抑制），让表征变稀疏、竞争出赢家（Mountcastle 1957）。
    # 简化说明：按各子脑放电数排序，前 keep 名保持 100%，其余按 放电数/峰值
    #   线性压低（不低于 floor）；只输出整数百分比增益，不改权重、不改放电。
    # 参数来源：keep=2、floor=20 —— 工程简化，未做生理标定。
    def lateral_inhibition(self, activations, keep=2, floor=20):
        """侧抑制：返回每个子脑的增益百分比（100 = 不抑制）。"""
        if not activations:
            return {}
        items = sorted(((int(s), int(a)) for s, a in activations.items()),
                       key=lambda x: -x[1])
        peak = max(1, items[0][1])
        gains = {}
        for rank, (sub, act) in enumerate(items):
            if rank < max(1, int(keep)):
                gains[sub] = 100
            else:
                gains[sub] = max(int(floor), min(99, (100 * act) // peak))
        return gains

    # 生物依据：默认模式网络（DMN）在无外部任务时仍保持自发活动，位于内侧前额叶/
    #   后扣带等联合-决策区（Raichle 2001；Buckner 2008）。
    # 简化说明：没有外部输入时，给决策层每个 tick 注入一小批固定的"内在节律"种子
    #   （确定、可复现，不用随机噪声），统计每 tick 的自发放电数和累计覆盖神经元；
    #   只读，不写权重、不改原型。
    # 参数来源：n_seed=32、强度 60、ticks=6 —— 工程简化，仅用于展示静息活动。
    def dmn_idle(self, ticks=6, n_seed=32, w=60):
        """DMN 自发活动：决策层无输入静息放电（只读，不写权重）。"""
        subs = self.hierarchy()["decision"]
        n_seed = max(1, int(n_seed))
        per_sub = {}
        total_spikes = 0
        total_unique = 0
        for s in subs:
            seen = set()
            counts = []
            for t in range(int(ticks)):
                start = (t * n_seed) % self.N     # 每 tick 换一批相邻神经元（确定节律）
                seeds = (start + np.arange(n_seed, dtype=np.int32)) % self.N
                wts = np.full(n_seed, int(w), dtype=np.int8)
                f, _, fired = self.fire(s, seeds, wts, tick=100000 + t)
                counts.append(int(f))
                total_spikes += int(f)
                seen.update(int(v) for v in fired.tolist())
            per_sub[s] = {"counts": counts, "spikes": int(sum(counts)),
                          "unique": len(seen)}
            total_unique += len(seen)
        denom = max(1, len(subs) * self.N * max(1, int(ticks)))
        return {"subs": subs, "per_sub": per_sub, "spikes": total_spikes,
                "unique": total_unique, "rate": round(total_spikes / float(denom), 6)}

    # 生物依据：睡眠期海马把白天的放电序列"重放"给皮层，完成记忆巩固
    #   （Wilson & McNaughton 1994；Diekelmann & Born 2010 主动系统巩固）。
    # 简化说明：把已保存的原型当内部输入重放（不再看原始文本），用 STDP 再巩固；
    #   结束后清零膜电位（睡眠复位），保证不影响第二天的推理。
    # 参数来源：每原型 strength=70、ticks=4 —— 工程简化，未做睡眠分期建模。
    def sleep_replay(self, epochs=1, ticks=4, strength=70):
        """睡眠重放：用已存原型做离线再巩固（只加强已有记忆，不新增标签）。"""
        if not self.protos:
            return {"replayed": 0, "epochs": int(epochs), "delta_pct": 0.0,
                    "weight_abs_before": 0, "weight_abs_after": 0}
        before = int(np.abs(self.weights.astype(np.int32)).sum())
        replayed = 0
        for ep in range(int(epochs)):
            for label, patterns in self.protos.items():
                sub = self.sub_of.get(label, self._sub_for(label))
                for pat in patterns:
                    if not pat:
                        continue
                    seeds = np.fromiter(sorted(pat), dtype=np.int32, count=len(pat))
                    wts = np.full(seeds.size, int(strength), dtype=np.int8)
                    prev = np.zeros(0, dtype=np.int32)
                    cur_s, cur_w = seeds, wts
                    for t in range(int(ticks)):
                        f, _, fired = self.fire(
                            sub, cur_s, cur_w, tick=200000 + ep * 1000 + t)
                        self.stdp_update(sub, fired, prev)
                        if f:
                            prev = fired
                            nb = self.neighbors[sub][fired].reshape(-1)
                            ww = self.weights[sub][fired].reshape(-1)
                            pos = ww > 0
                            cur_s, cur_w = nb[pos], ww[pos]
                            if len(cur_s) == 0:
                                cur_s, cur_w = seeds, wts
                        else:
                            cur_s, cur_w = seeds, wts
                    replayed += 1
        after = int(np.abs(self.weights.astype(np.int32)).sum())
        # 睡眠复位：清膜电位与时间戳（不把残留电位带进推理）
        self.state[:] = 0
        self.last_spike[:] = -1
        return {"replayed": replayed, "epochs": int(epochs),
                "weight_abs_before": before, "weight_abs_after": after,
                "delta_pct": round((after - before) * 100.0 / max(1, before), 3)}

    # 生物依据：突触稳态（synaptic scaling）——长期增强后神经元整体下调突触强度，
    #   把平均突触强度拉回工作区间（Turrigiano & Nelson 2004）。
    # 简化说明：按子脑计算"非零突触"的平均 |w|，高于 target 时整块乘性下调
    #   （整数乘除），低于 target 不放大（防止正反馈爆炸）；不改变符号。
    # 参数来源：target=2（INT8 量纲，训练后非零 |w| 约 2）—— 工程简化，未做生理标定。
    def homeostasis(self, target=2):
        """稳态：把每个子脑非零突触的平均 |w| 下调回目标（只降不升）。"""
        before, after = [], []
        for s in range(self.NSUB):
            w = self.weights[s].astype(np.int32)
            nz = w[w != 0]
            cur = int(np.abs(nz).sum()) // max(1, nz.size) if nz.size else 0
            before.append(cur)
            if cur > int(target):
                w = (w * int(target)) // cur
                self.weights[s] = np.clip(w, W_MIN, W_MAX).astype(np.int8)
                w2 = self.weights[s].astype(np.int32)
                nz2 = w2[w2 != 0]
                after.append(int(np.abs(nz2).sum()) // max(1, nz2.size) if nz2.size else 0)
            else:
                after.append(cur)
        return {"target": int(target), "mean_abs_before": before,
                "mean_abs_after": after}

    # 生物依据：睡眠期突触修剪（synaptic pruning）——弱连接被胶质细胞清除，
    #   保留被反复使用的强连接（Huttenlocher 1979；Tønnesen 2014）。
    # 简化说明：把 |w| <= threshold 的突触直接置 0（不是删除数组，保持形状可移植）。
    # 参数来源：threshold 由调用方给（默认 0 = 只清完全为 0 的），工程简化。
    def prune(self, threshold=0):
        """突触修剪：|w| <= threshold 置 0，返回剪掉数量与稀疏度。"""
        w = self.weights
        mask = np.abs(w.astype(np.int16)) <= int(threshold)
        pruned = int(mask.sum())
        w[mask] = 0
        total = int(w.size)
        return {"threshold": int(threshold), "pruned": pruned, "total": total,
                "sparsity": round(pruned / float(max(1, total)), 4),
                "nonzero": int(np.count_nonzero(w))}

    # ---------------- 保存 / 加载 ----------------

    def save(self, path):
        meta = {"nsub": self.NSUB, "n": self.N, "k": self.K, "seed": self.seed,
                "protos": {k: [sorted(p) for p in ps] for k, ps in self.protos.items()},
                "sub_of": self.sub_of}
        np.savez_compressed(
            path,
            state=self.state,
            weights=self.weights,
            neighbors=self.neighbors,
            last_spike=self.last_spike,
            meta=np.array(json.dumps(meta)),
        )
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        b = cls.__new__(cls)
        b.NSUB, b.N, b.K = int(meta["nsub"]), int(meta["n"]), int(meta["k"])
        b.seed = int(meta.get("seed", 0))
        b.state = z["state"].astype(np.int8)
        b.weights = z["weights"].astype(np.int8)
        b.neighbors = z["neighbors"].astype(np.int32)
        b.last_spike = z["last_spike"].astype(np.int32)
        raw = meta.get("protos", {})
        b.protos = {}
        for k, v in raw.items():
            if v and isinstance(v[0], list):          # 新模式：list[list[int]]
                b.protos[k] = [set(int(x) for x in p) for p in v]
            else:                                     # 兼容旧模式：list[int]
                b.protos[k] = [set(int(x) for x in v)]
        b.sub_of = {k: int(v) for k, v in meta.get("sub_of", {}).items()}
        return b

    # ---------------- 统计 ----------------

    def quant_stats(self):
        """量化体检：溢出/饱和/空值（稳定性护栏）。"""
        sat = int(np.sum(self.state >= SCALE))   # 真正饱和（顶到 100）的单元
        return {"dtype_state": str(self.state.dtype),
                "dtype_weights": str(self.weights.dtype),
                "saturated_cells": sat,
                "mem_mb": round((self.state.nbytes + self.weights.nbytes
                                  + self.neighbors.nbytes + self.last_spike.nbytes)
                                 / 1024 / 1024, 4)}

    def stats(self):
        w = self.weights
        return {"neurons": self.NSUB * self.N,
                "synapses": self.NSUB * self.N * self.K,
                "subs": self.NSUB,
                "state_mb": round(self.state.nbytes / 1048576, 3),
                "weights_mb": round(w.nbytes / 1048576, 3),
                "neighbors_mb": round(self.neighbors.nbytes / 1048576, 3),
                "total_mb": round((self.state.nbytes + w.nbytes + self.neighbors.nbytes
                                   + self.last_spike.nbytes) / 1048576, 3),
                "weight_min": int(w.min()), "weight_max": int(w.max()),
                "weight_mean": round(float(w.mean()), 3),
                "weight_nonzero": int(np.count_nonzero(w)),
                "sparsity": round(1.0 - int(np.count_nonzero(w)) / float(max(1, w.size)), 4),
                "protos": len(self.protos),
                "patterns": sum(len(v) for v in self.protos.values())}


def selftest():
    print("=== lianghua 量化类脑自检（8 子脑 × 2048，全 INT8/定点）===")
    b = QuantizedBrain()
    print("量化体检:", b.quant_stats())
    # 喂一批种子放电
    seeds = np.arange(64, dtype=np.int32)
    wts = np.full(64, 40, dtype=np.int8)   # 强输入（INT8），三轮累积让膜电位过 THR=50
    f1, _, _ = b.fire(0, seeds, wts, tick=1)
    f2, _, _ = b.fire(0, seeds, wts, tick=2)
    f3, _, _ = b.fire(0, seeds, wts, tick=3)
    print("子脑0 三轮放电 %d→%d→%d（膜电位 INT8 累积过阈）" % (f1, f2, f3))
    # 门控
    q = np.array([2, -3, 1, 0, 4, -1, 2, 3], dtype=np.int8)
    print("丘脑门控 top-4:", b.gate_topk(q))
    print("OK（纯 INT8/定点，无 float，可移植 MCU）")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
