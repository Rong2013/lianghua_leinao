# -*- coding: utf-8 -*-
"""rplus_simple.py - R+ 的简化版核心聚合（量化骨架专用，纯整数）。

# 生物依据：无，工程简化（R+ 是运行时/数据布局优化，不对应某个脑区）。
# 简化说明：把 I:/R+ 的「编码键一次排序完成分组+聚合 + 行缓存」搬到量化骨架：
#   - aggregate(): 三种等价聚合（key / unique / argsort），默认 key（int64 排序）；
#     数值越界或异常自动回退 unique（正确性是硬底线，加速是可选件）。
#   - RowCache(): 环形槽位行缓存，把反复访问的 (子脑, 神经元) 邻居/权重行留在内存。
#   - quant_fire() / propagate(): 与 QuantizedBrain.fire / propagate **逐位一致**的
#     R+ 路径，唯一区别是聚合用编码键排序，而不是 np.unique + np.add.at。
# 参数来源：缓存容量 65536 行（≈4 MB）、key 编码低 9 位存 w+256 —— 工程简化，
#   沿用 I:/R+/bin/rplus.py 的 key 编码思路（那边低 9 位、这里同样低 9 位）。
#
# 重要：本模块是**可选加速/演示路径**，默认不启用；默认训练与推理仍走
#   lianghua.py 的原路径，已验证的留出集指标不受影响。
#
# 用法：
#   python rplus_simple.py                 # 自检：R+ 路径 vs 原路径是否完全一致
"""
import os

import numpy as np

try:                       # 同目录，正常导入；单独拷走时可缺省
    import lianghua
except Exception:          # pragma: no cover
    lianghua = None

# 低 9 位存 (w + 256)，w ∈ [-128,127] → [128,383]，不越 511
_W_BITS = 9
_W_MASK = (1 << _W_BITS) - 1
_W_OFF = 256


def _agg_key(ids, wts):
    """编码键聚合：把 (目标 id, 权重) 压进一个 int64 键，一次排序完成分组+求和。"""
    n = int(ids.size)
    if n == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    if int(ids.min()) < 0 or int(wts.min()) < -128 or int(wts.max()) > 127:
        raise ValueError("id/weight out of key-encoding range")
    if int(ids.max()) > (1 << 40):
        raise ValueError("id too large for key encoding")
    key = (ids.astype(np.int64) << _W_BITS) | (wts.astype(np.int64) + _W_OFF)
    key.sort()
    ids_s = (key >> _W_BITS).astype(np.int32)
    wts_s = ((key & _W_MASK).astype(np.int32) - _W_OFF)
    change = np.empty(n, dtype=bool)
    change[0] = True
    np.not_equal(ids_s[1:], ids_s[:-1], out=change[1:])
    starts = np.flatnonzero(change)
    sums = np.add.reduceat(wts_s, starts)
    return ids_s[starts], sums.astype(np.int32)


def _agg_unique(ids, wts):
    """最保守：unique(return_inverse) + add.at（与 lianghua.fire 完全同法）。"""
    uniq, inv = np.unique(ids, return_inverse=True)
    sums = np.zeros(uniq.size, dtype=np.int32)
    np.add.at(sums, inv, wts)
    return uniq.astype(np.int32), sums


def _agg_argsort(ids, wts):
    """argsort + reduceat：O(n log n)，与 unique 结果等价。"""
    order = np.argsort(ids, kind="stable")
    ids_s = ids[order]
    wts_s = wts[order]
    if ids_s.size == 0:
        return ids_s.astype(np.int32), wts_s.astype(np.int32)
    starts = np.flatnonzero(np.r_[True, ids_s[1:] != ids_s[:-1]])
    sums = np.add.reduceat(wts_s, starts)
    return ids_s[starts].astype(np.int32), sums.astype(np.int32)


_AGG = {"key": _agg_key, "unique": _agg_unique, "argsort": _agg_argsort}


def aggregate(ids, wts, mode=None):
    """聚合重复目标：返回 (降序去重后的 id, int32 权重和)。

    默认读环境变量 RPLUS_AGG（缺省 key）。任何异常/非法值 → 回退 unique。
    """
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    wts = np.asarray(wts, dtype=np.int32).reshape(-1)
    if ids.size != wts.size:
        raise ValueError("ids/wts length mismatch")
    if ids.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    mode = mode or os.environ.get("RPLUS_AGG", "key")
    fn = _AGG.get(mode, _agg_key)
    try:
        out_ids, out_sums = fn(ids, wts)
        if out_ids.size == 0 or int(out_ids.min()) < 0:
            raise ValueError("bad aggregate output")
        if not np.all(np.isfinite(out_sums)):
            raise ValueError("non-finite sums")
        return out_ids, out_sums
    except Exception:
        return _agg_unique(ids, wts)


class RowCache:
    """环形槽位行缓存（简化版）：只缓存被访问过的 (子脑, 神经元) 行。

    命中率低时只是多一层 dict 查找；未命中直接从内存表 gather（本骨架全部在内存，
    所以这里展示的是策略而非 IO 加速）。跨进程大表版见 I:/R+/bin/rplus.py。
    """

    def __init__(self, rows=65536, k=32):
        self.cap = max(1024, int(rows))
        self.k = int(k)
        self.nb = np.zeros((self.cap, self.k), dtype=np.int32)
        self.wt = np.zeros((self.cap, self.k), dtype=np.int8)
        self.slot = {}
        self.next = 0
        self.hits = 0
        self.misses = 0

    def stats(self):
        tot = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / tot, 4) if tot else 0.0,
                "rows": len(self.slot), "cap": self.cap}

    def clear(self):
        self.slot.clear()
        self.next = 0
        self.hits = 0
        self.misses = 0

    def fetch(self, brain, sub, idx):
        """按 idx 顺序返回 (邻居行, 权重行)；同一行只解析一次。"""
        idx = np.asarray(idx, dtype=np.int32).reshape(-1)
        n = idx.size
        out_nb = np.empty((n, self.k), dtype=np.int32)
        out_wt = np.empty((n, self.k), dtype=np.int8)
        if n == 0:
            return out_nb, out_wt
        uniq, inv = np.unique(idx, return_inverse=True)
        rows_nb = np.empty((uniq.size, self.k), dtype=np.int32)
        rows_wt = np.empty((uniq.size, self.k), dtype=np.int8)
        slot = self.slot
        for j, gid in enumerate(uniq.tolist()):
            key = (int(sub), int(gid))
            s = slot.get(key)
            if s is None:
                rows_nb[j] = brain.neighbors[sub, gid]
                rows_wt[j] = brain.weights[sub, gid]
                s = self.next
                self.nb[s] = rows_nb[j]
                self.wt[s] = rows_wt[j]
                slot[key] = s
                self.next = (self.next + 1) % self.cap
                self.misses += 1
            else:
                rows_nb[j] = self.nb[s]
                rows_wt[j] = self.wt[s]
                self.hits += 1
        out_nb[:] = rows_nb[inv]
        out_wt[:] = rows_wt[inv]
        return out_nb, out_wt


def quant_fire(brain, sub, seeds, wts, tick, mode=None):
    """R+ 聚合版放电：与 QuantizedBrain.fire 的 fired 集合逐位一致。

    只替换「重复目标求和」这一步（key 编码一次排序），膜电位公式逐字照搬。
    """
    st = brain.state[sub]
    seeds = np.asarray(seeds, dtype=np.int32).reshape(-1)
    if seeds.size == 0:
        return 0, 0.0, np.zeros(0, dtype=np.int32)
    leak_num = int(getattr(lianghua, "LEAK_NUM", 90)) if lianghua else 90
    leak_den = int(getattr(lianghua, "LEAK_DEN", 100)) if lianghua else 100
    thr = int(getattr(lianghua, "THR", 50)) if lianghua else 50
    scale = int(getattr(lianghua, "SCALE", 100)) if lianghua else 100
    s_sorted, input_sum = aggregate(seeds, np.asarray(wts, dtype=np.int32), mode=mode)
    base = st[s_sorted].astype(np.int32)
    newv = (base * leak_num) // leak_den + input_sum
    newv = np.clip(newv, 0, scale).astype(np.int8)
    brain.state[sub][s_sorted] = newv
    fired = np.flatnonzero(newv >= thr)
    fired_global = s_sorted[fired]
    if fired_global.size:
        brain.state[sub][fired_global] = 0
        brain.last_spike[sub][fired_global] = tick
    sync = float(fired_global.size) / max(1, seeds.size)
    return int(fired_global.size), sync, fired_global


def propagate(brain, sub, seeds, wts, ticks=6, mode=None, cache=None, tick0=0):
    """R+ 版多 tick 传播；cache 非空时邻居行走行缓存。结果与 brain.propagate 一致。"""
    seeds = np.asarray(seeds, dtype=np.int32).reshape(-1)
    wts = np.asarray(wts, dtype=np.int8).reshape(-1)
    counts, fired_all = [], []
    cur_s, cur_w = seeds, wts
    for t in range(int(ticks)):
        f, _sync, fired = quant_fire(brain, sub, cur_s, cur_w, tick0 + t, mode=mode)
        counts.append(int(f))
        fired_all.append(fired)
        if f == 0:
            cur_s, cur_w = seeds, wts
            continue
        if cache is not None:
            nb_rows, wt_rows = cache.fetch(brain, sub, fired)
            pos = wt_rows > 0
            cur_s = nb_rows[pos]
            cur_w = wt_rows[pos]
        else:
            nb = brain.neighbors[sub][fired].reshape(-1)
            w = brain.weights[sub][fired].reshape(-1)
            pos = w > 0
            cur_s, cur_w = nb[pos], w[pos]
        if len(cur_s) == 0:
            cur_s, cur_w = seeds, wts
    ne = [x for x in fired_all if len(x)]
    total = np.unique(np.concatenate(ne)) if ne else np.zeros(0, dtype=np.int32)
    return {"counts": counts, "fired": total}


# ---------------- 一致性自检 ----------------

def _snapshot(brain):
    return brain.state.copy(), brain.last_spike.copy()


def _restore(brain, snap):
    brain.state[:] = snap[0]
    brain.last_spike[:] = snap[1]


def selftest(brain=None, texts=None, verbose=True):
    """同一输入下，R+ 路径 vs 原路径：单 tick fired 集合、多 tick fired 集合必须相等。"""
    import encoder
    import lianghua

    if brain is None:
        brain = lianghua.QuantizedBrain()
    texts = texts or ["今天天气怎么样，会下雨吗", "python 代码报错了怎么调试",
                      "钢琴和吉他哪个好听", "再见，明天见"]
    modes = ["key", "unique", "argsort"]
    subs = list(range(min(3, brain.NSUB)))

    fire_total, fire_ok = 0, 0
    prop_total, prop_ok = 0, 0
    per_mode = {m: {"fire_ok": 0, "prop_ok": 0} for m in modes}
    cache = RowCache(k=brain.K)

    for text in texts:
        seeds, wts = encoder.text_to_seeds(text, brain.N)
        if seeds.size == 0:
            continue
        for sub in subs:
            snap = _snapshot(brain)
            n0, _, f0 = brain.fire(sub, seeds, wts, tick=0)
            _restore(brain, snap)
            for mode in modes:
                n1, _, f1 = quant_fire(brain, sub, seeds, wts, tick=0, mode=mode)
                _restore(brain, snap)
                ok = (n0 == n1) and np.array_equal(np.sort(f0), np.sort(f1))
                fire_total += 1
                fire_ok += int(ok)
                per_mode[mode]["fire_ok"] += int(ok)

            p0 = brain.propagate(sub, seeds, wts, ticks=6)
            _restore(brain, snap)
            for mode in modes:
                p1 = propagate(brain, sub, seeds, wts, ticks=6, mode=mode, cache=cache)
                _restore(brain, snap)
                ok = np.array_equal(np.sort(p0["fired"]), np.sort(p1["fired"]))
                prop_total += 1
                prop_ok += int(ok)
                per_mode[mode]["prop_ok"] += int(ok)

    out = {"fire_cases": fire_total, "fire_equal": fire_ok,
           "prop_cases": prop_total, "prop_equal": prop_ok,
           "per_mode": per_mode, "cache": cache.stats()}
    if verbose:
        print("R+ 简化版一致性自检")
        print("  单 tick fired 集合一致：%d/%d" % (fire_ok, fire_total))
        print("  多 tick fired 集合一致：%d/%d" % (prop_ok, prop_total))
        for m in modes:
            print("    %-7s 单 tick %d/%d  多 tick %d/%d"
                  % (m, per_mode[m]["fire_ok"], fire_total // len(modes),
                     per_mode[m]["prop_ok"], prop_total // len(modes)))
        c = out["cache"]
        print("  行缓存：命中 %d / 未命中 %d（命中率 %.1f%%，%d/%d 行）"
              % (c["hits"], c["misses"], c["hit_rate"] * 100, c["rows"], c["cap"]))
        print("  结论：%s"
              % ("PASS 与原路径完全一致" if (fire_ok == fire_total and prop_ok == prop_total)
                 else "FAIL 存在不一致"))
    return out


if __name__ == "__main__":
    selftest()
