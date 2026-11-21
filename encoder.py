# -*- coding: utf-8 -*-
"""encoder.py - 文本 → 脉冲种子（量化类脑的输入编码器）。

# 生物依据：感觉输入在皮层被展开成稀疏的"神经元群"放电（稀疏编码）；
#   字符 1~3-gram 混合近似"字→词→短语"的多尺度表征，短词也有响应。
# 简化说明：用 blake2b 稳定哈希把 n-gram 映射到神经元编号（0..N-1），
#   每个 n-gram 产生 1 个种子 + 一个 INT8 权重（长 gram 权重更大）；
#   同一 n-gram 永远落到同一神经元（可复现，跨进程一致）。
# 参数来源：ngram=3、混合 1..3、权重 20..70（按 gram 长度缩放）—— 无，工程简化。
"""
import hashlib

import numpy as np

_DIGEST = 8


def _h(s: str, salt: int = 0) -> int:
    data = ("%d\x00%s" % (salt, s)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(data, digest_size=_DIGEST).digest(), "big")


def text_to_seeds(text, n, ngram=3, salt=0, max_seeds=320, wmin=60, wmax=110, mix=True):
    """文本 → (seeds int32[], wts int8[])。
    mix=True 时混合 1..ngram 的字符 n-gram：短 gram 权重小、长 gram 权重大。
    """
    text = (text or "").strip()
    if not text:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int8)

    s = " %s " % text.lower()
    items = []
    sizes = range(1, ngram + 1) if mix else (ngram,)
    for g in sizes:
        cnt = max(1, len(s) - g + 1)
        for i in range(cnt):
            items.append((g, s[i:i + g]))

    if len(items) > max_seeds:
        step = len(items) / float(max_seeds)
        items = [items[int(i * step)] for i in range(max_seeds)]

    acc = {}
    for g, gram in items:
        idx = _h(gram, salt) % n
        # 长 gram 权重更高（unigram 最小），再加一点哈希抖动
        w = int(wmin * g / float(max(1, ngram))) + (_h(gram, salt + 1) % 20)
        acc[idx] = min(127, acc.get(idx, 0) + w)

    seeds = np.fromiter(acc.keys(), dtype=np.int32, count=len(acc))
    wts = np.clip(np.fromiter(acc.values(), dtype=np.int32, count=len(acc)), 0, 127).astype(np.int8)
    return seeds, wts


def bytes_to_seeds(data, n, ngram=2, salt=0, max_seeds=256, wmin=60, wmax=110):
    """字节流 → 脉冲种子（规则同上，给非文本输入用）。"""
    if not data:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int8)
    grams = [bytes(data[i:i + ngram]).hex() for i in range(max(1, len(data) - ngram + 1))]
    if len(grams) > max_seeds:
        step = len(grams) / float(max_seeds)
        grams = [grams[int(i * step)] for i in range(max_seeds)]
    acc = {}
    for g in grams:
        idx = _h(g, salt) % n
        w = wmin + (_h(g, salt + 1) % (wmax - wmin + 1))
        acc[idx] = min(127, acc.get(idx, 0) + w)
    seeds = np.fromiter(acc.keys(), dtype=np.int32, count=len(acc))
    wts = np.clip(np.fromiter(acc.values(), dtype=np.int32, count=len(acc)), 0, 127).astype(np.int8)
    return seeds, wts


if __name__ == "__main__":
    import sys
    t = " ".join(sys.argv[1:]) or "hello brain"
    s, w = text_to_seeds(t, 2048)
    print("text:", t)
    print("seeds:", s.tolist()[:12], "... 共", len(s))
    print("wts  :", w.tolist()[:12], "max", int(w.max()) if len(w) else 0)
