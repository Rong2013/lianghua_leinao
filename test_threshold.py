# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"I:\lianghua")
import lianghua
b = lianghua.QuantizedBrain.load(r"I:\lianghua\demo_brain.npz")
print("问候类各种措辞的识别 score：")
for q in ["你好啊", "你好", "hi", "hello", "早上好", "嗨 好久不见", "哈喽 你今天好吗"]:
    r = b.think(q, top_k=4, ticks=6)
    print("  %-16s -> %s (score=%.3f)" % (q, r["label"], r["score"]))
print("\n当前阈值 0.30：低于它的短问候全判越界（bug）")
