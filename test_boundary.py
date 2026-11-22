# -*- coding: utf-8 -*-
"""test_boundary.py - 展示量化脑的知识库边界（群主问"1元2次方程是动物"）。"""
import sys
sys.path.insert(0, r"I:\lianghua")
import lianghua

b = lianghua.QuantizedBrain.load(r"I:\lianghua\demo_brain.npz")
print("知识库标签:", list(b.texts.keys()))
print("\n=== 知识库内（学过 → 能识别）===")
for q in ["1+1等于几", "今天天气怎么样", "猫 狗 宠物", "python 代码 函数"]:
    r = b.think(q, top_k=4, ticks=6)
    print("  %-18s -> %s (score=%.3f)" % (q, r["label"], r["score"]))

print("\n=== 知识库外（没学过 → 识别不准/乱猜，这是诚实边界）===")
for q in ["1元2次方程是动物", "量子纠缠原理", "秦始皇统一六国", "今天股票涨了吗"]:
    r = b.think(q, top_k=4, ticks=6)
    print("  %-18s -> %s (score=%.3f)  <- 没学过这个，结果是脑强行匹配最近的" % (q, r["label"], r["score"]))
