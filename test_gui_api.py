# -*- coding: utf-8 -*-
"""test_gui_api.py - 验证 GUI 的 /api/chat（Jaccard + Hopfield）+ /api/learn。"""
import urllib.request
import urllib.parse
import json

BASE = "http://localhost:8000"


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path).read())


def post_learn(payload):
    req = urllib.request.Request(
        BASE + "/api/learn",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def main():
    print("=== Jaccard 模式（一次匹配）===")
    for q in ["猫 狗", "今天天气怎么样", "1+1等于几"]:
        r = get("/api/chat?text=" + urllib.parse.quote(q) + "&mode=jaccard")
        print("  %-16s -> %s (score=%.3f in_kb=%s)" % (q, r["label"], r["score"], r["in_knowledge_base"]))

    print("\n=== Hopfield 模式（联想收敛）===")
    for q in ["猫 狗", "今天天气怎么样", "1+1等于几", "量子纠缠"]:
        r = get("/api/chat?text=" + urllib.parse.quote(q) + "&mode=hopfield")
        print("  %-16s -> %s (score=%.3f iters=%d in_kb=%s)"
              % (q, r.get("raw_label"), r["score"], r.get("iters", 0), r["in_knowledge_base"]))

    print("\n=== 在线学习（教脑新话题'游戏'）===")
    lr = post_learn({"label": "游戏", "texts": ["王者荣耀", "英雄联盟", "吃鸡 绝地求生", "原神 开放世界"]})
    print("  learn:", lr)
    # 学了之后
    r = get("/api/chat?text=" + urllib.parse.quote("王者荣耀") + "&mode=hopfield")
    print("  Hopfield 王者荣耀(学过): label=%r score=%.3f in_kb=%s" % (r.get("raw_label"), r["score"], r["in_knowledge_base"]))
    m = get("/api/meta")
    print("  当前知识库:", m["labels"])


if __name__ == "__main__":
    main()
