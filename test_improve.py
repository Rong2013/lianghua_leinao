# -*- coding: utf-8 -*-
"""test_improve.py - 验证改进：真联网 + 缓存 TTL + 联网信息喂脑。"""
import urllib.request
import urllib.parse
import json
import time

BASE = "http://localhost:8000"


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=40).read())


def post(path, payload):
    req = urllib.request.Request(BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def main():
    print("=== 真联网（清缓存后首次走网）===")
    r1 = get("/api/lookup?text=" + urllib.parse.quote("黑洞"))
    print("  首次: source=%s cached=%s offline=%s" % (r1["source"], r1["cached"], r1["offline"]))
    print("  内容:", r1["text"][:50])

    print("\n=== 缓存 TTL（紧接着再查应 cached=true）===")
    r2 = get("/api/lookup?text=" + urllib.parse.quote("黑洞"))
    print("  二次: cached=%s source=%s" % (r2["cached"], r2["source"]))

    print("\n=== 联网查到的信息喂脑（一键学习）===")
    # 用联网查到的"黑洞"信息，教脑新标签"天体"
    lr = post("/api/learn", {"label": "天体", "texts": [r1["text"][:80], "黑洞是引力极强的天体"]})
    d = lr.get("train_detail", {})
    print("  学「天体」: ok=%s 改突触=%s 子脑=%s" % (lr.get("ok"), d.get("synapses_changed"), d.get("subs_active")))
    m = get("/api/meta")
    print("  当前标签:", m["labels"])

    print("\n=== 学完能对话 ===")
    r = get("/api/chat?text=" + urllib.parse.quote("黑洞是引力极强的天体"))
    print("  问「黑洞是引力极强的天体」-> %s (in_kb=%s)" % (r.get("label"), r["in_knowledge_base"]))


if __name__ == "__main__":
    main()
