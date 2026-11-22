# -*- coding: utf-8 -*-
"""test_greeting.py - 验证短问候不再误判越界（阈值降到 0.15）。"""
import urllib.request
import urllib.parse
import json

BASE = "http://localhost:8000"


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=30).read())


def main():
    print("阈值 0.15 下的识别（短问候/近义措辞应 in_kb=True）：")
    for q in ["你好啊", "你好", "hi", "早上好", "哈喽 你今天好吗"]:
        r = get("/api/chat?text=" + urllib.parse.quote(q))
        print("  %-14s -> %s in_kb=%s score=%.3f reply=%s"
              % (q, r.get("label"), r["in_knowledge_base"], r["score"],
                 r["reply"][:20]))
    print("\n知识库外（仍应越界）：")
    for q in ["量子纠缠", "黑洞"]:
        r = get("/api/chat?text=" + urllib.parse.quote(q))
        print("  %-14s -> in_kb=%s score=%.3f" % (q, r["in_knowledge_base"], r["score"]))


if __name__ == "__main__":
    main()
