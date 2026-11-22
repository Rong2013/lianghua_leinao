# -*- coding: utf-8 -*-
"""test_final.py - 最终验证（数学答案 + 训练详情 + 干净脑）。"""
import urllib.request
import urllib.parse
import json

BASE = "http://localhost:8000"


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=30).read())


def post(path, payload):
    req = urllib.request.Request(BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def main():
    print("=== 数学算式（应该答答案，不背语料）===")
    for q in ["2+5等于几", "3*4", "10-3", "7*8"]:
        r = get("/api/chat?text=" + urllib.parse.quote(q))
        print("  %-12s -> %s" % (q, r["reply"]))

    print("\n=== 知识库内（回最相关语料）===")
    for q in ["猫 狗", "今天天气", "hello 你好"]:
        r = get("/api/chat?text=" + urllib.parse.quote(q))
        print("  %-14s -> %s" % (q, r["reply"][:30]))

    print("\n=== 越界（知识库外，诚实提示）===")
    for q in ["量子纠缠", "秦始皇"]:
        r = get("/api/chat?text=" + urllib.parse.quote(q))
        print("  %-12s -> %s" % (q, r["reply"][:40]))

    print("\n=== 训练详情（群主要看怎么训的）===")
    lr = post("/api/learn", {"label": "历史", "texts": ["唐朝诗人李白", "宋朝活字印刷"]})
    d = lr.get("train_detail", {})
    print("  STDP 改了 %s 突触，%s 子脑参与" % (d.get("synapses_changed"), d.get("subs_active")))
    print("  学完识别「%s」→ 「%s」置信度 %.0f%%" %
          ("唐朝诗人李白", d.get("probe_label"), d.get("probe_score", 0)*100))


if __name__ == "__main__":
    main()
