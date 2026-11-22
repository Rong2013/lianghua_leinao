# -*- coding: utf-8 -*-
"""test_chat_math.py - 验证数学答答案 + 训练详情。"""
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
    print("=== 数学类：应该答答案，不是背语料 ===")
    for q in ["2+5等于几", "3*4", "10-3"]:
        r = get("/api/chat?text=" + urllib.parse.quote(q) + "&mode=hopfield")
        print("  %-14s -> %s" % (q, r["reply"]))

    print("\n=== 非数学类：回最相关语料 ===")
    for q in ["猫 狗", "今天天气"]:
        r = get("/api/chat?text=" + urllib.parse.quote(q))
        print("  %-14s -> %s" % (q, r["reply"]))

    print("\n=== 训练详情（看怎么训的）===")
    lr = post("/api/learn", {"label": "游戏", "texts": ["塞尔达 任天堂", "我的世界 沙盒"]})
    print("  学会了「游戏」：", lr.get("train_detail"))
    if lr.get("retest_accuracy") is not None:
        print("  测试集重测: %.1f%%" % (lr["retest_accuracy"]*100))


if __name__ == "__main__":
    main()
