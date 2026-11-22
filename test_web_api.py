# -*- coding: utf-8 -*-
"""test_web_api.py - 验证 /api/lookup + /api/netmode（简化，不触发训练）。"""
import urllib.request
import urllib.parse
import json

BASE = "http://localhost:8000"


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=30).read())


def main():
    print("=== /api/netmode ===")
    m = get("/api/netmode")
    print("  ", m)

    print("\n=== /api/lookup（不带 label，只查）===")
    for q in ["狗", "秦始皇", "黑洞"]:
        r = get("/api/lookup?text=" + urllib.parse.quote(q))
        print("  %-8s -> found=%s source=%s text=%s"
              % (q, r["found"], r["source"], (r["text"][:25]+"…") if r["text"] else "(空)"))

    print("\n=== 知识库 ===")
    m2 = get("/api/meta")
    print("  标签数:", len(m2["labels"]), m2["labels"])


if __name__ == "__main__":
    main()
