# -*- coding: utf-8 -*-
"""test_real_web.py - 验证 GUI 真联网（Bing）+ 缓存。"""
import urllib.request
import urllib.parse
import json
import time

BASE = "http://localhost:8000"


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=40).read())


def main():
    print("=== /api/netmode（应有 bing=True）===")
    m = get("/api/netmode")
    print("  ", m)

    print("\n=== /api/lookup 真联网（首次走网络，缓存后秒回）===")
    for q in ["黑洞", "量子纠缠", "秦始皇"]:
        t0 = time.time()
        r = get("/api/lookup?text=" + urllib.parse.quote(q))
        print("  %-8s source=%s offline=%s cached=%s 耗时%.1fs"
              % (q, r["source"], r["offline"], r["cached"], time.time()-t0))
        print("    内容:", r["text"][:50])

    print("\n=== 知识库外的词也能联网查到（不是本地词典）===")
    r = get("/api/lookup?text=" + urllib.parse.quote("黑洞"))
    print("  黑洞 source=%s（bing=真联网，offline=本地词典）" % r["source"])


if __name__ == "__main__":
    main()
