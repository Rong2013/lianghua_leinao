# -*- coding: utf-8 -*-
"""test_network.py - 全面测本机网络可达性（多源 + 代理 + 长超时）。"""
import urllib.request
import json
import socket

UA = "Mozilla/5.0"


def probe(name, url, timeout=15):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        r = urllib.request.urlopen(req, timeout=timeout)
        return "OK %s" % r.status
    except Exception as e:
        return "FAIL %s %s" % (type(e).__name__, str(e)[:60])


def probe_proxy(name, url):
    """走 127.0.0.1:7890 代理访问。"""
    import urllib.request
    proxy = urllib.request.ProxyHandler({"http": "http://127.0.0.1:7890",
                                          "https": "http://127.0.0.1:7890"})
    opener = urllib.request.build_opener(proxy)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        r = opener.open(req, timeout=15)
        return "PROXY-OK %s" % r.status
    except Exception as e:
        return "PROXY-FAIL %s %s" % (type(e).__name__, str(e)[:50])


# 代理在跑吗
for p in [7890, 7897, 1080, 3128]:
    s = socket.socket(); s.settimeout(2)
    try:
        s.connect(("127.0.0.1", p)); print("  代理端口 %d 通" % p); s.close()
    except Exception:
        print("  代理端口 %d 不通" % p)

print("\n=== 直连测试（15s 超时）===")
tests = [
    ("Bing 搜索", "https://www.bing.com/search?q=test"),
    ("Baidu", "https://www.baidu.com"),
    ("Wikipedia 中", "https://zh.wikipedia.org/api/rest_v1/page/summary/Test"),
    ("Wikipedia 英", "https://en.wikipedia.org/api/rest_v1/page/summary/Dog"),
    ("DuckDuckGo", "https://duckduckgo.com"),
    ("DuckDuckGo API", "https://api.duckduckgo.com/?q=dog&format=json"),
    ("Wolfram|Alpha", "https://api.wolframalpha.com/v2/query"),
]
for name, url in tests:
    print("  %-14s %s" % (name, probe(name, url)))

print("\n=== 走 7890 代理 ===")
for name, url in [("Wikipedia 中(代理)", "https://zh.wikipedia.org/api/rest_v1/page/summary/Test"),
                  ("Bing(代理)", "https://www.bing.com/search?q=test")]:
    print("  %-14s %s" % (name, probe_proxy(name, url)))
