# -*- coding: utf-8 -*-
"""test_bing_snip.py - 测 Bing 能否抓摘要正文（不只标题）。"""
import urllib.request
import urllib.parse
import re

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

url = "https://www.bing.com/search?" + urllib.parse.urlencode(
    {"q": "黑洞", "count": "5", "setlang": "zh-CN"})
req = urllib.request.Request(url, headers={"User-Agent": UA})
html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")

# 看 b_algo 块里所有 p / span（摘要通常在这）
blocks = re.findall(r'<li class="b_algo">(.*?)</li>', html, re.S)
print("b_algo 块数:", len(blocks))
for i, b in enumerate(blocks[:2]):
    print("---- 块 %d ----" % i)
    # 抓 <h2>
    h = re.search(r'<h2[^>]*>(.*?)</h2>', b, re.S)
    print("标题:", re.sub(r"<[^>]+>", "", h.group(1)).strip()[:50] if h else "?")
    # 抓所有 <p> 和 <span class="b_lineclamp...">
    for m in re.finditer(r'<p[^>]*>(.*?)</p>', b, re.S):
        txt = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        if len(txt) > 5:
            print("p:", txt[:80])
    for m in re.finditer(r'<span class="b_[^"]*lineclamp[^"]*"[^>]*>(.*?)</span>', b, re.S):
        txt = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        if len(txt) > 5:
            print("span:", txt[:80])
