# -*- coding: utf-8 -*-
"""test_bing_urls.py - 测 Bing 能否抓结果 URL（用于取正文）。"""
import urllib.request
import urllib.parse
import re

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def probe(query):
    url = "https://www.bing.com/search?" + urllib.parse.urlencode(
        {"q": query, "count": "5", "setlang": "zh-CN"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")
    # b_algo 块里的 <a href>
    blocks = re.findall(r'<li class="b_algo">(.*?)</li>', html, re.S)
    for b in blocks[:3]:
        a = re.search(r'<a[^>]*href="([^"]+)"', b)
        h = re.search(r'<h2[^>]*>(.*?)</h2>', b, re.S)
        href = a.group(1) if a else "?"
        title = re.sub(r"<[^>]+>", "", h.group(1)).strip() if h else "?"
        print("   ", href[:60], "|", title[:30])
    return html


print("=== 查「黑洞」的 URL ===")
probe("黑洞")
