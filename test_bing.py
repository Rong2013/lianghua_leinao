# -*- coding: utf-8 -*-
"""test_bing.py - 测 Bing 能不能抓真搜索结果/摘要。"""
import urllib.request
import urllib.parse
import re
import json

UA = "Mozilla/5.0"


def search(query):
    """Bing 搜索：抓结果标题 + 摘要。"""
    url = "https://www.bing.com/search?" + urllib.parse.urlencode({"q": query, "count": "3"})
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "zh-CN"})
    html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")
    # 检测反爬
    if "验证" in html[:500] or "captcha" in html.lower()[:500]:
        return None, "反爬拦截"
    # 抓结果块（Bing 用 <li class="b_algo">，标题在 <h2>，摘要在 .b_caption）
    blocks = re.findall(r'<li class="b_algo">(.*?)</li>', html, re.S)
    results = []
    for b in blocks[:3]:
        title_m = re.search(r'<h2>.*?<a[^>]*>(.*?)</a>', b, re.S)
        cap_m = re.search(r'<p[^>]*>(.*?)</p>', b, re.S)
        title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip() if title_m else ""
        cap = re.sub(r"<[^>]+>", "", cap_m.group(1)).strip() if cap_m else ""
        if title or cap:
            results.append({"title": title, "snippet": cap})
    if not results:
        # 备用：抓所有 <h2>
        hs = re.findall(r'<h2[^>]*>(.*?)</h2>', html, re.S)
        for h in hs[:3]:
            results.append({"title": re.sub(r"<[^>]+>", "", h).strip(), "snippet": ""})
    return results, "ok"


for q in ["狗", "量子纠缠", "秦始皇", "黑洞"]:
    res, status = search(q)
    print("=== 查「%s」(%s) ===" % (q, status))
    if res:
        for r in res[:2]:
            print("  标题:", r["title"][:40])
            print("  摘要:", r["snippet"][:60])
    print()
