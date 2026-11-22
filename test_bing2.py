# -*- coding: utf-8 -*-
"""test_bing2.py - 改进 Bing 抓取（拿摘要正文 + 多模式）。"""
import urllib.request
import urllib.parse
import re

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def search(query, count=3):
    url = "https://www.bing.com/search?" + urllib.parse.urlencode(
        {"q": query, "count": "5", "setlang": "zh-CN", "cc": "cn"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")
    if "验证" in html[:300] or "captcha" in html.lower()[:300]:
        return []
    out = []
    # 模式1：b_algo 结果块
    for b in re.findall(r'<li class="b_algo">(.*?)</li>', html, re.S)[:count]:
        t = re.sub(r"<[^>]+>", "", re.search(r'<h2[^>]*>(.*?)</h2>', b, re.S).group(1)).strip() \
            if re.search(r'<h2[^>]*>(.*?)</h2>', b, re.S) else ""
        c = re.sub(r"<[^>]+>", "", re.search(r'<p[^>]*>(.*?)</p>', b, re.S).group(1)).strip() \
            if re.search(r'<p[^>]*>(.*?)</p>', b, re.S) else ""
        if t or c:
            out.append({"title": t, "snippet": c})
    # 模式2：百科卡片（bingsource / entity）
    if not out:
        for m in re.finditer(r'<div class="b_ans[^"]*">(.*?)</div>', html, re.S):
            txt = re.sub(r"<[^>]+>", " ", m.group(1)).strip()
            if len(txt) > 10:
                out.append({"title": "百科", "snippet": re.sub(r"\s+", " ", txt)[:200]})
                break
    # 模式3：通用 <h2>
    if not out:
        for h in re.findall(r'<h2[^>]*>(.*?)</h2>', html, re.S)[:count]:
            t = re.sub(r"<[^>]+>", "", h).strip()
            if t:
                out.append({"title": t, "snippet": ""})
    return out


for q in ["狗", "量子纠缠", "秦始皇", "黑洞", "咖啡"]:
    res = search(q)
    print("「%s」: %d 条" % (q, len(res)))
    for r in res[:2]:
        print("   ", r["title"][:30], "|", r["snippet"][:50])
