# -*- coding: utf-8 -*-
"""webfetch.py - lianghua 量化类脑联网获取信息（多源分层 + 缓存 + 可降级）。

# 生物依据：海马 + 新皮层"系统记忆"——工作记忆（脑内）不够时，外接
#   "外存"（网络）查新信息，再回写脑（在线学习）。
# 简化说明（全离线可用，不假装全能）：
#   - 分层源：依次尝试 ① 本地词典（离线兜底）② 网络源（Wikipedia/Baidu/...）
#   - 缓存：查到一次存本地 JSON，下次直接读（不吃网络）
#   - 降级：所有网络源都连不上时，回退本地词典 + 明确提示"离线模式"
#   - 默认关：不挂 cli 默认路径；GUI 里显式点"联网"才查，不破坏已验证指标。
# 参数来源：
#   - 超时 10s/源（避免卡死），缓存目录 viz/webcache/（工程简化）。
#   - 本地词典 demo_knowledge.json 随仓库带（离线兜底，群主可扩）。
"""
import os
import json
import time
import socket
import urllib.request
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "viz", "webcache")
OFFLINE_DICT = os.path.join(HERE, "demo_knowledge.json")
TIMEOUT = 5            # 单源超时（秒，短一点避免卡死 GUI 请求）
UA = "Mozilla/5.0 (lianghua-webfetch)"
CACHE_TTL = 3600       # 缓存有效期（秒，1 小时；过期重新联网，避免固化旧结果）
# 全局 socket 超时（任何 urlopen 都不超过此值，防 GUI 请求挂死）
socket.setdefaulttimeout(5)


def _cache_path(query):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = "".join(c for c in query if c.isalnum() or c in "_-") or "blank"
    return os.path.join(CACHE_DIR, key + ".json")


def _cache_get(query):
    """读缓存（带 TTL 检查）。命中且未过期返回 dict，否则 None。"""
    cp = _cache_path(query)
    if not os.path.exists(cp):
        return None
    try:
        with open(cp, encoding="utf-8") as f:
            data = json.load(f)
        ts = data.get("ts", 0)
        if time.time() - ts > CACHE_TTL:
            return None   # 过期，重新联网
        return data
    except Exception:
        return None


def _load_offline_dict():
    """本地词典（离线兜底）：query -> {label, text, source}。"""
    if not os.path.exists(OFFLINE_DICT):
        return {}
    try:
        with open(OFFLINE_DICT, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return {}


def _try_bing(query):
    """源①：Bing 搜索（本机可达，抓结果标题=高质量信息）。

    # 生物依据：无，工程简化。
    # 简化说明：本机 HTTPS 能通 Bing（Wikipedia/DDG 超时），Bing 反爬但标题可抓。
    #   标题常含百度百科定义（如"狗（犬科哺乳动物）"），当"联网获取的信息"。
    # 参数来源：count=5、setlang=zh-CN、超时 15s（工程简化）。
    """
    import re
    url = "https://www.bing.com/search?" + urllib.parse.urlencode(
        {"q": query, "count": "5", "setlang": "zh-CN", "cc": "cn"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")
    except Exception:
        return None
    if "验证" in html[:300] or "captcha" in html.lower()[:300]:
        return None
    # 抓 b_algo 结果标题（含百度百科定义，高质量）
    titles = []
    for b in re.findall(r'<li class="b_algo">(.*?)</li>', html, re.S)[:5]:
        m = re.search(r'<h2[^>]*>(.*?)</h2>', b, re.S)
        if m:
            t = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            if t:
                titles.append(t)
    if not titles:
        for h in re.findall(r'<h2[^>]*>(.*?)</h2>', html, re.S)[:5]:
            t = re.sub(r"<[^>]+>", "", h).strip()
            if t:
                titles.append(t)
    if titles:
        return " ".join(titles[:3]), "bing"
    return None


def _try_wikipedia(query):
    """源：Wikipedia REST API（中文）。本机常超时，作备用。"""
    url = "https://zh.wikipedia.org/api/rest_v1/page/summary/" + \
          urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=TIMEOUT).read())
        text = (d.get("extract") or d.get("content") or "").strip()
        if text:
            return text, "wikipedia"
    except Exception:
        return None
    return None


# 分层源顺序（在所有源函数定义后再赋值，见 _try_duckduckgo 之后）


def _try_baidu(query):
    """源②：百度搜索结果（抓标题/摘要）。受限网络可能返回安全验证页 → 降级。"""
    url = "http://www.baidu.com/s?" + urllib.parse.urlencode({"wd": query})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        import re
        html = urllib.request.urlopen(req, timeout=TIMEOUT).read().decode("utf-8", "ignore")
        # 检测安全验证页（反爬）→ 返回 None（降级）
        if "安全验证" in html or "百度安全" in html:
            return None
        # 抓结果块（简化：找 class=result 或 data-log）
        titles = re.findall(r'<h3[^>]*>(.*?)</h3>', html)
        titles = [re.sub(r"<[^>]+>", "", t).strip() for t in titles]
        if titles:
            return " | ".join(titles[:5]), "baidu"
    except Exception:
        return None
    return None


def _try_duckduckgo(query):
    """源③：DuckDuckGo 简化 API。"""
    url = "https://api.duckduckgo.com/?q=" + urllib.parse.quote(query) + \
          "&format=json&no_html=1"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=TIMEOUT).read())
        parts = []
        if d.get("AbstractText"):
            parts.append(d["AbstractText"])
        if d.get("Heading"):
            parts.append(d["Heading"])
        if d.get("Answer"):
            parts.append(d["Answer"])
        text = " ".join(parts).strip()
        if text:
            return text, "duckduckgo"
    except Exception:
        return None
    return None


# 分层源顺序（本机可达优先）：Bing → Wikipedia → DuckDuckGo → 离线词典
SOURCES = [
    ("bing", _try_bing),
    ("wikipedia", _try_wikipedia),
    ("duckduckgo", _try_duckduckgo),
]


def lookup(query, use_cache=True, allow_net=True):
    """联网获取信息（分层源 + 缓存 + 降级）。

    返回 dict：{
        "query": str,
        "text": str,            # 获取到的信息（可能是空）
        "source": str,          # "wikipedia" / "baidu" / "duckduckgo" / "offline" / "none"
        "offline": bool,        # True = 网络不可用，用了离线词典兜底
        "cached": bool,         # True = 命中本地缓存
        "found": bool,          # 是否真的拿到了信息
    }
    """
    q = str(query).strip()
    # 1. 缓存优先（带 TTL：未过期秒回，过期重新联网）
    cp = _cache_path(q)
    if use_cache:
        cached = _cache_get(q)
        if cached:
            return {"query": q, "text": cached.get("text", ""),
                    "source": cached.get("source", "cache"),
                    "offline": False, "cached": True,
                    "found": bool(cached.get("text"))}

    # 2. 联网源（Bing → Wikipedia → DDG，本机可达就用真联网）
    if allow_net:
        for name, fn in SOURCES:
            r = fn(q)
            if r:
                text, src = r
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(cp, "w", encoding="utf-8") as f:
                    json.dump({"query": q, "text": text, "source": src,
                               "ts": time.time()}, f, ensure_ascii=False)
                return {"query": q, "text": text, "source": src,
                        "offline": False, "cached": False, "found": True}

    # 3. 联网全失败 → 离线词典兜底（最后手段）
    d = _load_offline_dict()
    hit = d.get(q)
    if not hit:
        for k, v in d.items():
            if k in q or q in k:
                hit = v
                break
    if hit:
        return {"query": q, "text": hit.get("text", ""),
                "source": "offline", "offline": True,
                "cached": False, "found": True}

    # 4. 啥都没有
    return {"query": q, "text": "", "source": "none",
            "offline": not allow_net, "cached": False, "found": False}


def test_connectivity():
    """快速测各源可达性（GUI 启动时显示"当前网络模式"）。

    返回 {name: bool}。用短超时（3s）避免 GUI 请求挂死。
    """
    out = {}
    for name, _ in SOURCES:
        try:
            if name == "bing":
                req = urllib.request.Request(
                    "https://www.bing.com/search?q=test",
                    headers={"User-Agent": UA})
                urllib.request.urlopen(req, timeout=5).read()
                out[name] = True
            elif name == "wikipedia":
                req = urllib.request.Request(
                    "https://zh.wikipedia.org/api/rest_v1/page/summary/Test",
                    headers={"User-Agent": UA})
                urllib.request.urlopen(req, timeout=3).read()
                out[name] = True
            elif name == "duckduckgo":
                req = urllib.request.Request(
                    "https://api.duckduckgo.com/?q=test&format=json",
                    headers={"User-Agent": UA})
                urllib.request.urlopen(req, timeout=3).read()
                out[name] = True
        except Exception:
            out[name] = False
    return out


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "狗"
    print("=== 联网获取：", q, "===")
    conn = test_connectivity()
    print("网络可达性:", conn)
    r = lookup(q)
    print("结果:", json.dumps(r, ensure_ascii=False, indent=2))
