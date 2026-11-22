# -*- coding: utf-8 -*-
"""test_lookup_speed.py - 验证联网获取（离线词典命中应秒回）。"""
import urllib.request
import urllib.parse
import json
import time

BASE = "http://localhost:8000"


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=30).read())


def post(path, payload):
    req = urllib.request.Request(BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def main():
    print("=== 联网获取速度（离线词典命中应 <1s）===")
    for q in ["老虎", "秦始皇", "咖啡"]:
        t0 = time.time()
        r = get("/api/lookup?text=" + urllib.parse.quote(q))
        print("  %-8s found=%s source=%s 耗时 %.2fs" % (q, r["found"], r["source"], time.time()-t0))

    print("\n=== 新标签（不靠 prompt，用输入框）===")
    lr = post("/api/learn", {"label": "地理", "texts": ["长江 黄河", "青藏高原"]})
    print("  学「地理」: ok=%s taught_detail=%s total_labels=%s"
          % (lr.get("ok"), lr.get("train_detail"), lr.get("total_labels")))
    # 学了之后查"地理"
    r = get("/api/chat?text=" + urllib.parse.quote("长江 黄河"))
    print("  学后 chat「长江 黄河」-> %s (in_kb=%s)" % (r.get("label"), r["in_knowledge_base"]))
    m = get("/api/meta")
    print("  当前标签:", m["labels"])


if __name__ == "__main__":
    main()
