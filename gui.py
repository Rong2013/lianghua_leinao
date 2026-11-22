# -*- coding: utf-8 -*-
"""gui.py - lianghua 量化类脑网页 GUI（零依赖，纯 Python HTTP 服务）。

# 生物依据：无，工程简化（GUI 是交互壳，不对应脑区）。
# 简化说明：
#   - 起一个本地 HTTP 服务（:8000），浏览器打开就能交互。
#   - 零第三方依赖：纯 Python http.server + 手写 JSON API（不用 Flask）。
#   - 前端 index.html：聊天式对话（用户说 → 脑回话）+ 在线学习（加新语料）+ 知识库边界。
#   - 把"训练好的量化脑"加载进来，实时展示"脑在想什么、能聊什么、不会聊什么"。
# 参数来源：端口 8000（可 --port 改）。工程简化。
#
# 用法：
#   python gui.py --brain demo_brain.npz              # 起 :8000，浏览器打开
#   python gui.py --brain demo_brain.npz --port 8080
#   浏览器：http://localhost:8000
#
# API：
#   GET  /api/meta         脑规模/原型数/内存
#   GET  /api/chat?text=狗  对话（识别话题 + 脑回话 + 置信度 + 是否越界）
#   GET  /api/dataset      训练集/测试集/评估结果（能力范围）
#   GET  /api/corpus       当前知识库（标签→训练语料）
#   POST /api/learn        在线学习（加新标签/新语料，更新脑+语料文件）
#   POST /api/corpus       更新/删除语料（可选）
"""
import os
import sys
import json
import time
import argparse
import http.server
import socketserver
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lianghua
import encoder
import visualizer
import hopfield
import webfetch

# 置信度阈值：低于此 score 视为"没学过，无法可靠识别"（诚实边界，不瞎猜）
# 调低到 0.15：覆盖短问候/近义措辞（"你好啊"0.206、"hi"0.269、"嗨 好久不见"0.139），
# 同时"量子纠缠"0.06 等知识库外的仍判越界。0.30 太严（误杀短问候）。
CONFIDENCE_THRESHOLD = 0.15
# 越界回复（知识库外的统一提示）
OOB_REPLY = "这个我没学过，不在我的知识范围内。用「在线学习」教我（标签 + 语料），我就能聊这个话题了。"


def build_raster_data(brain, text, ticks=6, max_neurons=64, n_sub=None):
    """生成栅格图数据（JSON 前端画 Canvas）。"""
    sub = n_sub if n_sub is not None else 0
    seeds, wts = encoder.text_to_seeds(text, brain.N, salt=0)
    if len(seeds) == 0:
        return {"raster": [], "n_cols": 0, "neuron_ids": []}
    n_cols = ticks
    raster = np.zeros((brain.N, n_cols), dtype=bool)
    cur_s, cur_w = seeds, wts
    for t in range(ticks):
        f, sync, fired = brain.fire(sub, cur_s, cur_w, tick=t)
        if len(fired):
            raster[fired, t] = True
        if f == 0:
            cur_s, cur_w = seeds, wts
        else:
            nb = brain.neighbors[sub][fired].reshape(-1)
            ww = brain.weights[sub][fired].reshape(-1)
            pos = ww > 0
            cur_s, cur_w = nb[pos], ww[pos]
            if len(cur_s) == 0:
                cur_s, cur_w = seeds, wts
    counts = raster.sum(axis=1)
    order = np.argsort(-counts)[:max_neurons]
    order = np.sort(order)
    show = raster[order]
    return {
        "raster": show.tolist(),
        "neuron_ids": [int(x) for x in order],
        "n_cols": n_cols,
        "ticks": ticks,
        "sub": int(sub),
    }


def pick_reply(brain, label):
    """从标签的训练语料里选一句最相关的回复（按与"该标签原型"的 Jaccard 最高）。

    # 生物依据：海马重放——记忆提取时调取该话题的代表性模式。
    # 简化说明：texts[label] 是该标签学过的所有语料句；取与该标签"增强放电模式"
    #   Jaccard 最高的一句（最典型的代表句），而非简单轮换。
    # 参数来源：按 Jaccard 选代表句（工程简化）。
    """
    texts = brain.texts.get(label, [])
    if not texts:
        return "", -1
    # 用该标签增强放电模式（原型的并集）当"代表模式"
    proto = brain.hopfield.attractors[brain.hopfield.labels.index(label)] \
        if getattr(brain, "hopfield", None) and label in brain.hopfield.labels else None
    if proto is None:
        # 没挂 Hopfield 时退化为轮换
        if not hasattr(brain, "_reply_idx"):
            brain._reply_idx = {}
        i = brain._reply_idx.get(label, 0)
        brain._reply_idx[label] = (i + 1) % max(1, len(texts))
        return texts[i], i
    import encoder
    import numpy as np
    best_text, best_score = texts[0], -1.0
    for t in texts:
        seeds, _ = encoder.text_to_seeds(t, brain.N, salt=0)
        if not len(seeds):
            continue
        s = np.zeros(brain.N, dtype=np.int8)
        s[seeds] = 1
        inter = int(np.sum(s & proto))
        union = int(np.sum(s | proto)) or 1
        score = inter / union
        if score > best_score:
            best_text, best_score = t, score
    return best_text, 0


def compute_math(text):
    """尝试把文本里的"数字 运算符 数字"算出来（答"几"类问题）。

    # 生物依据：无，工程简化（脑不真做算术，但能识别算式并给答案）。
    # 简化说明：正则抓 \\d+\\s*[+\\-×*]\\s*\\d+，求值返回"X"。
    # 参数来源：只支持 + - × / 四则，工程简化。
    """
    import re
    m = re.search(r"(\d+)\s*([+\-×*])\s*(\d+)", text)
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    if op == "+":
        return str(a + b)
    if op == "-":
        return str(a - b)
    if op in ("×", "*"):
        return str(a * b)
    return None


def make_handler(brain, args):
    class Handler(http.server.BaseHTTPRequestHandler):
        _netmode_cache = None
        _netmode_cache_ts = 0

        def log_message(self, fmt, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_body(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return {}

        def do_GET(self):
            import urllib.parse
            if self.path in ("/", "/index.html"):
                p = os.path.join(HERE, "index.html")
                if os.path.exists(p):
                    self._send(200, open(p, "rb").read(), "text/html; charset=utf-8")
                else:
                    self._send(404, "index.html missing")
            elif self.path.startswith("/api/chat"):
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                text = q.get("text", [""])[0]
                mode = q.get("mode", ["jaccard"])[0]   # jaccard / hopfield
                if mode == "hopfield" and getattr(brain, "hopfield", None) is not None:
                    # Hopfield 收敛识别（模式完成）
                    r = brain.hopfield.recognize(text, top_k=4, ticks=6)
                    label, score, conv = r["label"], r["score"], r["converged"]
                    iters = r["iters"]
                else:
                    # 默认 Jaccard 一次匹配
                    r = brain.think(text, top_k=4, ticks=6, salt=args.salt)
                    label, score, conv = r["label"], r["score"], True
                    iters = 0
                in_kb = label is not None and score >= CONFIDENCE_THRESHOLD
                # 数学算式：不管识别置信度，只要输入是"数字 运算符 数字"就答答案
                math_ans = compute_math(text)
                if math_ans and label and ("数学" in label or "算术" in label):
                    # 识别到数学话题 → 直接答
                    reply = "%s = %s" % (text.strip(), math_ans)
                    in_kb = True   # 算式能答就算在知识库内
                elif math_ans:
                    # 没识别到数学但输入是算式 → 仍答（算式本身是确定的）
                    reply = "%s = %s" % (text.strip(), math_ans)
                    in_kb = True
                elif in_kb:
                    # 非算式：回该标签最相关的语料
                    reply, _ = pick_reply(brain, label)
                else:
                    reply = OOB_REPLY
                rd = build_raster_data(brain, text, ticks=6)
                out = {
                    "text": text,
                    "mode": mode,
                    "label": label if in_kb else None,
                    "raw_label": label,
                    "score": round(score, 4),
                    "converged": bool(conv),
                    "iters": iters,
                    "in_knowledge_base": in_kb,
                    "spikes": r.get("spikes", 0) if "spikes" in r else 0,
                    "subs": r.get("subs", []) if "subs" in r else [],
                    "per_sub": {str(k): v for k, v in r.get("per_sub", {}).items()},
                    "reply": reply,
                    "oob_reply": not in_kb,
                    "threshold": CONFIDENCE_THRESHOLD,
                    "raster": rd,
                }
                self._send(200, json.dumps(out, ensure_ascii=False))
            elif self.path.startswith("/api/think"):  # 兼容旧接口
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                text = q.get("text", [""])[0]
                r = brain.think(text, top_k=4, ticks=6, salt=args.salt)
                rd = build_raster_data(brain, text, ticks=6)
                out = {
                    "text": text, "label": r["label"], "score": r["score"],
                    "spikes": r["spikes"], "subs": r["subs"],
                    "per_sub": {str(k): v for k, v in r["per_sub"].items()},
                    "top": r["top"][:16], "raster": rd,
                }
                self._send(200, json.dumps(out, ensure_ascii=False))
            elif self.path == "/api/meta":
                self._send(200, json.dumps({
                    "nsub": brain.NSUB, "n": brain.N, "k": brain.K,
                    "protos": len(brain.protos), "neurons": brain.NSUB * brain.N,
                    "labels": sorted(brain.texts.keys()),
                    "mem_mb": round((brain.state.nbytes + brain.weights.nbytes +
                                     brain.neighbors.nbytes + brain.last_spike.nbytes) / 1048576, 3),
                    "threshold": CONFIDENCE_THRESHOLD,
                }, ensure_ascii=False))
            elif self.path == "/api/dataset":
                eval_path = os.path.join(HERE, "demo_eval.json")
                if os.path.exists(eval_path):
                    with open(eval_path, encoding="utf-8") as f:
                        self._send(200, open(eval_path, encoding="utf-8").read())
                else:
                    self._send(404, json.dumps({"error": "demo_eval.json missing"}))
            elif self.path == "/api/corpus":
                # 返回当前知识库（标签 → 语料）
                corpus = {k: v for k, v in brain.texts.items()}
                self._send(200, json.dumps(corpus, ensure_ascii=False))
            elif self.path == "/api/netmode":
                # 各网络源可达性（GUI 启动时显示"在线/离线"）
                # 缓存在类级别：测一次（可能 9s），后续秒回
                if not hasattr(Handler, "_netmode_cache") or Handler._netmode_cache_ts == 0:
                    Handler._netmode_cache = webfetch.test_connectivity()
                    Handler._netmode_cache_ts = time.time()
                self._send(200, json.dumps(Handler._netmode_cache, ensure_ascii=False))
            elif self.path.startswith("/api/lookup"):
                # 联网获取信息（分层源 + 缓存 + 降级）
                import urllib.parse
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                text = q.get("text", [""])[0].strip()
                label_hint = q.get("label", [None])[0]
                if not text:
                    self._send(400, json.dumps({"error": "text 必填"}))
                    return
                r = webfetch.lookup(text)
                found = r["found"]
                out = {
                    "query": text,
                    "text": r["text"],
                    "source": r["source"],          # wikipedia/baidu/duckduckgo/offline/none
                    "offline": r["offline"],
                    "cached": r["cached"],
                    "found": found,
                }
                # 找到了 + 用户给标签提示 → 顺便教脑（在线学习，扩知识库）
                if found and label_hint:
                    lr = brain.train_samples([(label_hint, r["text"])],
                                             epochs=2, ticks=6, verbose=False)
                    brain.save(args.brain)
                    out["taught"] = True
                    out["taught_label"] = label_hint
                # 找到了但没标签提示 → 给个建议标签（脑当前最接近的）
                elif found and not label_hint:
                    probe = brain.think(r["text"][:20], top_k=4, ticks=6)
                    out["suggest_label"] = probe["label"]
                self._send(200, json.dumps(out, ensure_ascii=False))
            else:
                self._send(404, "not found")

        def do_POST(self):
            if self.path == "/api/learn":
                body = self._read_body()
                label = str(body.get("label", "")).strip()
                texts = [str(t).strip() for t in body.get("texts", []) if str(t).strip()]
                if not label or not texts:
                    self._send(400, json.dumps({"error": "label 和 texts 都必填"}))
                    return
                # 记录训练前权重（算 STDP 改了多少突触）
                before = brain.weights.copy()
                # 在线学习：把新语料喂进脑（STDP 强化），并持久化
                new_samples = [(label, t) for t in texts]
                brain.train_samples(new_samples, epochs=2, ticks=6, verbose=False)
                after = brain.weights
                # 算改了哪些子脑/多少突触（训练可视化）
                changed_mask = (before != after)
                per_sub_changed = int(changed_mask.reshape(brain.NSUB, -1).sum(axis=1).astype(bool).sum())
                total_changed = int(changed_mask.sum())
                # 学完立刻识别第一条（看脑是否真学会了）
                probe = brain.think(texts[0], top_k=4, ticks=6)
                brain.save(args.brain)  # 更新脑文件
                # 更新语料文件
                corpus_path = os.path.join(HERE, "demo_samples.tsv")
                _append_corpus(corpus_path, new_samples)
                # 重新评估（如果测试集存在）
                eval_path = os.path.join(HERE, "demo_eval.json")
                test_path = os.path.join(HERE, "demo_samples_test.tsv")
                extra = {}
                if os.path.exists(test_path):
                    import cli
                    test = cli.load_samples(test_path)
                    if test:
                        out = cli._eval(brain, test, top_k=4, ticks=6, salt=args.salt)
                        extra = {"retest_accuracy": out["accuracy"],
                                 "retest_correct": out["correct"],
                                 "retest_total": out["samples"]}
                        _update_eval(eval_path, brain, test, out)
                self._send(200, json.dumps({
                    "ok": True, "label": label, "added": len(texts),
                    "total_labels": len(brain.texts),
                    # 训练可视化（群主要看"怎么训的"）
                    "train_detail": {
                        "synapses_changed": total_changed,   # STDP 改了这么多突触
                        "subs_active": per_sub_changed,       # 这些子脑参与了学习
                        "probe_label": probe["label"],        # 学完立刻识别第一条 → 标签
                        "probe_score": probe["score"],         # 置信度
                        "probe_spikes": probe["spikes"],       # 放电数
                    },
                    **extra,
                }, ensure_ascii=False))
            else:
                self._send(404, "not found")

    return Handler


def _append_corpus(path, samples):
    """把新语料追加进 TSV（label\ttext，同 label 不重复行）。"""
    existing = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.rstrip("\n\r")
                if line.strip() and "\t" in line:
                    existing.add(line)
    with open(path, "a", encoding="utf-8") as f:
        for label, text in samples:
            rec = "%s\t%s" % (label, text)
            if rec not in existing:
                f.write(rec + "\n")
                existing.add(rec)


def _update_eval(eval_path, brain, test, out):
    """把重新评估结果写回 demo_eval.json。"""
    data = {}
    if os.path.exists(eval_path):
        try:
            with open(eval_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data["accuracy"] = out["accuracy"]
    data["correct"] = out["correct"]
    data["total"] = out["samples"]
    data["avg_score"] = out.get("avg_score", 0.0)
    data["train_samples"] = [l for l in brain.texts.items()]  # 占位，下面覆盖
    data["test_samples"] = test
    data["labels"] = sorted(brain.texts.keys())
    # 训练集：按 label 展开
    train_samples = []
    for label, ts in brain.texts.items():
        for t in ts:
            train_samples.append([label, t])
    data["train_samples"] = train_samples
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def serve(port=8000):
    class Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
    brain = lianghua.QuantizedBrain.load(args.brain)
    # 挂 Hopfield 网络（联想记忆 + 模式完成，可选识别方式）
    try:
        hf = hopfield.attach(brain)
        brain.hopfield = hf
        print("Hopfield 网络：吸引子 %d 个，W_per 矩阵 %s（INT8，对称零对角线）"
              % (len(hf.labels), next(iter(hf.W_per.values())).shape if hf.W_per else 'N/A'))
    except Exception as e:
        print("Hopfield 挂载失败（不影响 Jaccard 识别）：", e)
        brain.hopfield = None
    handler = make_handler(brain, args)
    srv = Srv(("0.0.0.0", port), handler)
    print("=== lianghua 量化类脑网页 GUI（对话式 + 在线学习）===")
    print("脑：%d 子脑 × %d 神经元 = %d 神经元，标签 %d 个，内存 %.3f MB"
          % (brain.NSUB, brain.N, brain.NSUB * brain.N, len(brain.texts),
             (brain.state.nbytes + brain.weights.nbytes +
              brain.neighbors.nbytes + brain.last_spike.nbytes) / 1048576))
    print("知识库:", ", ".join(sorted(brain.texts.keys())))
    print("置信度阈值 %.2f（低于此 score 视为'没学过'，不瞎猜）" % CONFIDENCE_THRESHOLD)
    print("浏览器打开：http://localhost:%d  （Ctrl+C 退出）" % port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="lianghua 量化类脑网页 GUI（零依赖，对话式）")
    ap.add_argument("--brain", default=os.path.join(HERE, "demo_brain.npz"),
                    help="训练好的 .npz（默认 demo_brain.npz）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--salt", type=int, default=0)
    args = ap.parse_args()
    if not os.path.exists(args.brain):
        print("缺脑文件 %s（先 python train_demo.py 或 cli.py build+train）" % args.brain)
        sys.exit(1)
    sys.exit(serve(args.port))
