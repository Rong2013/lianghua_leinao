# -*- coding: utf-8 -*-
"""metrics.py - lianghua 指标可观测性：Prometheus 文本导出器（零重依赖）。

# 生物依据：无，工程简化（可观测性是运维基础设施，不对应脑区）。
# 简化说明：
#   把训练/推理的关键指标暴露成 Prometheus 时序指标（文本格式，:8080/metrics 抓），
#   从"日志排查"升级为"实时监控"，支撑大规模部署。
#   零重依赖：纯 Python 输出 Prometheus 文本 exposition format（不用 prometheus_client）。
#   Grafana 直接配 Prometheus 抓取 http://host:8080/metrics 即可画时序面板。
# 参数来源：指标名/单位 = 工程简化；采样窗口 = 滑动 100 条（够画趋势，不爆内存）。
#
# 用法：
#   1. 训练/推理时采集：
#        import metrics
#        metrics.tick_seconds(0.42)          # 单 tick 耗时
#        metrics.saturation_rate(0.003)     # 膜电位饱和率
#        metrics.prune_count(225440359)     # 累计剪枝数
#        metrics.cache_hit_rate(0.846)      # 行缓存命中率
#   2. 起 HTTP 服务（:8080/metrics）：
#        python metrics.py --port 8080
#   3. Grafana 配 Prometheus 抓取该端点，画时序面板。
#
# 指标清单（全部 counter/gauge，单位在 _HELP 里）：
#   lianghua_tick_seconds_total        counter  累计 tick 耗时（秒）
#   lianghua_tick_seconds             gauge    最近一次 tick 耗时
#   lianghua_saturation_rate          gauge    膜电位饱和率（[0,1]）
#   lianghua_prune_count_total        counter  累计剪枝突触数
#   lianghua_cache_hit_ratio          gauge    行缓存命中率（[0,1]）
#   lianghua_train_epochs_total       counter  累计训练轮数
#   lianghua_think_calls_total        counter  累计推理次数
#   lianghua_accuracy                 gauge    最近一次评估准确率（[0,1]）
#   lianghua_timestamp_seconds        gauge    最近一次采集 Unix 时间戳
"""
import json
import os
import threading
import time
import http.server
import socketserver

# ------------------------------------------------------------------
# 指标存储（线程安全：GIL + 单锁）
# ------------------------------------------------------------------
_lock = threading.Lock()

# gauge（瞬时值）
_GAUGES = {
    "lianghua_tick_seconds": 0.0,
    "lianghua_saturation_rate": 0.0,
    "lianghua_cache_hit_ratio": 0.0,
    "lianghua_accuracy": 0.0,
    "lianghua_timestamp_seconds": 0.0,
}

# counter（累计值）
_COUNTERS = {
    "lianghua_tick_seconds_total": 0.0,
    "lianghua_prune_count_total": 0,
    "lianghua_train_epochs_total": 0,
    "lianghua_think_calls_total": 0,
}

# 指标帮助文本（Prometheus 的 HELP）
_HELP = {
    "lianghua_tick_seconds": "最近一次 tick 耗时（秒）",
    "lianghua_tick_seconds_total": "累计 tick 耗时（秒）",
    "lianghua_saturation_rate": "膜电位饱和率（[0,1]，越高越接近 INT8 顶格）",
    "lianghua_prune_count_total": "累计剪枝突触数（睡眠修剪）",
    "lianghua_cache_hit_ratio": "R+ 行缓存命中率（[0,1]）",
    "lianghua_train_epochs_total": "累计训练轮数",
    "lianghua_think_calls_total": "累计推理次数",
    "lianghua_accuracy": "最近一次评估准确率（[0,1]）",
    "lianghua_timestamp_seconds": "最近一次采集 Unix 时间戳",
}

# 采样窗口（画趋势用，滑动 100 条）
_MAX_SAMPLES = 100
_samples = []          # [(ts, snapshot_dict), ...]
_snap_lock = threading.Lock()


def _snap():
    with _lock:
        return dict(_GAUGES), dict(_COUNTERS)


def _record(gauges=None, counters=None):
    """记录一条采样（供 Grafana 画时序）。"""
    g, c = _snap()
    if gauges:
        g.update(gauges)
    if counters:
        c.update(counters)
    ts = time.time()
    g["lianghua_timestamp_seconds"] = ts
    with _snap_lock:
        _samples.append((ts, {"gauges": g, "counters": c}))
        if len(_samples) > _MAX_SAMPLES:
            _samples.pop(0)


# ------------------------------------------------------------------
# 采集 API（训练/推理代码里调）
# ------------------------------------------------------------------

def tick_seconds(t):
    """记录一次 tick 耗时（秒）：gauge 更新 + counter 累加 + 采样。"""
    with _lock:
        _GAUGES["lianghua_tick_seconds"] = float(t)
        _GAUGES["lianghua_timestamp_seconds"] = time.time()
        _COUNTERS["lianghua_tick_seconds_total"] += float(t)
    _record()


def saturation_rate(r):
    """膜电位饱和率（[0,1]）。"""
    with _lock:
        _GAUGES["lianghua_saturation_rate"] = float(r)
        _GAUGES["lianghua_timestamp_seconds"] = time.time()
    _record()


def prune_count(n):
    """累计剪枝突触数（counter 累加）。"""
    with _lock:
        _COUNTERS["lianghua_prune_count_total"] += int(n)
        _GAUGES["lianghua_timestamp_seconds"] = time.time()
    _record()


def cache_hit_ratio(r):
    """R+ 行缓存命中率（[0,1]）。"""
    with _lock:
        _GAUGES["lianghua_cache_hit_ratio"] = float(r)
        _GAUGES["lianghua_timestamp_seconds"] = time.time()
    _record()


def train_epoch():
    """训练完成一轮（counter +1）。"""
    with _lock:
        _COUNTERS["lianghua_train_epochs_total"] += 1
    _record()


def think_call():
    """推理一次（counter +1）。"""
    with _lock:
        _COUNTERS["lianghua_think_calls_total"] += 1
    _record()


def accuracy(a):
    """最近一次评估准确率（[0,1]）。"""
    with _lock:
        _GAUGES["lianghua_accuracy"] = float(a)
    _record()


# ------------------------------------------------------------------
# Prometheus 文本输出（exposition format）
# ------------------------------------------------------------------

def render():
    """当前指标快照 → Prometheus 文本格式。"""
    g, c = _snap()
    lines = []
    # gauge
    for k in ("lianghua_tick_seconds", "lianghua_saturation_rate",
              "lianghua_cache_hit_ratio", "lianghua_accuracy",
              "lianghua_timestamp_seconds"):
        lines.append("# HELP %s %s" % (k, _HELP[k]))
        lines.append("# TYPE %s gauge" % k)
        lines.append("%s %.6f" % (k, g.get(k, 0.0)))
    # counter
    for k in ("lianghua_tick_seconds_total", "lianghua_prune_count_total",
              "lianghua_train_epochs_total", "lianghua_think_calls_total"):
        lines.append("# HELP %s %s" % (k, _HELP[k]))
        lines.append("# TYPE %s counter" % k)
        lines.append("%s %s" % (k, c.get(k, 0)))
    return "\n".join(lines) + "\n"


def render_json():
    """当前指标快照 → JSON（供程序读取，非 Prometheus）。"""
    g, c = _snap()
    return {"gauges": g, "counters": c, "samples": len(_samples)}


# ------------------------------------------------------------------
# HTTP 服务（:8080/metrics，Prometheus 抓取）
# ------------------------------------------------------------------

class _MetricsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            body = render().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/metrics.json":
            body = json.dumps(render_json(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass   # 静默（不刷日志，守住"不吃 CPU"理念）


def serve(port=8080, host="0.0.0.0"):
    """起 HTTP 服务（后台线程，daemon）。返回端口。"""
    class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
    s = _Server((host, port), _MetricsHandler)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    return port


# ------------------------------------------------------------------
# CLI：python metrics.py [--port 8080] [--demo]
# ------------------------------------------------------------------

def _demo():
    """演示：采几个指标 + 渲染 + 起服务 1 秒，打印文本。"""
    print("=== metrics 演示（采几个指标 + 渲染）===")
    tick_seconds(0.42)
    saturation_rate(0.003)
    prune_count(225440359)
    cache_hit_ratio(0.846)
    train_epoch()
    think_call()
    accuracy(0.812)
    print(render())
    port = serve(8080)
    print("HTTP 服务已起在 :%d/metrics（daemon 线程，5s 后自动结束）" % port)
    time.sleep(5)
    print("演示结束")
    return 0


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="lianghua Prometheus 指标导出器（零重依赖）")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--demo", action="store_true", help="演示采集 + 起服务 5s")
    ap.add_argument("--json", action="store_true", help="打印当前指标 JSON 快照")
    a = ap.parse_args(argv)
    if a.demo:
        return _demo()
    if a.json:
        print(json.dumps(render_json(), ensure_ascii=False, indent=2))
        return 0
    # 默认：起服务（前台）
    print("lianghua 指标服务：http://0.0.0.0:%d/metrics（Ctrl+C 退出）" % a.port)
    port = serve(a.port)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n已退出")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
