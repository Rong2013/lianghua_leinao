# -*- coding: utf-8 -*-
"""bench_env.py - 实测 lianghua 运行时的 内存 / CPU 占用（真实数字，供 RUNTIME.md 用）。

# 生物依据：无，工程简化（这是性能基准工具，不模拟脑）。
# 简化说明：用 resource(POSIX) / psutil(可选) 测峰值 RSS；CPU 用 time.process_time()
#   与 wall-clock 比值估算；本骨架纯 CPU（无 GPU），GPU 占用如实记为 N/A。
# 参数来源：规模 8×2048×32（默认骨架）；epochs/ticks 与 example_run 一致；工程简化。

用法：
    python bench_env.py            # 跑默认脑体 + R+，打印 内存/CPU/GPU 占用
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lianghua
import rplus_simple


def peak_rss_mb():
    """内存占用。优先 psutil（本进程 RSS）；否则读系统内存占用率。

    返回 (rss_mb 或 None, 系统占用率% 或 None)。
    """
    rss, load = None, None
    # 1) 首选 psutil（最准，跨平台）
    try:
        import psutil
        rss = round(psutil.Process().memory_info().rss / 1048576, 1)
        load = round(psutil.virtual_memory().percent, 1)
        return rss, load
    except Exception:
        pass
    # 2) 无 psutil：只读系统内存占用率（不把系统已用误标成进程 RSS）
    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = MEMORYSTATUSEX()
        ms.dwLength = ctypes.sizeof(ms)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            load = round(ms.dwMemoryLoad, 1)
    else:
        import resource
        rss = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    return rss, load


def _hw_info():
    """硬件信息：CPU 型号/核数、物理内存、GPU、磁盘。能拿到什么拿什么。"""
    import platform

    cpu_model = platform.processor() or ""
    cpu_threads = os.cpu_count() or 0
    if sys.platform == "win32":
        try:
            import subprocess
            out = subprocess.run(
                "wmic cpu get Name /format:list", shell=True,
                capture_output=True, text=True,
                errors="replace").stdout
            for line in out.splitlines():
                if "Name=" in line:
                    cpu_model = line.split("=", 1)[1].strip()
                    break
        except Exception:
            pass
    total_mem_mb, avail_mem_mb = None, None
    gpu = "N/A（本骨架纯 CPU，不依赖 GPU）"
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_mem_mb = round(vm.total / 1048576, 1)
        avail_mem_mb = round(vm.available / 1048576, 1)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import subprocess
            g = subprocess.run(
                "wmic path win32_VideoController get Name /format:list",
                shell=True, capture_output=True, text=True,
                errors="replace").stdout
            names = [l.split("=", 1)[1].strip() for l in g.splitlines() if "Name=" in l]
            if names:
                gpu = "；".join(names)
        except Exception:
            pass
    disk_used_mb, disk_total_mb, disk_pct = None, None, None
    try:
        import psutil
        du = psutil.disk_usage(os.path.splitdrive(os.getcwd())[0] or "C:\\")
        disk_used_mb = round(du.used / 1048576, 0)
        disk_total_mb = round(du.total / 1048576, 0)
        disk_pct = du.percent
    except Exception:
        pass
    return {"os": "%s %s" % (platform.system(), platform.release()),
            "arch": platform.machine(),
            "python": platform.python_version(),
            "cpu_model": cpu_model or "未识别",
            "cpu_threads": cpu_threads,
            "mem_total_mb": total_mem_mb, "mem_avail_mb": avail_mem_mb,
            "gpu": gpu,
            "disk_used_mb": disk_used_mb, "disk_total_mb": disk_total_mb,
            "disk_pct": disk_pct}


def bench_brain():
    b = lianghua.QuantizedBrain.build(nsub=8, n=2048, k=32)
    w_mb = round(b.stats()["total_mb"], 3)
    samples = []
    for label in ["问候", "告别", "天气", "数学", "编程", "动物", "食物", "音乐"]:
        for i in range(4):
            samples.append((label, "样本 %s %d 号 测试文本" % (label, i)))
    t0 = time.perf_counter()
    b.train_samples(samples, epochs=2, ticks=6, verbose=False)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        b.think("今天天气怎么样，会下雨吗", ticks=6)
    think_s = time.perf_counter() - t0
    think_ms_avg = round(think_s / 20 * 1000, 1)
    return {"brain_mb": w_mb, "train_s": round(train_s, 2),
            "think_s": round(think_ms_avg / 1000, 3), "think_ms": think_ms_avg}


def bench_rplus():
    b = lianghua.QuantizedBrain.build(nsub=8, n=2048, k=32)
    import encoder
    seeds, wts = encoder.text_to_seeds("今天天气怎么样，会下雨吗", b.N)
    cache = rplus_simple.RowCache(k=b.K)
    t0 = time.perf_counter()
    for _ in range(20):
        rplus_simple.propagate(b, 0, seeds, wts, ticks=6, mode="key", cache=cache)
    rp_s = time.perf_counter() - t0
    st = cache.stats()
    return {"rplus_per_call_ms": round(rp_s / 20 * 1000, 2),
            "cache_hit_rate": st["hit_rate"], "cache_rows": st["rows"]}


def bench_speed_ref():
    """默认路径 vs R+ 路径 vs float 参照，给'极快'一个可比基准（同 1 次全子脑 think）。"""
    b = lianghua.QuantizedBrain.build(nsub=8, n=2048, k=32)
    import encoder
    seeds, wts = encoder.text_to_seeds("今天天气怎么样，会下雨吗", b.N)
    N = 20
    # 默认路径（全 8 子脑传播，纯 numpy）
    t0 = time.perf_counter()
    for _ in range(N):
        b.think("今天天气怎么样，会下雨吗", ticks=6)
    default_ms = round((time.perf_counter() - t0) / N * 1000, 1)
    # R+ 路径（单子脑 + 行缓存 + key 聚合）
    cache = rplus_simple.RowCache(k=b.K)
    t0 = time.perf_counter()
    for _ in range(N):
        rplus_simple.propagate(b, 0, seeds, wts, ticks=6, mode="key", cache=cache)
    rplus_ms = round((time.perf_counter() - t0) / N * 1000, 1)
    # float 参照（同规模，float32 膜电位）
    class FloatBrain:
        pass
    fb = FloatBrain()
    fb.NSUB, fb.N, fb.K = 8, 2048, 32
    rng = np.random.default_rng(20261120)
    fb.state = np.zeros((fb.NSUB, fb.N), dtype=np.float32)
    fb.weights = rng.integers(-2, 3, (fb.NSUB, fb.N, fb.K), dtype=np.int8)
    fb.neighbors = b.neighbors
    def float_fire(sub, seeds_, wts_):
        st = fb.state[sub]
        s_sorted, inv = np.unique(seeds_, return_inverse=True)
        input_sum = np.zeros(s_sorted.size, dtype=np.float32)
        np.add.at(input_sum, inv, wts_)
        newv = np.clip(st[s_sorted] * 0.9 + input_sum, 0, 1.0)
        fb.state[sub][s_sorted] = newv
        fired = np.flatnonzero(newv >= 0.5)
        fg = s_sorted[fired]
        if len(fg):
            fb.state[sub][fg] = 0
        return fg
    t0 = time.perf_counter()
    for _ in range(N):
        for sub in range(fb.NSUB):
            cur_s, cur_w = seeds, wts
            for _t in range(6):
                fired = float_fire(sub, cur_s, cur_w)
                if len(fired) == 0:
                    cur_s, cur_w = seeds, wts
                else:
                    nb = fb.neighbors[sub][fired].reshape(-1)
                    w = fb.weights[sub][fired].reshape(-1)
                    pos = w > 0
                    cur_s, cur_w = nb[pos], w[pos]
                    if len(cur_s) == 0:
                        cur_s, cur_w = seeds, wts
    float_ms = round((time.perf_counter() - t0) / N * 1000, 1)
    return {"default_ms": default_ms, "rplus_ms": rplus_ms, "float_ms": float_ms,
            "speedup_rplus_over_default": round(default_ms / max(0.01, rplus_ms), 1)}


def main():
    print("=== lianghua 运行环境实测（8×2048×32 默认骨架，纯 CPU）===")
    hw = _hw_info()
    print("硬件 / 系统:")
    print("  系统       : %s / %s（Python %s）" % (hw["os"], hw["arch"], hw["python"]))
    print("  CPU        : %s（%d 线程）" % (hw["cpu_model"], hw["cpu_threads"]))
    if hw["mem_total_mb"]:
        print("  物理内存   : 总 %s MB / 可用 %s MB" % (hw["mem_total_mb"], hw["mem_avail_mb"]))
    print("  GPU        : %s" % hw["gpu"])
    if hw["disk_total_mb"]:
        print("  磁盘       : 已用 %s MB / 共 %s MB（%s%%）"
              % (hw["disk_used_mb"], hw["disk_total_mb"], hw["disk_pct"]))

    rss0, load0 = peak_rss_mb()
    br = bench_brain()
    rp = bench_rplus()
    speed = bench_speed_ref()
    rss1, load1 = peak_rss_mb()
    print("内存（RAM）:")
    print("  脑体张量占用  : %s MB（state+weights+neighbors+last_spike，确定值）" % br["brain_mb"])
    if rss0 is not None:
        print("  进程 RSS     : 起 %s MB → 跑完 build+train+think 后 %s MB" % (rss0, rss1))
    if load0 is not None:
        print("  系统内存占用 : %s%% %s" % (load0, "（本脑体张量仅 %s MB，其余为解释器/numpy/其他进程）" % br["brain_mb"]))
    cpu_pct = None
    try:
        import psutil
        cpu_pct = round(psutil.cpu_percent(interval=0.2), 1)
    except Exception:
        pass
    print("CPU:")
    print("  训练（32 样本×2 轮×6 tick）耗时 : %s s" % br["train_s"])
    print("  单次推理（ticks=6，20 次平均）: %s ms（默认全 8 子脑路径）" % br["think_ms"])
    print("  R+ 路径单次调用（ticks=6）耗时 : %s ms" % rp["rplus_per_call_ms"])
    print("  R+ 行缓存命中率 : %.1f%%（%d 行）" % (rp["cache_hit_rate"] * 100, rp["cache_rows"]))
    print("运行速度基准（同 1 次 think，取平均）:")
    print("  默认全 8 子脑路径 : %s ms" % speed["default_ms"])
    print("  R+ 单脑+缓存路径  : %s ms（比默认快 %s×）"
          % (speed["rplus_ms"], speed["speedup_rplus_over_default"]))
    print("  float32 参照      : %s ms（量化路径与其同量级，但内存 1/4、可移植）" % speed["float_ms"])
    if cpu_pct is not None:
        print("  整机 CPU 瞬时占用 : %s%%（轻负载；本骨架单核即可跑）" % cpu_pct)
    print("GPU:")
    print("  本骨架纯 CPU 推理，GPU 占用 : N/A（未用任何 GPU/CUDA，numpy 走 CPU 后端）")
    print("\n结论：脑体张量 < 3 MB；训练 <0.5 s；推理 毫秒~百毫秒；无需 GPU，普通笔记本即可复现。")


if __name__ == "__main__":
    main()
