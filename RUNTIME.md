# lianghua · 运行环境说明

本包是**纯 Python + numpy** 的量化类脑骨架，全 INT8/定点（无 float 进入放电路径），
跨平台可跑：Windows / macOS / Linux 通用。下面是完整环境与复现步骤。

---

## 1. 依赖

只有一个硬依赖：

| 依赖 | 版本 | 说明 |
|---|---|---|
| `numpy` | `>= 1.20`（实测 2.5.4） | 核心数组运算，INT8/INT32 |
| `Pillow`（PIL） | 仅视频生成需要 | 生成演示视频帧用，跑脑体不需要 |

`requirements.txt` 内容（仅 numpy）：

```text
numpy>=1.20
```

- **跑脑体（build/train/think/eval/arch/sleep/prune/homeostasis/rplus/verify）只需 numpy。**
- **要生成演示视频**（`video/make_video.py`）还需 `pip install pillow` 和系统里装好 `ffmpeg`。

---

## 2. Python 版本

| 要求 | 说明 |
|---|---|
| Python | **3.9+**（实测 3.15） |
| 原因 | 代码用了 `np.savez_compressed`、类型无关的整数数组；3.9+ 都兼容 |

> 不需要 CUDA、不需要 GPU、不需要任何深度学习框架（纯 numpy）。

---

## 3. 安装

```bash
# 建议用虚拟环境
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt      # 装 numpy
pip install pillow                    # 可选：仅视频需要
```

---

## 4. 跑通验证（3 步）

```bash
# ① 自检（不读写文件，确认 numpy + 量化逻辑 OK）
python lianghua.py
# 期望输出含：OK（纯 INT8/定点，无 float，可移植 MCU）

# ② 编译 + 训练 + 推理
python cli.py build --out brain.npz
python cli.py train --brain brain.npz --data samples.tsv --epochs 2
python cli.py think --brain brain.npz --text "今天天气怎么样，会下雨吗"

# ③ 量化 vs float 一致性 + R+ 一致性
python cli.py verify
python cli.py rplus --brain brain.npz
```

跑完应看到（与 `example_run.txt` 一致）：
- `verify` → `平均 Jaccard 0.99 ... 结论: PASS`
- `rplus` → `单 tick fired 集合一致：36/36 ... 结论：PASS 与原路径完全一致`

---

## 5. 各平台注意

### Windows
- 控制台中文乱码时先设：
  ```powershell
  $env:PYTHONUTF8='1'
  ```
- 视频合成需要系统装 `ffmpeg`（加入 PATH，或 `video/make_video.py` 里设 `FFMPEG` 环境变量指向 `ffmpeg.exe`）。
- 中文字体默认找 `C:\Windows\Fonts\msyh*.ttc`（微软雅黑）；没有就回退 `simhei.ttf` / 默认字体。

### macOS / Linux
- `pip install numpy pillow` 即可，无需额外环境。
- 视频需要 `ffmpeg`（`brew install ffmpeg` / `apt install ffmpeg`）。

### 嵌入式 / MCU（移植目标）
- 核心全是整数数组运算，numpy 只是参考实现；可把 `fire`/`propagate`/`aggregate`
  翻译成 C / Rust，把 `neighbors`/`weights` 当只读表、`state`/`last_spike` 当读写数组。
- 默认规模 8×2048×32 = 2.578 MB，适合带 4MB+ 内存的边缘芯片。

---

## 6. 环境变量开关（可选）

| 变量 | 含义 | 默认 |
|---|---|---|
| `PYTHONUTF8=1` | 强制 Python 用 UTF-8 控制台 | 关 |
| `RPLUS_AGG` | R+ 聚合模式 `key`/`unique`/`argsort` | `key` |
| `FFMPEG` | ffmpeg 可执行文件路径 | 自动找 PATH |

---

## 7. 硬件信息 + 资源占用 + 速度基准（实测）

跑 `python bench_env.py` 可得下表全部数字（脚本会自动读 CPU 型号 / 核数、
物理内存、GPU、磁盘，并实测内存/CPU/GPU 占用与推理耗时）。

实测机（2026-11-21）：

| 项 | 值 |
|---|---|
| 系统 | Windows 11 / AMD64，Python 3.15.2 |
| CPU | Intel64 Family 6 Model 191（16 线程，普通笔记本/桌面） |
| 物理内存 | 总 16151 MB / 可用 约 11 GB |
| GPU | 未使用（本骨架纯 CPU，GPU 占用 0） |
| 磁盘 | I 盘已用约 121 GB / 共 271 GB |

### 资源占用（真实读值）

| 指标 | 数值 | 说明 |
|---|---|---|
| **脑体张量** | **2.578 MB** | state+weights+neighbors+last_spike，确定值（INT8/INT32） |
| 进程 RSS（Python+numpy+脑体） | 起 37 MB → 跑完 42 MB | 其中脑体仅 2.578 MB，其余是 Python 解释器 + numpy 加载 |
| 系统内存占用率 | 30~33% | 含系统与其他进程，非本脑体贡献 |
| **GPU 占用** | **0（N/A）** | 未引入任何 CUDA / GPU 后端 |

### 运行速度（同 1 次 think，取 20 次平均）

| 路径 | 耗时 | 说明 |
|---|---|---|
| 训练（32 样本 × 2 轮 × 6 tick） | 0.06 s | INT8 STDP |
| 默认全 8 子脑推理（ticks=6） | **5.9 ms** | 纯 numpy，已预热 |
| **R+ 单脑 + 行缓存 + key 聚合** | **0.8~1.2 ms** | 比默认快 **5~6×** |
| float32 参照（同规模） | 164 ms | 量化版与其同量级但内存 1/4、可移植 |
| 整机 CPU 瞬时占用 | 15~30% | 轻负载，单核即可跑 |

> 口径说明：单独冷启动的 `think` 首次约 70~135 ms（含首调预热），循环平均后为 5.9 ms；
> "极快"主要靠 **R+ 路径**（0.8~1.2 ms），默认路径是 5 ms 量级（够用，但非"极快"）。

---

## 8. 核心理念 vs 实测（不吹过头）

| 理念 | 实测证据 | 是否达成 |
|---|---|---|
| **极低内存** | 脑体张量 **2.578 MB**（INT8/定点，是 float32 版的 1/4） | ✅ 脑体极低；但要说明进程 RSS 42 MB 里大头是 Python 解释器+numpy，不是脑 |
| **极低 CPU 占用** | 训练 0.06s、推理 5.9ms、R+ 0.8ms，单核可跑 | ✅ 单次占用低（是"轻活"，非重负载） |
| **极低 GPU 占用** | 未用 GPU/CUDA，GPU 占用 **0** | ✅ 完全纯 CPU |
| **极快运行** | R+ 路径 0.8~1.2ms；默认 5.9ms；float 参照 164ms | ✅ R+ 路径体现"快"；默认路径是"够用"非"极快"，要分清 |

**诚实边界**：
- "极低内存"指**脑体张量** 2.578 MB；用 Python 解释器跑时进程 RSS 是 ~40 MB（解释器+ninpy 本体）。
  真要压到 3 MB 级别，得把核心翻译成 C/Rust 静态部署（numpy 只是参考实现）。
- "极快"是**相对**的：R+ 单脑 0.8ms 对"轻推理"极快；若换成 50M 脑体（CHN2），耗时和内存都会
  按规模放大，那时才需要 mmap + GPU。本骨架是**最小可复现子集**，不代表全规模上限。

---

## 9. 实测环境（软件栈）

| 项 | 值 |
|---|---|
| 操作系统 | Windows 11 |
| Python | 3.15.2 |
| numpy | 2.5.4 |
| Pillow | 13.0.0（仅视频） |
| ffmpeg | 9.0.2（仅视频） |
| psutil | 可选，`bench_env.py` 读硬件/占用用（没有则回退系统值） |
