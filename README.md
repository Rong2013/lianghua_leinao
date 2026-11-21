# lianghua（量化类脑）

一个**全量化**的通用类脑骨架：皮层分级 + 丘脑门控 + 默认模式网络（DMN），
全部张量 INT8 / 定点，**无 float**，可移植到 MCU / 边缘设备。

> **开源版定位声明（2026-11-22）**：
> 本仓库（lianghua）是**算法参考实现**——8 子脑 × 2048 神经元（16,384 神经元 / 2.6 MB），
> 用于教学、可移植、嵌入式验证。
> **生产版**基于**分片存储（100 子脑 × 2M = 2 亿神经元 / 33 GB memmap）+ R+ 运行时
> （编码键聚合 + 行缓存 + 冷启动顺序预读）+ memmap 调度**，不在本仓库分发。
> 深度用户（需要 50M 双脑 / 2 亿子脑 / 大规模训练）请**申请完整版访问**
> （联系仓库 owner；完整版含分片表、R+ 双份、磁盘物理距离最小化、睡眠优化）。
>
> **AI 辅助开发声明**：本项目在 AI 助手参与下开发（架构与参数由人确定，
> 代码与文档由 AI 辅助生成）。所有实测数字均来自仓库内脚本的真实运行结果。

- 规模默认：**8 子脑 × 2048 神经元 × 32 邻居**（16,384 神经元 / 524,288 突触）
- 内存：**2.578 MB**（state 16 KB + weights 512 KB + neighbors 2 MB）
- 纯 Python + numpy，Windows / macOS / Linux 通用

---

## 快速开始

```bash
# 1. 依赖
pip install numpy          # 只需 numpy

# 2. 编译神经元（生成拓扑 + 初始权重）
python cli.py build --out brain.npz

# 3. 训练（INT8 STDP，默认无奖赏调制；--plot 自动出栅格图）
python cli.py train --brain brain.npz --data samples.tsv --epochs 2 --plot viz/train_raster.png

# 4. 推理
python cli.py think --brain brain.npz --text "今天天气怎么样，会下雨吗"

# 5. 评估准确率
python cli.py eval --brain brain.npz --data samples_test.tsv

# 6. 类脑结构演示（分级 / 门控 / 侧抑制 / DMN / 睡眠前后准确率）
python cli.py arch --brain brain.npz --data samples_test.tsv

# 7. 自检 / 量化一致性 / R+ 一致性
python lianghua.py
python cli.py verify
python cli.py rplus --brain brain.npz
```

Windows 控制台如果出现中文乱码，先执行：

```powershell
$env:PYTHONUTF8='1'
```

---

## 目录结构

| 文件 | 作用 |
|---|---|
| `lianghua.py` | 核心：`QuantizedBrain`（放电 / 门控 / STDP / 传播 / 类脑结构 / 保存加载） |
| `encoder.py` | 文本 → 脉冲种子（1~3-gram 混合哈希编码） |
| `cli.py` | 命令端：`build / train / think / eval / stats / arch / sleep / prune / homeostasis / rplus / selftest / verify` |
| `rplus_simple.py` | R+ 简化版：向量化核心聚合 + 行缓存 + 分层一致性自检 |
| `bench_env.py` | 硬件信息 + 内存/CPU/GPU 占用 + 速度基准实测 |
| `RUNTIME.md` | 运行环境说明（依赖/安装/各平台/硬件与占用实测） |
| `verify_quant.py` | 量化版 vs float 版放电一致性验证 |
| `visualizer.py` | Spike Raster Plot + 放电活动图（PIL 出 PNG，零重依赖） |
| `metrics.py` | Prometheus 指标导出器（tick 耗时/饱和率/剪枝数/缓存命中率，零重依赖） |
| `samples.tsv` | 示例训练集（8 类 × 4 条） |
| `samples_test.tsv` | 示例留出集（8 类 × 2 条，未参与训练） |
| `example_run.txt` | 完整流程的真实运行输出（build → train → eval → arch → prune → homeostasis → rplus → verify） |
| `LICENSE` | MIT |

运行产物：`brain.npz`（神经元状态 + 权重 + 邻居表 + 训练原型）。

---

## 类脑设计理念

lianghua 不是"把几个参数改成 INT8"，而是把**人脑的真实结构**做成最小可移植骨架：

```text
  文本 / 字节
     │  稀疏编码（encoder.py：1~3-gram → 少量神经元）
     ▼
  丘脑中继 gate_topk / thalamic_relay（Moruzzi & Magoun 1949：注意的瓶颈）
     │
     ▼
  感觉层 L1 [0..2]  →  联合层 L2 [3..4]  →  决策层 L3 [5..7]
     │  Hubel & Wiesel / Felleman & Van Essen：层级皮层
     │
     ├── 侧抑制 lateral_inhibition（E/I 平衡，Mountcastle 1957）
     ├── STDP + 神经调质 train_samples(reward=...)（Schultz 1998）
     └── 放电模式 → 与记忆原型做 Jaccard 匹配
     ▲
  离线巩固：
     ├── 睡眠重放 sleep_replay（Diekelmann & Born 2010）
     ├── 突触稳态 homeostasis（Turrigiano & Nelson 2004）
     └── 突触修剪 prune（Huttenlocher 1979）
  DMN dmn_idle（Raichle 2001）：无外部输入时决策层自发活动
```

### 生物结构 → 代码 → 参数 → 命令映射

| 生物机制 | 代码 | 参数（默认 / 可开关） | 命令 |
|---|---|---|---|
| 皮层分级（层级连接） | `hierarchy()` / `layer_of()` | 感觉 3 / 联合 2 / 决策 3（按 `NSUB` 比例划分） | `arch` / `stats` |
| 丘脑门控（注意瓶颈） | `gate_topk()` / `thalamic_relay()` | INT8 点积 / 255，top-k=4 | `arch` / `think` |
| 侧抑制（E/I 平衡） | `lateral_inhibition()` + `think(inhibit=...)` | keep=2，floor=20，默认**不生效** | `think --inhibit` / `eval --inhibit` |
| 赫布可塑性 + 三因子 | `stdp_update()` + `train_samples(reward=...)` | A_PLUS=6，A_MINUS=2，reward=100 不调制 | `train --reward ...` |
| 睡眠重放（离线巩固） | `sleep_replay()` | 每原型 strength=70，ticks=4，结束后膜电位复位 | `sleep` |
| 突触稳态 | `homeostasis()` | 目标非零 \|w\|=2，只降不升 | `homeostasis` |
| 突触修剪 | `prune()` | \|w\| ≤ threshold 置 0 | `prune` |
| DMN 自发活动 | `dmn_idle()` | 决策层每 tick 32 个内在节律种子，强度 60 | `arch` |
| 核心聚合（R+ 简化版） | `rplus_simple.aggregate()` / `quant_fire()` / `propagate()` / `RowCache` | 默认关闭；`RPLUS_AGG` 可选 key / unique / argsort | `rplus` |

### 关键设计约束

- **规则 C**：原有 `fire()` / `gate_topk()` 一行未改；所有新机制默认不进入已验证路径
  （`train` / `think` 默认值不变），只通过 `arch` / `sleep` / `prune` / `homeostasis` /
  `rplus` / `--inhibit` / `--reward` 显式调用。
- **规则 A**：每个新增机制代码上方三行注释：`# 生物依据：` / `# 简化说明：` / `# 参数来源：`。
- **纯 INT8 / 定点**：侧抑制用 INT16 中间量、重放用 INT8 输入、稳态/修剪用整数乘除，
  **没有任何 float 进入放电路径**（诊断输出里的浮点数仅用于展示，不参与计算）。

---

## 命令参考

### `build` —— 编译神经元

```bash
python cli.py build --out brain.npz [--nsub 8] [--n 2048] [--k 32] [--seed 20261120]
```

| 参数 | 含义 | 默认 |
|---|---|---|
| `--nsub` | 子脑数（皮层分区） | 8 |
| `--n` | 每个子脑的神经元数 | 2048 |
| `--k` | 每个神经元的出边数（邻居） | 32 |
| `--seed` | 随机种子（拓扑可复现） | 20261120 |

拓扑规则：**70% 环形近邻（±1..8）+ 30% 随机长程**，近似皮层的局部连接与长程投射。

内存估算（1 个神经元 = 1 B state + K B 权重 + 4K B 邻居）：

| 规模 | 神经元 | 突触 | 内存 |
|---|---|---|---|
| 8 × 2048 × 32 | 16,384 | 524,288 | ≈ 2.6 MB |
| 16 × 4096 × 32 | 65,536 | 2,097,152 | ≈ 10.1 MB |
| 32 × 8192 × 48 | 262,144 | 12,582,912 | ≈ 60.3 MB |
| 100 × 8192 × 64 | 819,200 | 52,428,800 | ≈ 250 MB |

`build` 只分配内存、不训练，可以反复用不同 seed 生成不同个体。

### `train` —— 训练

```bash
python cli.py train --brain brain.npz --data samples.tsv \
    [--epochs 2] [--ticks 6] [--reward 100] [--out brain_trained.npz] [--plot viz/raster.png]
```

- 每个标签固定落到一个子脑（`label` 哈希 → 子脑号），不同类别分区域学习。
- 每个样本传播 `ticks` 步：外部输入 → 放电 → 沿权重传给下一 tick；
  放电为空时重新注入输入（保证能量不灭）。
- 每个 tick 做一次 **INT8 STDP**：共激活的邻居 `+6`，其余 `-2`，截断在 `[-127, 127]`。
- 每个样本的放电集合作为一条**记忆原型**（同一标签可有多条原型）。
- **三因子可塑性**：`--reward` 是奖赏百分比（100 = 不调制；>100 加强可塑性；<100 减弱），
  只缩放 `a_plus`，不改变默认行为。
- **`--plot PATH`**：可选，训练完自动收集 spike 并用 `visualizer.py`（PIL）出栅格图 PNG。

### `think` —— 推理

```bash
python cli.py think --brain brain.npz --text "hello" [--ticks 6] [--inhibit] [--json]
```

输出：门控子脑（含层级）、总放电数、识别标签 + 置信度（Jaccard）、top 神经元编号。
`--inhibit` 启用侧抑制（默认关，不改变已验证指标）。

### `eval` —— 评估

```bash
python cli.py eval --brain brain.npz --data samples_test.tsv [--inhibit] [--json]
```

逐条推理并与真值比较，输出准确率、平均 Jaccard、错例。

### `arch` —— 类脑结构演示

```bash
python cli.py arch --brain brain.npz [--text "今天天气怎么样"] [--data samples_test.tsv] \
                   [--out _arch.npz] [--epochs 1] [--ticks 6]
```

打印：
- 皮层分级（感觉 / 联合 / 决策 各含哪些子脑）
- 丘脑中继：放行子脑 + 相关度 + 层级
- 侧抑制：各子脑放电数 → 增益百分比
- DMN 静息：决策层每 tick 自发活动、累计覆盖、自发率
- 若给了 `--data`：顺带跑一次睡眠重放，打印睡眠前后准确率

### `sleep` —— 睡眠重放

```bash
python cli.py sleep --brain brain.npz [--data samples_test.tsv] [--epochs 1] \
                    [--ticks 4] [--strength 70] [--out _sleep.npz]
```

用已保存的放电原型当内部输入重放（海马-皮层对话的简化版），再用 STDP 巩固；
结束后清零膜电位（睡眠复位）。若给了 `--data`，打印睡眠前后准确率。

### `prune` —— 突触修剪

```bash
python cli.py prune --brain brain.npz [--threshold 1] [--data samples_test.tsv] [--out _prune.npz]
```

把 \|w\| ≤ threshold 的突触置 0（保留数组形状，可移植），打印剪掉数量、稀疏度、前后准确率。

### `homeostasis` —— 突触稳态

```bash
python cli.py homeostasis --brain brain.npz [--target 2] [--out _homeo.npz]
```

把每个子脑非零突触的平均 \|w\| 下调回目标（只降不升），打印前后均值。

### `rplus` —— R+ 简化版一致性

```bash
python cli.py rplus --brain brain.npz [--text "你好"] [--json]
```

对照三种 R+ 聚合模式（`key` / `unique` / `argsort`）与默认路径，
验证单 tick / 多 tick 的 fired 集合完全一致，并打印行缓存命中率。

### `stats` / `selftest` / `verify`

```bash
python cli.py stats  --brain brain.npz      # 规模/内存/权重分布/稀疏度/原型
python cli.py selftest                      # 内置自检（不读写文件）
python cli.py verify                        # 量化 vs float 一致性
```

---

## 训练数据格式

每行一个样本，三种写法都支持（自动识别）：

```text
# TSV：标签 <TAB> 文本
天气	今天天气怎么样 会下雨吗
编程	python 代码 函数

# 竖线分隔
天气|今天天气怎么样

# 纯文本：标签自动取 "样本N"
今天天气怎么样
```

JSONL 也支持：

```json
{"label": "天气", "text": "今天天气怎么样"}
```

---

## 量化方案（固定）

| 量 | 类型 | 约定 |
|---|---|---|
| 膜电位 `state` | INT8 | `[0,100]` ↔ `[0.0,1.0]`，leak = 90/100 |
| 突触权重 `weights` | INT8 | `[-127,127]`（STDP 截断） |
| 放电阈值 | 定点 | `50`（= 0.5） |
| 邻居索引 `neighbors` | INT32 | 神经元编号，不量化 |
| 门控相关性 | INT8 点积 / 255 | 无 float |

膜电位更新（整数运算）：

```text
new = (old * 90) // 100 + 输入和        # 旧=40、输入=40 → 76 ≥ 50 → 放电
放电后 state = 0
```

- 中间量用 INT32，落盘用 INT8，`np.clip` 兜底防溢出。
- `verify_quant.py` 实测：同输入下量化版 vs float 版
  **平均 Jaccard 0.99、放电总数比 1.01**（量化几乎无损）。

---

## 实测结果（可复现）

环境：Windows / Python 3.15 / numpy 2.5。
数据集：`samples.tsv`（8 类 × 4 条，32 条训练）+ `samples_test.tsv`（8 类 × 2 条，16 条留出）。

### 默认路径（无新机制）

| 指标 | 结果 |
|---|---|
| 训练集准确率 | **100.0%**（32/32，平均 Jaccard 0.952） |
| 留出集准确率 | **81.2%**（13/16，平均 Jaccard 0.200） |
| 量化一致性 | **Jaccard 0.99 / 放电比 1.01** |
| 内存 | **2.578 MB** |
| 权重范围（训练后） | `[-127, 104]` |
| 原型 | 8 标签 / 64 放电模式 |

### 类脑机制实测（真实输出，未手工调整）

| 机制 | 命令 | 实测结果 |
|---|---|---|
| 侧抑制 | `eval --inhibit` | 留出 81.2% → 81.2%（平均 Jaccard 0.200 → 0.192） |
| 睡眠重放（1 轮） | `sleep --epochs 1` | \|w\| 839734 → 1063233（+26.6%），留出 81.2% → 75.0% |
| 睡眠重放（2 轮，strength=90） | `sleep --epochs 2 --strength 90` | \|w\| 839734 → 1289763（+53.6%），留出 81.2% → 75.0% |
| 突触修剪（\|w\|≤1） | `prune --threshold 1` | 剪掉 301609 / 524288（57.5%），稀疏度 19.2% → 57.5%，留出 81.2% → 81.2% |
| 突触稳态（目标 1） | `homeostasis --target 1` | 各子脑非零 \|w\| 均值 `[2,1,1,1,1,3,1,1] → [1,1,1,1,1,1,1,1]` |
| DMN 静息 | `arch` | 决策层每 tick 32 个神经元放电，6 tick 覆盖 192 / 2048（9.4%），自发率 1.5625% / 神经元·tick |
| R+ 简化版一致性 | `rplus` | 单 tick 36/36、多 tick 36/36 全一致；行缓存命中 2054 / 未命中 373（84.6%） |
| 栅格图（Spike Raster） | `train --plot viz/train_raster.png` | `viz/train_raster.png`（1310×654 PNG，PIL 出图，零重依赖；展示类分离） |

> 说明：
> - 训练集 100% 是"记忆"（原型与输入完全一致），留出集 81.2% 才是泛化。
> - 睡眠重放在本骨架上**没有提升留出集**（甚至 1 轮就退化到 75%），这是真实结果，
>   没有调参把它藏起来；在更大规模 / 更长训练 / 更大原型库上睡眠重放才可能体现巩固效应。
> - 修剪 57.5% 弱突触后留出集不变，说明骨架的记忆确实集中在少数强突触上。
> - 这些数字全部来自 `cli.py` 的真实输出，没有手工调整。

---

## 可视化：Spike Raster Plot（神经元放电图）

`visualizer.py` 把训练/推理时的脉冲记录画成**栅格图**（SNN 最直观的诊断/展示工具）：
- **每行一个神经元，每列一个时间步，黑点 = 该神经元在该 tick 放电**。
- 训练后不同标签的放电模式在"神经元空间"里会分开（类分离），一张栅格图就能看出
  "脑"是否真的学到了类别差异——这是放给 README 里最"酷"的一张图。

```bash
# 训练时自动收集 spike 并出图（推荐）
python cli.py train --brain brain.npz --data samples.tsv --plot viz/train_raster.png

# 或单独出图（已有 brain.npz）
python visualizer.py --brain brain.npz --text 猫 狗 汽车 --out viz/raster.png
python visualizer.py --brain brain.npz --text 猫 狗 汽车 --activity   # 也出子脑活动图
```

出图后端：**PIL（Pillow）**，零重依赖（不需要 matplotlib）；matplotlib 可用时可用
`visualizer.raster_plot(..., use_mpl=True)` 切换。PIL 不可用时打印提示，不崩、不影响训练。

实测：`train --plot` 生成 `viz/train_raster.png`（1310×654，约 12 KB，32 样本 × 6 ticks）。

---

## 指标可观测性：Prometheus / Grafana 导出

`metrics.py` 把训练/推理的关键指标暴露成 **Prometheus 时序指标**（纯 Python 文本 exposition format，
零重依赖，不用 `prometheus_client`），从"日志排查"升级为"实时监控"，支撑大规模部署。

```bash
# 起指标服务（:8080/metrics，daemon 线程）
python metrics.py --port 8080
# 或演示（采集 + 起服务 5s）
python metrics.py --demo
# 或打印当前指标 JSON 快照
python metrics.py --json
```

**指标清单**（全部 counter/gauge，Prometheus 抓取 `http://host:8080/metrics`）：

| 指标 | 类型 | 含义 |
|---|---|---|
| `lianghua_tick_seconds` | gauge | 最近一次 tick 耗时（秒） |
| `lianghua_tick_seconds_total` | counter | 累计 tick 耗时（秒） |
| `lianghua_saturation_rate` | gauge | 膜电位饱和率（[0,1]，越高越接近 INT8 顶格） |
| `lianghua_prune_count_total` | counter | 累计剪枝突触数（睡眠修剪） |
| `lianghua_cache_hit_ratio` | gauge | R+ 行缓存命中率（[0,1]） |
| `lianghua_train_epochs_total` | counter | 累计训练轮数 |
| `lianghua_think_calls_total` | counter | 累计推理次数 |
| `lianghua_accuracy` | gauge | 最近一次评估准确率（[0,1]） |

**采集已接进 cli.py**：`train` 自动记 tick 耗时 + 饱和率 + 训练轮；`think` 记推理次数 + 准确率；
`eval` 记准确率。Grafana 配 Prometheus 抓取该端点即可画时序面板。

程序内采集（不依赖 cli）：
```python
import metrics
metrics.tick_seconds(0.42)
metrics.saturation_rate(0.003)
metrics.prune_count(225440359)
metrics.cache_hit_ratio(0.846)
print(metrics.render())     # Prometheus 文本
```

---

## R+ 独立开源

R+（`I:\R+\rplus-open\`）已剥离为**自包含独立仓库**，可单独发布（MIT License）：

- **`rplus/`**：核心运行时（`rplus_new.py` 原生语法解释器 + `rplus_compiler.py` Python 子集编译器
  + `rplus_lib.py` 标准库 + `rplus_io.py` 跨平台 IO + `rplus_core.py` 核心聚合）
- **`examples/`**：`.rplus` 示例（原生语法 + Python 风格）
- **`tests/`**：`test_rplus.py`（解析 + numpy 跑 + 脱 numpy 纯 Python 兜底，3 项全 PASS）
- **`setup.py`**：`pip install -e .`（零依赖，`[fast]` extra 可选 numpy）
- **`README.md`**：语法速查 + 与 Python/C++ 对照表 + 设计纪律（独立/快速/简单/稳定/万能）

跑测试：`python I:\R+\rplus-open\tests\test_rplus.py`（全 PASS，exit 0）。

R+ 吸引编译器/DL 底层开发者，形成技术生态外溢；类脑源码全 R+ 重编译时走 `I:\R+\rplus-open\rplus\`。

---

## 核心理念：极低内存 / 极低 CPU / 极低 GPU / 极快

这是 lianghua 的设计目标。跑 `python bench_env.py` 可复现下表全部数字：

| 理念 | 实测证据 | 达成情况 |
|---|---|---|
| **极低内存** | 脑体张量 **2.578 MB**（INT8/定点，是 float32 版 1/4） | ✅ 脑体极低；进程 RSS ~42 MB 大头是 Python+numpy，非脑 |
| **极低 CPU** | 训练 0.06 s、默认推理 5.9 ms、**R+ 0.8~1.2 ms**，单核可跑 | ✅ 轻负载，占用低 |
| **极低 GPU** | 未用 GPU/CUDA，GPU 占用 **0** | ✅ 纯 CPU |
| **极快** | R+ 路径 0.8~1.2 ms；默认 5.9 ms；float32 参照 164 ms | ✅ R+ 路径"快"；默认路径是"够用"，要分清 |

**诚实边界**：
- "极低内存"指脑体张量 2.578 MB；用 Python 解释器跑时进程 RSS 约 40 MB（解释器+numpy 本体）。
  要压到 3 MB 级得把核心翻译成 C/Rust 静态部署（numpy 只是参考实现）。
- "极快"是相对的：R+ 单脑 0.8 ms 对轻推理极快；换成 50M 脑体（CHN2）会按规模放大，
  那时才需要 mmap + GPU。本骨架是最小可复现子集。

硬件信息（CPU 型号/核数、物理内存、GPU、磁盘）与完整占用读值见 `RUNTIME.md` 第 7 节。

---

## 运行环境（摘要）

依赖只有 `numpy`（跑脑体）；视频生成另需 `pillow` + 系统 `ffmpeg`。完整说明见 **`RUNTIME.md`**：
- 依赖 / Python 版本（3.9+，实测 3.15）/ 安装 / 三步验证
- 各平台注意（Windows 乱码、字体、ffmpeg）
- 硬件信息 + 内存/CPU/GPU 占用 + 速度基准（`bench_env.py` 实测）
- 核心理念 vs 实测（不吹过头的边界说明）

---

## R+ 简化版

R+ 是 `I:/R+` 的运行时核心聚合（向量化 + 行缓存）。本仓库里 `rplus_simple.py`
把它**简化**成量化骨架可用的三件东西：

1. **`aggregate(ids, wts, mode)`** —— 重复目标权重和，三种等价实现：
   - `key`（默认）：把 `(目标, 权重)` 压进一个 int64 键，一次排序完成分组 + 求和；
   - `unique`：`np.unique` + `np.add.at`（与 `lianghua.fire` 原路径同法）；
   - `argsort`：`np.argsort` + `np.add.reduceat`。
   任何越界 / 异常自动回退 `unique`（正确性是硬底线，加速是可选件）。
2. **`quant_fire()` / `propagate()`** —— 与 `QuantizedBrain.fire` / `propagate`
   **fired 集合逐位一致**的 R+ 路径，唯一区别是"重复目标求和"用编码键排序。
3. **`RowCache`** —— 环形槽位行缓存（简化版），把反复访问的 `(子脑, 神经元)` 邻居/权重行留在内存。

`cli.py rplus` 打印真实一致性结果（默认 4 条输入 × 3 子脑 × 3 模式）：

```text
R+ 简化版一致性自检
  单 tick fired 集合一致：36/36
  多 tick fired 集合一致：36/36
    key     单 tick 12/12  多 tick 12/12
    unique  单 tick 12/12  多 tick 12/12
    argsort 单 tick 12/12  多 tick 12/12
  行缓存：命中 2054 / 未命中 373（命中率 84.6%，373/65536 行）
  结论：PASS 与原路径完全一致
```

环境变量开关：

| 变量 | 含义 | 默认 |
|---|---|---|
| `RPLUS_AGG` | 聚合模式 `key` / `unique` / `argsort` | `key` |

R+ 是**可选路径**，默认训练 / 推理仍走 `lianghua.py` 原路径；`rplus` 命令只用于
验证一致性，不改变 `brain.npz` 里的权重或原型。

---

## Python API

```python
import lianghua, encoder
import rplus_simple

b = lianghua.QuantizedBrain.build(nsub=8, n=2048, k=32)
seeds, wts = encoder.text_to_seeds("你好", b.N)

r = b.propagate(sub=0, seeds=seeds, wts=wts, ticks=6)
print(r["counts"], r["sync"], r["fired"][:10])

b.train_samples([("问候", "你好"), ("天气", "下雨")], epochs=2, reward=100)
print(b.think("今天下雨吗"))
print(b.think("今天下雨吗", inhibit=True))   # 启用侧抑制

print(b.hierarchy())                # 皮层分级
print(b.thalamic_relay(seeds, top_k=4))   # 丘脑中继（含相关度）
print(b.dmn_idle())                 # DMN 静息
print(b.sleep_replay(epochs=1))    # 睡眠重放
print(b.homeostasis(target=2))     # 稳态
print(b.prune(threshold=1))         # 修剪

cache = rplus_simple.RowCache(k=b.K)
r = rplus_simple.propagate(b, 0, seeds, wts, ticks=6, mode="key", cache=cache)
print(cache.stats())

b.save("brain.npz")
b2 = lianghua.QuantizedBrain.load("brain.npz")
print(b2.stats())
```

`QuantizedBrain` 方法：

| 方法 | 作用 |
|---|---|
| `build(nsub, n, k, seed)` | 编译神经元（类方法） |
| `fire(sub, seeds, wts, tick)` | 单 tick 放电（原有逻辑，未改） |
| `gate_topk(q, top_k)` | 丘脑门控（原有逻辑，未改） |
| `stdp_update(sub, fired, prev)` | INT8 STDP |
| `propagate(sub, seeds, wts, ticks)` | 多 tick 传播 |
| `train_samples(samples, epochs, ticks, reward)` | 训练（`reward=100` 不调制） |
| `think(text, ticks, inhibit)` | 推理（`inhibit` 可选启用侧抑制） |
| `hierarchy()` / `layer_of(sub)` | 皮层分级（组织视图） |
| `thalamic_relay(q, top_k)` | 丘脑中继（含相关度 + 层级） |
| `lateral_inhibition(activations)` | 侧抑制（输出整数增益百分比） |
| `dmn_idle(ticks, n_seed, w)` | DMN 自发活动（只读） |
| `sleep_replay(epochs, ticks, strength)` | 睡眠重放（离线再巩固） |
| `homeostasis(target)` | 突触稳态（只降不升） |
| `prune(threshold)` | 突触修剪 |
| `save(path)` / `load(path)` | 落盘 / 读取（`.npz`） |
| `stats()` / `quant_stats()` | 规模 / 量化体检 / 稀疏度 |

`rplus_simple` 模块：

| 函数 / 类 | 作用 |
|---|---|
| `aggregate(ids, wts, mode)` | 重复目标权重和（key / unique / argsort） |
| `quant_fire(brain, sub, seeds, wts, tick, mode)` | 与 `fire` 一致的 R+ 放电 |
| `propagate(brain, sub, seeds, wts, ticks, mode, cache)` | 与 `propagate` 一致的 R+ 传播 |
| `RowCache` | 环形槽位行缓存 |
| `selftest(brain)` | R+ vs 原路径一致性自检 |

---

## 与完整 50M 版的区别

| | lianghua（本仓库） | CHN2 完整版 |
|---|---|---|
| 规模 | 8 × 2048（骨架） | 100 子脑 × 50 万神经元 |
| 数值 | 全 INT8 / 定点 | float32 + INT8 混合 |
| 用途 | 可移植、可教学、可嵌入 | 生产 / 大规模训练 |
| 内存 | 2.6 MB | 数十 GB（memmap 落盘） |
| 核心聚合 | `rplus_simple`（简化版，纯整数） | `I:/R+`（float32 + 编码键 + 行缓存） |

lianghua 证明"量化也能跑通类脑全流程"；CHN2 是"全精度生产版"。
两者共享同一套生物结构（皮层分级 / 丘脑门控 / DMN / STDP / 睡眠巩固），本仓库是它的
**最小可复现子集**。

---

## 参数怎么调

| 想要 | 调整 |
|---|---|
| 更强记忆、更少泛化 | `--epochs`↑ |
| 更好泛化 | 增大 `--n`、`--k`，或改 `encoder.py` 的 n-gram 混合 |
| 更省内存 | 降 `--k`（权重/邻居线性下降） |
| 放电更稀疏 | 提高 `THR`（默认 50） |
| 训练更稳 | 降低 `A_PLUS` 或提高 `A_MINUS`（默认 6 / 2） |
| 启用侧抑制 | `think --inhibit` / `eval --inhibit` |
| 睡眠巩固 | `sleep --epochs 1 --strength 70`（本骨架上不会提升留出集） |
| 修剪弱突触 | `prune --threshold 1`（57.5% 突触可剪而不损失留出集） |
| 稳态控制 | `homeostasis --target 1`（把非零 \|w\| 拉到 1） |

---

## 常见问题

**Q：`think` 总是返回"无匹配"？**
A：还没训练，或训练集为空。先跑 `train`。

**Q：训练集 100% 但留出集低？**
A：正常。骨架的字符哈希编码泛化有限；增大规模或改编码器。

**Q：能不能不用 numpy？**
A：核心全是整数数组运算，可移植到 C / Rust / 嵌入式；numpy 只是参考实现。

**Q：`.npz` 里有什么？**
A：`state`（INT8）、`weights`（INT8）、`neighbors`（INT32）、`last_spike`（INT32）、
`meta`（JSON：规模 / seed / 原型 / 标签→子脑映射）。

**Q：R+ 会不会改默认行为？**
A：不会。`rplus` 命令只做一致性自检；默认训练 / 推理仍走 `lianghua.py` 原路径。

---

## License

MIT（见 `LICENSE`）。

## 贡献

欢迎提交编码器改进、规模基准、嵌入式移植。提交前请跑：

```bash
python lianghua.py && python cli.py verify && python cli.py rplus --brain brain.npz
```
