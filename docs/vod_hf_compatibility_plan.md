# VOD HF/Diffusers 兼容性前瞻设计 + 多模态生成区隔分析

> 整合 2026-05-03 的多轮讨论：LPM 1.0 参照、image/video/audio 算力区隔、HF/SD 适配标准、Codex 与 Claude 联合评估。
> 解决两个问题：(1) 未来上 HF/Diffusers 生态时如何避免 APT 那种"临时大改架构"；(2) 多模态架构如何避免用户只要图却把视频/音频也算了。

---

## 0. 当前 VOD 状态盘点

| 项 | 状态 |
|---|---|
| 架构定位 | type B (substrate-shared field U(t,y,x,c)) 多模态扩散 |
| 参数规模 | ~524K (UNet 三层 + 4 个 1×1 enc/dec Linear) |
| Latent 形状 | (T=8, H=32, W=32, C=4) |
| Gate 0 | 通过：L_recon=0.0285, L_clean_noop=0.0000 |
| Generation | DDPM 训练 + DDIM 采样跑通；sample 是 rough grid，未达 publishable quality |
| 训练数据 | 64 个合成 Chladni 场，static 模式 |

关键代码位置：
- `D:\VOD\prototype\vod_minimal\native.py` — NativeVOD 主类、encode/decode/denoise/denoise_path
- `D:\VOD\prototype\vod_minimal\denoisers.py` — UNetDenoiser (default) + PointwiseMLPDenoiser (legacy)
- `D:\VOD\prototype\vod_minimal\diffusion.py` — NoiseSchedule + ddim_sample
- `D:\VOD\prototype\run_diffusion_train.py` — DDPM 训练入口
- `D:\VOD\prototype\run_gate0_verify.py` — Gate 0 验证

---

## 1. LPM 1.0 参照系（type C，不照抄）

LPM 1.0 是 Anuttacon (米哈游相关) 2026/4 发布的 17B 视频角色表演模型。表情/情绪生成机制（来自论文 §2.4 / §3.1 / §6.4）：

- **架构**：17B DiT，从 Wan2.1-I2V (16B) 初始化，flow matching；每 block: self-attn + AdaLN → cross-attn (text + speak/listen 交错) → FFN + AdaLN
- **表情机制**：1–8 张 facial expression reference 图作 token 拼到 self-attn 序列尾部；用 3D RoPE 偏移区分类型 (expression vs body-view) 和子类型 (happy/sad/...)
- **数据 pipeline**：1080P+ → EmotiEff Lib 检测 8 类表情 → VLM 二次验证标签
- **文本控制**：22 expression bases + 78 emotions 分类法
- **训练阶段**：speak → listen → conversation；30% clip 加 multi-view + expression refs
- **后训练**：DPO 治"frozen frame"和肢体 artifact

**对 VOD 的判断**：
- LPM 是 type C 路线（独立 modality 编码器 + 共享 DiT），与 VOD type B 平行，**不构成对 VOD claim 的支持也不构成反例**
- 上述五大机制现在**全部不该搬到 VOD**：VOD 没 identity 概念、没 audio、没 text encoder、没人脸数据、基础 sample quality 都未达标
- LPM 论文存档为"type C 参照系"，未来若 VOD 进入 conditional generation 阶段可借鉴 multi-reference token + RoPE 偏移这套机制

---

## 2. 多模态算力区隔分析

### 算力分布

| 环节 | 占比 | 能否按 modality 裁 |
|---|---|---|
| DDIM sampling on U(T,H,W,C) | ~99% | ❌ substrate 一体 |
| encode (4 个 1×1 Linear) | <0.1% | 已有 active_media() gating |
| decode head (4 个 1×1 Linear) | <0.1% | 硬编码 image+video 全开 |

**核心结论**：decoder 几乎不花钱（1×1 Linear），花钱全在 substrate 迭代去噪。"用户只要图，VOD 却把视频/音频也算了"——decoder 层不是浪费，**真正浪费是 substrate 必须 sample 完整 (T=8, H=32, W=32, C=4) latent，单帧 image 也走 8 帧反向链**。

### 三种技术路线

| 路线 | 节省 | 代价 | 取舍 |
|---|---|---|---|
| 解码门控 (API 层) | ~0% | 5 行代码 | **现在做**：契约清晰、输出体积小 |
| Substrate 形状自适应 (T 随机训练，inference 按需) | ~8x for image-only | 重训 + UNet bottleneck Conv1d 在 T=1 退化 | **基础 quality 过关后做** |
| Cascade (low-res → high-res refiner) | 中 | 分裂 substrate，伤 type B claim | **不做** |
| Early-exit / ACT | 小 | 三层 UNet 不够深 | **规模上去再说** |

### 是否 type B 原罪？

**不是**。type C (Sora/LPM) 同样问题：单帧 image-only 也得跑完整 DiT 时空 token。差别在 condition 编码，不在 sampling cost。**diffusion 架构通病，不能用来打 type B claim**。

---

## 3. HF/Diffusers 2026 生态调研

| 系统 | 当前版本 | 关键特性 |
|---|---|---|
| **Modular Diffusers** | 2026/3 新发布 | Composable blocks 替代 monolithic pipeline；向后兼容老 Diffusers repo；`modular_model_index.json` 支持跨 repo 组件复用 |
| **PyTorchModelHubMixin** | huggingface_hub v1.12.1 | 继承即得 `save_pretrained / from_pretrained / push_to_hub`；class-level metadata (library_name, repo_url, paper_url, tags); `coders` 字典处理非 JSON 类型 |
| **diffusers ModelMixin** | v0.38 | `_repeated_blocks` regional compile / `enable_layerwise_casting` (含 fp8_e4m3fn) / `enable_group_offload` / `set_attention_backend` / safetensors 默认 |
| **safetensors** | 加入 PyTorch Foundation (2025) | diffusers `save_pretrained` 默认；.bin 已被默认拒收在新生态 |
| **flashpack** | fal-ai 2026 | 新加速加载格式，早期，暂不适配 |

### Modular Diffusers 对 VOD 的关键含义

**Modular 范式让 type B substrate 上 HF 反而比 type C 更容易。**

老 SD 范式强制 `text_encoder + VAE + UNet + scheduler` 固定结构，VOD 套上去等于装假肢、抹掉 type B claim。Modular 范式下 VOD 可作单个 `ModularPipelineBlocks` 子类：
- `expected_components`: NativeVOD substrate
- `inputs`: dict[modality] (image/video/text/audio optional)
- `intermediate_outputs`: dict[modality] (按 requested 返回)
- `__call__(components, state) → (components, state)`

**这意味着原先"L3 完整 diffusers integration 会污染 type B claim"的判断要撤回**——现在 Modular 是友好目标。

---

## 4. Phase 1/2/3 前瞻适配清单（最终执行版）

按 Codex 与 Claude 联合判定，统一为 **3 个 Phase**。每个 Phase 同时含**接口/序列化护栏**和**仓库 layout 工作**两类，在同一时间窗内做，**不再分两套编号**（旧称 Phase A/B/C 与 Phase Repo-1/2/3 已废弃）。

**核心原则：现在不继承 HF 类，但立刻实现 HF-compatible surface（影子契约）。仓库形态整理与接口护栏并行。**

研究里程碑（Gate 0 / DDPM-DDIM / unconditional sample fidelity / scaling / conditional generation）是**质量门槛**不是工程 Phase，不参与本编号体系。

---

### Phase 1 — Now: Repository + Interface Guardrails

#### 1A. 接口/序列化护栏（已完成）

| # | 改动 | 状态 | 不做的代价 |
|---|---|---|---|
| A1 | `NativeVODConfig.to_dict() / from_dict()`，所有字段 JSON-serializable | done | 未来 config.json 报错 |
| A2 | `architecture_version: str = "1.0"` 字段 | done | 未来 checkpoint 不兼容时无法拒绝错误加载 |
| A3 | 薄壳 `save_pretrained(dir) / from_pretrained(dir)`：写 `config.json` + `model.safetensors`（无 safetensors 时 fallback `pytorch_model.bin`），**不依赖 huggingface_hub** | done | 上 Hub 时整套要重写；pickle 安全风险 |
| A4 | **state_dict key contract**：`STATE_DICT_KEYS.md` 锁定命名 + `tests/test_state_dict_contract.py` 检测 rename；rename 必走 `_load_from_state_dict` alias hook | done | **最重要**：发布后 key 改名 = 全部老 checkpoint 失效 |
| A5 | `@dataclass NativeVODOutput(sample=..., latent=...)` 替代 forward 的裸 tuple，**保留 tuple 解包兼容** | done | 加字段就破老 caller |
| A6 | dtype audit：`sinusoidal_time_embedding` / `ddim_sample` 加 optional `dtype` 参数 | done | 未来 bf16/fp16 报 dtype mismatch |
| A7 | `decode(U, *, requested: tuple[str, ...] \| None = None)` 选择性 head dispatch | done | 服务 API 输出契约模糊；用户拿到不需要的 tensor |
| A8 | `model_index.json` / `modular_model_index.json` 草案文件（占位，不实际加载） | done | 未来 Modular Diffusers wrapper 要重新设计 |
| A9 | `docs/hf_modular_mapping.md` 写清未来 block 的 inputs / intermediate_outputs / expected_components 形状 | done | 未来 wrap 时设计反复 |

**1A 不动 forward / loss / denoiser 主逻辑，纯序列化 + 公共 API 层。**

#### 1B. 仓库 layout 工作

| # | 改动 | 状态 | 不做的代价 |
|---|---|---|---|
| L1 | `requirements.txt`（numpy / torch / safetensors / Pillow / pytest，不固定 CUDA wheel URL） | done | clone 用户不知道装什么 |
| L2 | `pyproject.toml`（`pip install -e .`，`pythonpath = ["prototype"]`，包路径不动） | done | 仍要 `sys.path.insert` hack |
| L3 | `scripts/sample.py` 最简 CLI：默认 Gate 0 / Chladni round-trip demo（orig / recon_no_denoise / pipeline 三 PNG），无 checkpoint 时打 UNTRAINED warning | done | 用户上手成本高 |
| L4 | 把 `prototype/run_*.py` / `train_*.py`（19 个）收纳到 `scripts/ablations/`，统一加 portable sys.path block（relative to `__file__`） | done | 顶层散乱像研究生工作目录 |
| L5 | README quickstart 对齐新入口（`pip install -e .` / `pytest prototype/tests` / `scripts/sample.py`） | pending | 文档与代码不一致 |
| L6 | `prototype/tests/test_artifact_metrics.py` 路径常量增加 `ABLATIONS_ROOT`，指向新位置 | done | 测试找不到脚本 |

**1B 不动包路径**：`prototype/vod_minimal/` 保留原位，避免破坏 import 全链。包名迁移留到 Phase 2。

#### 1B 禁止事项（来自 Codex）

- 不引入 `vod/` public package（留到 Phase 2）
- 不大搬 `prototype/vod_minimal`（同上）
- 不写完整 HF `ModelMixin` / `DiffusionPipeline`（留到 Phase 3）
- 不写 `ModularPipelineBlocks`（留到 Phase 3）
- 不改 Chladni field / model / loss 主逻辑（与 Phase 1 正交）

---

### Phase 2 — After Sample Fidelity: Public Demo Readiness

触发条件：unconditional sample fidelity 达到 publishable quality（生成 sample 一眼能识别为 Chladni，不是 grid noise）。

#### 2A. 接口

- HF Spaces custom model demo（Gradio + 自定义模型，**不依赖 diffusers**）
- README + model card（license, intended use, limitations: toy scale）
- 可选继承 `PyTorchModelHubMixin`，把 A3 薄壳替换成正规 mixin

#### 2B. 包名迁移（带 compatibility shim）

1. 新建 `vod/` 作为正式 public package
2. 把 `vod_minimal/` 内容**逐步迁过去**（不一刀切）
3. 保留 `vod_minimal/__init__.py` 作为 compatibility shim：
   ```python
   from vod import *  # noqa: F401,F403
   import warnings
   warnings.warn("vod_minimal is deprecated; use `vod` instead",
                 DeprecationWarning, stacklevel=2)
   ```
4. tests 改用 `from vod import ...`
5. 老脚本仍然 `import vod_minimal` 不立刻炸
6. 几个版本后再正式删除 `vod_minimal`

成本：~3-4h。中等风险（影响 import 全链）。**Claude 第一版"30min 小搬家"判断错误**——一刀切重命名会同时影响 imports / tests / scripts / README quickstart / checkpoint loading / 用户已有命令 / 未来 package 名。Codex 修正：**分阶段 + compatibility shim**。

---

### Phase 3 — HF Hub / Diffusers Integration

触发条件：真要上 HF Hub 发布（Phase 2 demo 受到关注后）。

#### 3A. HF / Diffusers 接入

- 包 `ModularPipelineBlocks` 子类（按 `docs/hf_modular_mapping.md` 契约）
- 包 `ModelMixin` / `SchedulerMixin` wrapper（按需）
- 真实标记 `_skip_layerwise_casting_patterns = (...)` 把 4 个 1×1 enc/dec Linear 列入 skip
- 真实标记 `_no_split_modules = ["UNetDenoiser"]`
- `_repeated_blocks` 标记（如果未来加 Transformer block）

#### 3B. 模块化深拆

- 加 HF-compatible save/load 的 model card 资产（HF 格式 README, NOTICE）
- `vod/` 内拆 `losses/` `data/` `models/` `sampling/` `io/` 子目录
- `vod/constraints/` 收纳 4/e / Binary-Twin / TTNM / AIMP（**只是代码目录，不是理论模块拼装**）
- 抽 `vod/constants.py`、`vod/config.py`

成本：~半天。

#### 3 后的理想发布型结构

```
VOD/
├── vod/
│   ├── __init__.py
│   ├── config.py
│   ├── constants.py
│   ├── models/
│   ├── fields/                 # Chladni, projections substrate
│   ├── projections/            # encoder/decoder heads
│   ├── constraints/            # 注意：只是代码目录，不是理论模块拼装
│   │   ├── four_e.py
│   │   ├── binary_twin.py
│   │   ├── ttnm.py
│   │   └── aimp.py
│   ├── sampling/               # DDPM/DDIM
│   └── io/                     # save_pretrained / from_pretrained
├── scripts/
│   ├── sample.py
│   └── ablations/              # 旧的 run_*.py
├── examples/
├── tests/
├── docs/
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
└── NOTICE
```

**README 必须显式声明**（对齐 memory `feedback_field_vs_modules_framing.md`）：

> `4/e / TTNM / Binary-Twin / AIMP are field constraints/readouts on
> the same Chladni entropy field, not separate generators.`

不能让目录结构暗示 type B substrate 被拆成模块拼装。

---

## 4.5 仓库形态参照（HunyuanVideo 调研）

来源：[Tencent-Hunyuan/HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo)。**不是架构参照，是 repo 形态参照**——HunyuanVideo 是 type C，VOD 是 type B，架构不照搬。

```
HunyuanVideo/                           hyvideo/
├── assets/  (demo 媒体)                ├── __init__.py
├── ckpts/   (权重 + README)            ├── config.py        ← 集中配置
├── hyvideo/ (主包)                     ├── constants.py     ← 集中常量
├── scripts/                            ├── inference.py     ← 集中推理入口
├── tests/                              ├── prompt_rewrite.py
├── utils/                              ├── diffusion/       ← 按职责分子目录
├── LICENSE.txt + Notice                ├── modules/
├── README.md + README_zh.md            ├── text_encoder/
├── sample_video.py    ← 1 个 CLI       ├── vae/
├── gradio_server.py   ← 1 个 web       └── utils/
└── requirements.txt
```

整齐度差距诊断（Phase 1 完成后大部分项已收敛）：

| 维度 | HunyuanVideo | VOD（Phase 1 后） | 剩余差距 |
|---|---|---|---|
| 顶层 layout | 单层包 + 2 入口 | `prototype/` 仍存在；`scripts/sample.py` 已加 | 中（Phase 2 处理） |
| 包内模块化 | 5 个职责子目录 | `vod_minimal/` 仍平铺 | 中（Phase 3 处理） |
| 配置集中 | `config.py` + `constants.py` | `LATENT_T` 仍在 `native.py` | 小（Phase 3 处理） |
| 推理入口数 | 2 | `scripts/sample.py` + `scripts/ablations/` 收纳完成 | 小 |
| 依赖清单 | `requirements.txt` | done | 无 |
| 安装方式 | `pip install` | `pip install -e .` done | 无 |
| LICENSE + Notice | 双文件 | 单 LICENSE | 小（Phase 2 加 NOTICE） |
| 双语 README | 中英 | 单英 | 小（不做） |

---

## 5. 不做清单（明确边界）

- ❌ 不继承 `ModelMixin` / `SchedulerMixin` —— 生态在快速迭代，过早绑定限制研究
- ❌ 不写完整 `DiffusionPipeline` —— 假需求工程
- ❌ 不写 `ModularPipelineBlocks` 子类 —— sample quality 没过关，没 block 可注册
- ❌ 不接 conditioning hook 假参数 —— 还不知道未来 conditioning 形状（text? class? cross-attention?），YAGNI
- ❌ 不动 5-D `(B, T, H, W, C)` substrate —— type B claim 核心；Modular Diffusers 本来就允许任意 tensor shape
- ❌ 不引入 `huggingface_hub` / `diffusers` 作为依赖 —— 仅做接口形状兼容，依赖延后到真要上传时
- ❌ 不为 HF 适配动 Chladni field / denoiser / loss 主逻辑

---

## 6. APT 大改根因（猜测）+ 这次如何避免

按搜到的当前 HF API 看，APT 当时大改最可能两条触发点：
- **state_dict key 命名变更**（对应清单 A4）—— 发布后改名 = 全部老 checkpoint 失效
- **forward 返回 tuple 不能塞新字段**（对应清单 A5）—— 加 condition / latent 输出时 break 全部下游

Phase 1 / 1A 优先级排序就是按这两条致命点设计的。其他项（safetensors / dtype / decode gating）都是次级。

---

## 7. 实施顺序与触发条件

```
[Phase 1: 1A 接口护栏 9 项 + 1B repo layout 6 项] ── 现在做
        │
        ▼
[主线: unconditional sample fidelity] ── 当前阻塞点（研究里程碑，非 Phase）
        │
        ├─► [Phase 2: 2A HF Spaces demo + 2B 包名迁移 vod/] ── sample 能见人后
        │         │
        │         ▼
        └─► [conditional generation: text/class] ── scale 验证后（研究里程碑）
                  │
                  ▼
            [Phase 3: 3A Modular Diffusers wrapper + 3B 模块化深拆] ── 真要上 Hub 时
```

**Phase 1 与主线研究并行，互不阻塞**。Phase 2/3 由 sample quality 触发，不由日历触发。研究里程碑（sample fidelity / scaling / conditional generation）不是 Phase。

---

## 7.5 实施分工与 Agent 派发评估

派 Agent 的可行性按改动类型分级：

| 类型 | 项 | 派 Agent | 理由 |
|---|---|---|---|
| 序列化层 | A1, A2, A3 | 否 | overhead > 收益；Agent 容易引多余依赖（如擅自 import huggingface_hub）违反 Phase 1 原则 |
| Key contract | A4 | **否（最关键）** | 一旦发布永久锁定；必须懂 type B substrate 全局 + APT 教训上下文，无本对话上下文的 Agent 容易跑偏 |
| API 契约 | A5, A7 | 否 | "保留 tuple 解包兼容"、"复用现有 active_media() 契约"等细节，Agent 容易破老 caller |
| 代码审查 | A6 dtype audit | 否（写）/ 是（调研） | Agent 容易把必要的 `.float()`（如 `to_png` 里的 numpy 转换）一并删 |
| 文档/JSON | A8, A9 | 否 | 30min 事；A9 需要 LPM 论文 + Modular Diffusers 契约双向理解，Agent 重新查浪费 |
| **read-only 调研** | dtype 硬编码扫描 / forward 调用点 / state_dict 接口点列表 | **是 (Explore)** | 零风险，节省 ~20min 扫文件时间 |

### Agent 派发决策原则（针对接口护栏类任务）

1. 任务规模 < 3 小时整批改动 → 自己做
2. 涉及"发布后不可改"的契约决定 → 自己做
3. read-only 调研 + 列表/统计 → 派 Explore
4. write 类改动只在任务可完整自包含 + 风险小时才派 general-purpose
5. **不为协作而协作**：拆任务给 Agent 反而碎片化的项不派

### 实际分工

| 工作 | 执行方 | 形态 |
|---|---|---|
| Phase 1 主体 (1A A1–A9 + 1B L1–L6 写改) | 自己 | 1A 完成于初始 commit (9a3c38c)；1B 在 Phase 1 收尾时合并 |
| 并行调研 | Explore agent | 一次性扫 `D:\VOD\prototype\` 列出 dtype 硬编码 + `forward` 调用点 + `state_dict` 接口点 |
| 调研报告用途 | A4/A5/A6 实施依据 | 不让 Agent 写代码 |

---

## 8. 关键判断归档

| 问题 | 判断 | 理由 |
|---|---|---|
| 现在是否套 SD pipeline 形状？ | 否 | 强行套 = 装假肢，抹 type B claim |
| 现在是否继承 ModelMixin？ | 否 | 生态迭代快，过早绑定 |
| Modular Diffusers 是否友好？ | 是 | 不强求固定 component 拼装，type B 反而自然 |
| 多模态算力浪费如何解？ | 短期门控 + 长期 substrate 形状自适应 | 真正算力大头在 substrate sampling，不在 head |
| LPM 机制是否搬到 VOD？ | 否 | type C 平行路线；VOD 基础能力都未达标 |
| safetensors 是否现在切？ | 是 | 已是 diffusers 默认；.bin 在新生态被拒收 |
| state_dict key 是否现在锁定？ | 是 | 发布后改名 = breaking change，APT 教训 |
| 是否照抄 HunyuanVideo 架构？ | 否 | HunyuanVideo 是 type C；VOD claim 是 type B；只参照 repo 形态，不照搬架构 |
| 包名 `vod_minimal → vod` 现在做？ | 否（Phase 2 / 2B 才做） | Claude 第一版"30min 小搬家"判断错误；要带 compatibility shim 分阶段迁 |
| `requirements.txt` / `pyproject.toml` 现在做？ | 是（Phase 1 / 1B） | 任何对外 repo 的最低门槛，零风险 |
| 19 个 `run_*.py` / `train_*.py` 现在收纳？ | 是（Phase 1 / 1B） | 顶层散乱让 repo 看起来像研究生工作目录 |
| Phase 编号 A/B/C + Repo-1/2/3 是否合并？ | 是（合并为 Phase 1/2/3） | 6 个 Phase 标签让记忆负担过重；两轴时间点本来就对齐 3 个触发点 |
| `constraints/` 子目录会不会暗示 module 拼装？ | 风险存在 | README 必须显式写"field constraints/readouts on same field, not separate generators" |

---

## Sources

- [Modular Diffusers (HF Blog, 2026/3)](https://huggingface.co/blog/modular-diffusers)
- [ModularPipelineBlocks docs](https://huggingface.co/docs/diffusers/modular_diffusers/pipeline_block)
- [HF Hub mixins (huggingface_hub v1.12.1)](https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins)
- [diffusers ModelMixin v0.38](https://huggingface.co/docs/diffusers/api/models/overview)
- [PyTorchModelHubMixin guide](https://huggingface.co/blog/not-lain/building-hf-integrated-libraries)
- [Safetensors joins PyTorch Foundation](https://huggingface.co/blog/safetensors-joins-pytorch-foundation)
- [LPM 1.0 paper (arxiv 2604.07823)](https://huggingface.co/papers/2604.07823)
- [LPM project page](https://lpm-ai.org/)
- [HunyuanVideo official repo (Tencent-Hunyuan)](https://github.com/Tencent-Hunyuan/HunyuanVideo)

---

*文档版本：v1.1 (2026-05-04)*
*更新历史：*
- *v1.0 (2026-05-03) 初版：LPM 1.0 参照、多模态算力区隔、HF/Diffusers 2026 调研、初版接口护栏清单（旧称 Phase A/B/C）*
- *v1.1 (2026-05-04) 新增仓库发布形态章节（旧称 Phase Repo-1/2/3，HunyuanVideo 对比，Codex 修正包名迁移成本判断），关键判断归档新增 5 条*
- *v1.2 (2026-05-05) 统一为 Phase 1/2/3 编号体系。废弃 Phase A/B/C 与 Phase Repo-1/2/3 旧称。§4 与 §4.5 合并为单一 §4，每个 Phase 同时含 1A/1B/2A/2B/3A/3B 接口+layout 工作。Phase 1 / 1B 完成（requirements / pyproject / sample.py / scripts/ablations / 测试 path 修复 / 235 tests pass）*

*维护：Phase 1 完整收尾后回填实际 commit hash；Modular Diffusers API 若有不兼容更新需重审第 3/4 节*
