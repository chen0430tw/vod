# 分层视觉启蒙数据集（Hierarchical Visual Bootstrapping Dataset, HVBD）

分层视觉启蒙数据集是一套专为大规模视觉/多模态扩散模型预训练而设计的数据集，其核心思想在于以分层结构呈现视觉信息的基本组成单元，从最基础的几何原子到完整的自然彩色图像表达，逐步引导模型建立起视觉构造法则与形式几何逻辑。

它是 NLP 域 **HLBD（分层语言启蒙数据集）** 在视觉域的对应物，命名上延续 `HL_D` 字母模式（HLBD → HL**V**D，将 *Bilingual* 替换为 *Visual*）。设计意图是让 VOD 这类 type-B substrate-shared diffusion 模型在预训练初期就能从最简单的视觉原子（点、线、形状、emoji）逐级过渡到完整的真实场景，类似学龄前儿童的视觉启蒙阶段。

---

## 设计理念

* **层级化结构**：
  HVBD 将视觉表示分为多个层次，每一层都聚焦于不同的视觉信息。初始层包含最基础的"形状卡"与 emoji 表示，帮助模型捕捉直观的视觉原子；接着依次引入简单纹理、频域/形式几何结构、轮廓 sketch、单通道（灰度/luminance）、完整自然 RGB 图像，最终达到多视角与多风格变体。这种从简到繁的设计类似于 NLP 域的 HLBD 启蒙阶段，使 substrate 能迅速掌握视觉信息的基本构造规则。

* **高密度信息编码**：
  数据集通过将各层信息用明确分隔符（例如 `<sep>` / `level_n` 字段）连接，形成高密度信息样本。每个样本不仅提供了原始 RGB 图像，还附带了结构化的几何、频域和跨视图标注，确保模型在预训练初期就能捕捉到多层次的视觉规律。

* **超形式几何与频谱学支撑**：
  HVBD 的设计借鉴了超形式数学和视觉特性对比矩阵的理论，将基础几何原子、纹理、频域基、边缘特征、灰度通道形式化、向量化，从而为模型学习过程中视觉单元间的非线性关系和动态权重链提供数学支撑。这种方法有助于 substrate 在训练中更快地内化视觉构造规律——尤其是 type-B 设计下 substrate `U(t,y,x,c)` 直接 patch-locally 处理的特性，本身就跟"分层视觉原子"的设计哲学吻合。

---

## 数据集结构

HVBD 主要分为以下八个层级，每个层级都承载着不同层次的视觉信息：

* **形状卡层（L1）**：
  提供单个几何原子（点、直线、圆形、方块、三角形、波形 ...）及其对应的 emoji glyph，类似于学前形状卡，帮助模型初步认识最基本的视觉原子和构图元素。

* **纹理层（L2）**：
  由简单纹理与平铺 pattern 构成（条纹、网格、渐变、Voronoi、Perlin noise、Chladni 驻波），展示基础的视觉组合规则。

* **数学层（L3）**：
  通过频域基底和形式化几何描述（DCT 系数 peak / Sobel 方向梯度 / Laplacian 锐度 / FFT 谱），描述图像的结构性骨架，帮助模型理解视觉的"语法"——空间频率成分与方向选择性。

* **轮廓层（L4）**：
  将彩色自然图转换为线稿/sketch/Canny 边缘表示，提供"视觉发音"——纯结构而无填色的外观骨架。这一层对应 HLBD 的拼音层（surface 表征）。

* **灰度层（L5）**：
  以单通道亮度（luminance / grayscale / depth-only / Y/Cr/Cb split）对应彩色 RGB 表达，构建跨通道的映射关系。这一层对应 HLBD 的英文层（同 concept 的另一种"语言"——通道维度的另一种表达）。

* **彩色层（L6）**：
  展示完整、自然的 RGB 图像和场景，为模型生成连贯彩色输出提供目标。对应 HLBD 的中文层。

* **视角变体层（L7）**：
  同一 concept 的不同视角/光照/距离/裁切表达，对应 HLBD 的日文层（同义异形）。

* **风格变体层（L8）**：
  同一 concept 的不同绘画/摄影/草图/水彩/油画风格，对应 HLBD 的韩文层（再一种同义异形）。

---

## 应用场景

* **预训练与快速视觉启蒙（quickcook 视觉版）**：
  HVBD 可用于加速大规模扩散模型的预训练，使 substrate 从最初的几何原子学习迅速过渡到复杂彩色场景生成。通过层级化学习，模型能够在较短时间内捕捉到视觉的核心构成规则和几何结构。这与 APT-Transformer 在 NLP 域用 HLBD 实现速食预训练（pretrain_quickcook）是同一方法论在不同模态的实例化。

* **多视图 / 多风格融合**：
  数据集不仅包含 RGB 自然图像，还引入轮廓、灰度和多视角 / 多风格变体，帮助模型理解视觉信号在不同表征下的共性和差异，从而支持 cross-view、cross-style、cross-depth 任务的训练和生成。

* **形式几何与频谱逻辑**：
  通过明确的频域和形式几何标注，HVBD 能帮助模型建立起视觉构造法则和形式逻辑的内在关系，这对于后续高级任务（如视觉推理、layout-aware 生成、可控生成）具有重要意义。

* **type-B vs type-A 公平对比基准**：
  HVBD 同时是 VOD（type-B substrate-shared）与 LDM（type-A modality-private VAE）的公平比较测试床。VOD 整张数据集一次性进 substrate；LDM 必须按 domain 拆分为 16 个子集分别训练 VAE。两者在同一份源数据下的 compute / storage / cross-domain coherence 差距，直接体现了类型 B 的工程经济优势。详见 `docs/single_image_anchor_design.md` 第 4.5 节"LDM-baseline split"。

---

## 与其他领域的关系

* **视觉学（vision science）**：
  HVBD 将视觉拆解为最基本的构件，符合 vision science 对视觉原子（line / edge / blob / texture / shape / color / depth）的基本划分，为视觉感知研究提供结构化数据支持。

* **频谱学 / 信号处理**：
  数据集中的形式化标注（如 DCT 谱、Sobel 梯度、Laplacian 锐度）为探讨视觉信号-意义映射提供了定量工具，契合信号处理对视觉构造的精确定义和频域关系的研究。

* **符号学 / 图形学**：
  HVBD 强调视觉符号系统的构建，通过分层数据展示几何原子、纹理、频域基、轮廓与彩色场景之间的相互关系，提供了对视觉生成、传递和解释的新视角。

* **生成模型（diffusion / GAN / VAE）**：
  作为预训练数据集，HVBD 通过分层结构和高密度信息编码，帮助大规模扩散模型快速掌握视觉基本规律和复杂语义，提升图像生成、视频生成和理解的效率和质量。

---

## 总结

**分层视觉启蒙数据集（HVBD）** 以分层、结构化、超形式的方式呈现视觉信息的各个基本层次：从最初的几何原子和形状卡（如 emoji 与 primitive shape），到纹理、频域、轮廓、灰度，再到完整的彩色图像、多视角变体、多风格变体，为大规模扩散模型预训练提供了极具信息密度的数据基础。它不仅帮助 substrate 迅速捕捉视觉构造法则和几何结构，同时通过多视图、多风格和形式逻辑标注，支持跨视图学习和深层场景生成。HVBD 的设计体现了视觉学、频谱学、符号学与生成模型的交叉融合，为实现更高效、更精准的视觉学习提供了全新的理论与实践框架。

它对 type-B substrate-shared 设计（VOD）的核心价值是：**substrate 仅需要"看过这一份数据集"就能产生 anchor，从而触发速食训练**——LDM 等 type-A 模型必须按 domain 切分数据集分别训练 VAE，这是 HVBD 直接揭示的 type-B vs type-A 工程经济差距。

By: 430 + Claude Opus 4.7 (1M ctx)

---

## Sample 示例

```python
samples = [
    {
        "concept": "苹果",
        "level_1": {"primitive": "圆形 + 短弯线（茎）", "shape_glyph": "🍎"},
        "level_2": {"texture": "stippled-red-gradient + smooth-skin"},
        "level_3": {"math": "DCT peak at low-freq red channel; "
                            "Sobel: closed convex contour; "
                            "Laplacian: smooth interior, sharp boundary"},
        "level_4": {"sketch": "apple_outline.png  (Canny edge)"},
        "level_5": {"grayscale": "apple_grayscale.png  (luminance only)"},
        "level_6": {"rgb": "apple_natural.png  (full RGB photograph)"},
        "level_7": {"variations": [
            "apple_side.png", "apple_top.png", "apple_cut.png"
        ]},
        "level_8": {"styles": [
            "apple_sketch.png", "apple_watercolor.png",
            "apple_oil_painting.png", "apple_photograph.png"
        ]}
    },
    {
        "concept": "下雨",
        "level_1": {"primitive": "短斜直线条纹", "shape_glyph": "🌧️"},
        "level_2": {"texture": "vertical-streak-pattern"},
        "level_3": {"math": "high-freq vertical Sobel-y; "
                            "FFT peak in vertical-line basis"},
        "level_4": {"sketch": "rain_outline.png"},
        "level_5": {"grayscale": "rain_grayscale.png"},
        "level_6": {"rgb": "rain_natural.png  (rainy street RGB)"},
        "level_7": {"variations": [
            "rain_heavy.png", "rain_drizzle.png", "rain_storm.png"
        ]},
        "level_8": {"styles": [
            "rain_woodblock.png", "rain_watercolor.png",
            "rain_photo.png", "rain_anime.png"
        ]}
    },
    {
        "concept": "汽车",
        "level_1": {"primitive": "矩形 + 两个圆形", "shape_glyph": "🚗"},
        "level_2": {"texture": "metallic gradient + tire-rubber pattern"},
        "level_3": {"math": "horizontal symmetry axis; "
                            "DCT peak at mid-freq horizontal; "
                            "rectangle-with-rounded-corners contour"},
        "level_4": {"sketch": "car_outline.png"},
        "level_5": {"grayscale": "car_grayscale.png"},
        "level_6": {"rgb": "car_natural.png  (parked sedan RGB)"},
        "level_7": {"variations": [
            "car_front.png", "car_side.png", "car_three_quarter.png"
        ]},
        "level_8": {"styles": [
            "car_blueprint.png", "car_concept_art.png",
            "car_photograph.png", "car_anime.png"
        ]}
    },
    # ... (HVBD v1 = 64 concepts, each with 8 levels = 64×8 ≈ 512+ images
    #      organized into the 2048×2048 master mosaic per
    #      docs/single_image_anchor_design.md §2)
]
```

---

## 跟 HLBD 的字段对应表

| HVBD 字段 | HLBD 字段 | 对应解释 |
|-----------|-----------|---------|
| level_1 形状卡 + emoji | level_1 字卡 + emoji | 最基础的视觉/语言原子 |
| level_2 纹理 / pattern | level_2 短语 | 基础组合规则 |
| level_3 频域 / 形式几何 | level_3 数学 (S=NP+VP) | 形式化结构 |
| level_4 轮廓 sketch | level_4 拼音 | 表层结构（无内容填充） |
| level_5 灰度 / 单通道 | level_5 英文 | 跨表征翻译 |
| level_6 完整 RGB | level_6 中文 | 母语完整表达 |
| level_7 多视角变体 | level_7 日文 | 同义异形 #1 |
| level_8 多风格变体 | level_8 韩文 | 同义异形 #2 |

字段一一映射保证 HVBD 是 HLBD 在视觉域的真正同构 mirror。任何使用 HLBD 的 NLP 训练 pipeline（curriculum hot-swap、anchor + dilution、`<sep>`-joined dense encoding）都能 1:1 移植到 HVBD 的视觉训练 pipeline。

---

## 参考

* `docs/single_image_anchor_design.md` — HVBD 的 anchor-image 浓缩形态设计（2048×2048 master + 16 sub-tiles）。
* `docs/omni_diffusion_lessons_for_vod.md` — HVBD 在 Stage 3 中作为 substrate-priming 工具的位置。
* `docs/paper_v16_baseline.md` — VOD 主论文，HVBD 章节将作为 §8.6 单独 contribution。
* APT-Transformer `apt/trainops/scripts/pretrain_quickcook.py` — HLBD 在 NLP 域的训练 pipeline 实现，HVBD 视觉版基于此 1:1 移植。
