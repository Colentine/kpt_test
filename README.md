# 知识点预测作业评测

根据《知识点预测算法设计作业说明》实现，支持 **DA-20K / XES3G5M**。学生只需要对教师给定的测试题生成预测文件，即可计算严格准确率、宽松准确率、Precision、Recall、Micro-F1、Macro-F1、层级 P/R/F1、语义 P/R/F1 和交互容错性。模型训练框架不限，评测无需加载学生模型。

## 1. 安装与快速体验

Python 3.10 或更新版本；Windows、Linux 均可。

```bash
git clone https://github.com/Colentine/kpt_test.git
cd kpt_test
python -m pip install -e .
python -m kpt_test --help
```

仓库的 `examples/da20k` 和 `examples/xes3g5m` 是人工构造的格式示例，**不是真实数据集或模型实验结果**。它们包含测试包、原始/扰动预测示例和语义缓存；基础安装即可离线运行全部指标。

```bash
python -m kpt_test evaluate --data-dir examples/da20k --predictions examples/da20k.predictions.jsonl --noisy-predictions examples/da20k.noisy.predictions.jsonl --output outputs/da20k.report.json --details outputs/da20k.details.jsonl --require-complete

python -m kpt_test evaluate --data-dir examples/xes3g5m --predictions examples/xes3g5m.predictions.jsonl --noisy-predictions examples/xes3g5m.noisy.predictions.jsonl --output outputs/xes3g5m.report.json --require-complete
```

两个示例的精确匹配预期均为：Strict=0.50，Loose=0.75，Precision=Recall=Micro-F1=0.60，Macro-F1=7/15≈0.466667，交互容错性=1.25。语义分数取决于分发的固定向量，缓存中记录了模型来源。示例故意包含错误预测，因而不满足作业准确率门槛。

## 2. 学生使用

教师为每个数据集准备并分发一个固定测试包。学生读取 `test.inputs.jsonl`，用自己的模型预测；再读取 `test.noisy.inputs.jsonl`，**用同一个模型和相同推理配置重新预测**。不能仅通过修改原始预测标签来代替扰动文本上的推理。

### 测试输入

UTF-8 JSONL，一行一道题：

```json
{"qid":"1001","text":"在等差数列中，a1=2，公差为3，求a10。"}
```

`text` 包含题干、必要公式和选项；不包含答案、解析或知识点路径。数据中引用的图片占位符原样保留，若模型使用图片，教师还需提供原数据的图片目录。本工具不做 OCR 或图片转文本。

允许预测的标签及其名称见 `taxonomy.json`。**统一提交标签 ID**，不提交标签名称、自由生成文本或概率。标签 ID 按字符串处理；整数输入也会转为字符串，`"001"` 与 `"1"` 不同。

### 学生预测输出

将 `predictions.template.jsonl` 复制到测试包之外的文件，填入模型输出：

```json
{"qid":"1001","labels":["101","102"]}
{"qid":"1002","labels":["103"]}
```

每行必须恰好有 `qid` 和 `labels` 两个字段。`labels` 必须为一个或多个合法末级标签 ID，顺序不影响结果。预测行可以乱序，评测按 `qid` 对齐；题目必须完整覆盖测试集，禁止缺失、额外题目、重复题目、重复标签和体系外标签。也支持 `.json` 对象数组，字段相同。

原始输入的预测保存为 `predictions.jsonl`，扰动输入的预测保存为 `noisy_predictions.jsonl`，两者使用相同题目 ID。

```bash
python -m kpt_test evaluate --data-dir data/da20k --predictions predictions.jsonl --noisy-predictions noisy_predictions.jsonl --output outputs/report.json --details outputs/details.jsonl --require-complete
```

XES3G5M 仅将 `--data-dir` 替换为相应测试包目录。数据集身份来自包内元数据，无需改评测代码。

默认自动读取包内的 `embeddings.json`；也可用 `--embeddings path/to/embeddings.json` 指定教师提供的缓存。学生无需安装语义模型或联网。

只有原始预测时，可以先不传 `--noisy-predictions`；尚无语义缓存时，必须显式加 `--skip-semantic`。这两种情况会输出已有指标，并将 `evaluation_complete` 设为 `false`、缺失指标设为 `null`。正式提交使用 `--require-complete`。

默认拒绝空预测。调试弃权行为时可用 `--allow-empty-predictions`，空集合按 0 分计算，并将 `requirements.nonempty_predictions` 设为 `false`；这不满足作业的“一个或多个知识点”格式要求。

### 输出约定

终端打印主要结果，`--output` 保存完整 JSON；`--details` 可选，保存逐题 JSONL。所有比例使用 **0 到 1** 的小数（交互容错性可能大于 1），不先舍入再判断达标。

| JSON 字段 | 含义 |
| --- | --- |
| `dataset` / `evaluator_version` | 数据集名称与评测程序版本 |
| `evaluation_complete` | 是否提供了语义资源及扰动预测；不表示准确率达标 |
| `metrics.strict_accuracy` / `loose_accuracy` | 严格/宽松准确率 |
| `metrics.exact.precision` / `recall` / `micro_f1` / `macro_f1` | 精确标签匹配指标 |
| `metrics.hierarchical` / `semantic` | 各自的软 TP、FP、FN 及 P/R/F1 |
| `metrics.per_label` | 每个末级类别的支持数及精确匹配 P/R/F1 |
| `interaction_robustness.ratio` | 扰动 Precision / 原始 Precision |
| `requirements.thresholds` | Strict≥0.90、Loose≥0.95 的实测值及布尔判定 |
| `requirements.accuracy_thresholds_passed` | 两项准确率门槛是否同时满足；不代表整个实验作业通过 |
| `semantic_resource` / `inputs` | 语义模型信息及输入文件 SHA-256，便于复核 |

退出码：`0` 表示评测成功；`2` 表示输入、配置或文件错误；加 `--require-complete` 后指标不全返回 `1`；加 `--fail-on-threshold` 后未达到准确率门槛返回 `1`。退出码 `1` 时仍保存评测报告。

## 3. 教师准备真实测试包

原始数据不随仓库再分发。教师下载后执行一次转换、冻结测试划分及语义资源，让全班使用同一个版本。学生不能自行重划测试集、调整测试标签集合或更换评分语义模型。

### DA-20K

来源：[官方仓库](https://github.com/xuqiang124/atmk_system)、[原始数据下载](https://figshare.com/s/2be2eb2c06d00a9e4349)。兼容两种格式。

**格式 A：SGPE 使用的分表格式。**

```json
// questions.json（实际 JSON 文件中不要写注释）
[{"id":1,"text_processed":"判断两个集合的关系。"}]
// links.json
[{"qid":1,"label_id":3}]
// knowledge.json
[{"id":1,"name":"集合","uuid":"u1","parent_uuid":""},
 {"id":3,"name":"子集","uuid":"u3","parent_uuid":"u1"}]
```

```bash
python -m kpt_test prepare --dataset da20k --questions data/raw/da20k/latex_aligned_to_roberta.json --links data/raw/da20k/math_questions_knowledgetag.json --labels data/raw/da20k/math_questions_knowledge.json --output-dir data/da20k --seed 42
```

题目文本按 `text_processed`、`text`、`content`、`question` 的优先级读取；题目 ID 为 `qid` 或 `id`。知识树支持 `uuid/parent_uuid` 或 `id/parent_id`，空父节点应使用 `null` 或空 `parent_uuid`。悬空父节点和环会报错。若发布版本文件名不同，传入实际路径。

**格式 B：题目内嵌标签 + 官方父类标签表。**

```json
[{"id":1,"content":"判断两个集合的关系。","labels":[3]}]
```

使用官方 `DA-20k-labels-with-father.json`（字段为 `label_id/name/label_father_id/label_father_name`）：

```bash
python -m kpt_test prepare --dataset DA-20K --questions data/raw/da20k/questions.json --labels data/raw/da20k/DA-20k-labels-with-father.json --output-dir data/da20k --seed 42
```

内嵌标签也接受 `label_ids`。同一题的父标签只有在同时存在对应末级后代标签时才被去掉，并记录数量；仅有父标签会报错。官方两层父类表只能体现同父节点关系，评测不会补造更高层祖先。若使用完整知识树，则按其实际层级评分。HDF5 token 文件和 pickle 多热映射不是此转换器的输入，应使用原始题目文本、ID 和标签表。

### XES3G5M

来源：[官方仓库及 Download](https://github.com/ai4ed/XES3G5M)。使用题目元数据，不使用学生答题序列。

```bash
python -m kpt_test prepare --dataset XES3G5M --questions data/raw/XES3G5M/metadata/questions.json --labels data/raw/XES3G5M/metadata/kc_routes_map.json --output-dir data/xes3g5m --seed 42
```

`questions.json` 是题目 ID 到对象的映射，每题含 `content`、`kc_routes`、可选 `options`。`kc_routes_map.json` 是知识点 ID 到完整知识点路径的映射。路径可为名称数组，或以 `----`、`->`、`>`、`/`、`::` 分隔的字符串；也可用 `--route-separator` 明确指定分隔符。`kc_routes` 是路径列表，单个路径也要放入列表。

同名末级知识点用完整路径消歧，内部祖先用完整路径标识，避免不同分支同名节点误匹配。标签 ID 使用原映射 ID，不重新编号。若只有叶子名称，只有在整个词表中名称唯一时才允许解析。映射中包含内部节点时，去掉作为其他路径前缀的节点；题目仅标内部节点而无末级后代时会报错。

### 固定划分与文件

默认按题目 ID 的 `SHA256(seed:qid)` 排序后按 8:1:1 划分，边界为 `floor(0.8N)` 与 `floor(0.9N)`；源文件行顺序变化不会改变划分。该策略是确定性随机划分，不保证每个罕见标签都出现在测试集。自动划分要求至少 10 道题。

已有划分时用 `--splits splits.json`，内容为：

```json
{"train":["1","2"],"valid":["3"],"test":["4","5"]}
```

三个集合必须不重叠且恰好覆盖输入题目 ID；测试集不能为空。不接受用同一道题的多条作答记录作为不同样本。转换器遇到无文本、无标签或未知标签会报错，不会静默丢弃题目。

生成目录：

```text
data/da20k/
  train.jsonl                  # 训练题目及标签
  valid.jsonl                  # 验证题目及标签
  test.inputs.jsonl            # 原始测试输入，只有 qid/text
  test.noisy.inputs.jsonl      # 扰动测试输入，只有 qid/text
  test.gold.jsonl              # 评测程序使用的真实标签
  predictions.template.jsonl   # 待填预测，空 labels 不可直接提交
  taxonomy.json                # 允许预测的末级标签及路径、定义
  splits.json                  # 固定题目划分
  manifest.json                # 文件摘要、随机种子、扰动参数及转换统计
  embeddings.json              # 下一步生成的语义缓存
```

`prepare` 拒绝覆盖非空输出目录。评测会验证测试包的文件摘要，以发现混用版本或意外修改。摘要不是防作弊机制，正式评分应在教师端使用教师保存的真实标签及缓存。需要学生本地自测时可分发包含真实标签的完整包；隐藏测试时只给学生输入和标签体系，收到预测后由教师运行评测。不要将 `test.gold.jsonl`、答案或解析传给学生模型。

### 生成语义缓存

只需教师安装一次额外依赖并下载模型：

```bash
python -m pip install -e ".[semantic]"
python -m kpt_test build-embeddings --taxonomy data/da20k/taxonomy.json --output data/da20k/embeddings.json
python -m kpt_test build-embeddings --taxonomy data/xes3g5m/taxonomy.json --output data/xes3g5m/embeddings.json
```

默认模型为作业指定的 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，默认 CPU。可用 `--revision` 固定模型 commit，或用 `--model` 指定本地模型目录。**正式评测应固定并分发同一份缓存**；报告记录该文件哈希，即使模型库后续更新也能复核同一组向量。

编码文本为 `名称 [SEP] 定义（有则加） [SEP] 根 / ... / 末级名称`。原数据无定义时不生成虚构定义；DA 知识表可提供 `definition`。教师若补充定义，应在生成测试包和冻结缓存前完成。缓存校验完整标签覆盖、维度一致、非零且有限向量、标签体系摘要。模型加载失败会明确报错，不使用随机向量替代。

## 4. 指标口径

- **Strict Accuracy**：预测集合与真实集合完全相等的题目比例。
- **Loose Accuracy**：至少命中一个真实标签的题目比例。
- **Precision / Recall / Micro-F1**：汇总所有题的精确匹配 TP、FP、FN 后计算。
- **Macro-F1**：先逐标签计算 F1，再对 `taxonomy.json` 中的**全部允许末级标签**平均。没有真实也没有预测的类别 F1=0，因此测试集只覆盖部分词表时，即使所有题都预测正确，Macro-F1 也可能小于 1。输出包含 `macro_scope` 和逐类统计，避免混淆为只对出现类别或题目平均。
- **层级 P/R/F1**：完全相同=1，同父=0.5，同祖父=0.2，其他共享祖先=0.05，否则=0。先在每道题的预测与真实标签间做最大权重一对一匹配，得到软 TP；FP=预测标签数−TP，FN=真实标签数−TP，最后跨题 micro 汇总。
- **语义 P/R/F1**：配对分数为 `max(0, cosine(ep, et))`，数值误差截断到 [0,1]。同样使用一对一最优匹配后 micro 汇总。
- **交互容错性**：原始与扰动输入分别产生预测，用精确标签匹配的 `Precision(noisy)/Precision(clean)`。不截断到 1；若原始 Precision 为 0，则数学上不可定义，返回 `null` 和 `undefined_zero_baseline`，同时保留两项 Precision。指标齐全但比值不可定义时，`evaluation_complete` 仍为 `true`。

所有 P/R/F1 的零分母按 0 处理，交互容错性比值除外。用 SciPy 的最大权重线性分配求最优匹配，预测标签再多也不能对一个真实标签重复计分。

扰动沿用 SGPE 的字符规则：文本长度≤4 不变；每个非空白字符以 0.35 概率删除，若剩余字符少于原长的三分之一（向下取整且至少为 1），则撤销本轮删除；随后扫描相邻的两个非空白字符，以 0.35 概率交换，交换后跳过这两个位置。空白不删除。每题使用由 seed 和 qid 派生的独立随机数流，重复运行及重排题目可复现。可用 `--drop-prob/--swap-prob` 改教师侧设置，实际参数写入 manifest。字符扰动可能破坏公式，这是测试输入鲁棒性的一部分。

## 5. 验证与参考

```bash
python -m unittest discover -s tests -v
```

测试覆盖手算指标、最优匹配反例、超过 20 标签的一对一匹配、语义余弦、异常向量、两种数据适配、划分隔离与复现、输入完整性、报告与退出码。GitHub Actions 在 Windows/Linux、Python 3.10/3.13 上运行离线测试。

实现参考 [SGPE/scripts](https://github.com/Colentine/SGPE/tree/main/scripts) 中的 `kp_eval_metrics.py`、`generate_llm_labels.py` 和 `prepare_rr2qc_data.py`，核对版本为 `3dd52245740ab878c4a62baf957c336e880b9e09`。本仓库独立实现评测，不依赖 SGPE 仓库及训练环境。与参考脚本的差别：所有规模都求全局最优匹配；Macro-F1 固定对完整词表平均；缺失语义资源明确报错或显式跳过；提交文件严格校验。作业说明的指标定义优先。

仓库没有运行学生模型，也不承诺其达到 90%/95% 门槛。完整真实数据上的模型得分取决于学生提交的预测。
