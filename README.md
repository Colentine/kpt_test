# 知识点预测评测

支持 **DA-20K 和 XES3G5M**。准备好预测文件后，一条命令即可得到各项指标和作业达标结果，无需加载预测模型。

## 1. 安装与体验

需要 Python 3.10 或更新版本。

```bash
git clone https://github.com/Colentine/kpt_test.git
cd kpt_test
python -m pip install -e .
```

先用仓库自带的合成示例体验完整评测（无需下载数据或模型）：

```bash
python -m kpt_test evaluate --data-dir examples/da20k --predictions examples/da20k.predictions.jsonl --noisy-predictions examples/da20k.noisy.predictions.jsonl --output outputs/report.json --require-complete
```

将命令中的 `da20k` 全部替换为 `xes3g5m`，即可运行另一个示例。示例说明见 [examples/README.md](examples/README.md)。

## 2. 准备数据

已有测试包可跳过本节。原始数据需自行下载，转换时默认按题目 ID 以 **8:1:1** 划分训练集、验证集和测试集，随机种子默认为 42。

**DA-20K**：[数据下载](https://figshare.com/s/2be2eb2c06d00a9e4349) · [官方仓库](https://github.com/xuqiang124/atmk_system)

使用题目表、题目标签关联表和知识点表：

```bash
python -m kpt_test prepare --dataset da20k --questions data/raw/da20k/questions.json --links data/raw/da20k/links.json --labels data/raw/da20k/knowledge.json --output-dir data/da20k
```

将路径替换为实际文件路径，格式参考 [examples/raw](examples/raw)。若题目已包含 `labels` 或 `label_ids` 数组，可省略 `--links`；`--labels` 也支持官方的 `DA-20k-labels-with-father.json`。

**XES3G5M**：[官方仓库与下载地址](https://github.com/ai4ed/XES3G5M)

使用下载后的题目元数据：

```bash
python -m kpt_test prepare --dataset xes3g5m --questions data/raw/XES3G5M/metadata/questions.json --labels data/raw/XES3G5M/metadata/kc_routes_map.json --output-dir data/xes3g5m
```

输出目录需为空。生成的主要文件：

| 文件 | 用途 |
| --- | --- |
| `train.jsonl` / `valid.jsonl` | 训练集 / 验证集，含标签 |
| `test.inputs.jsonl` | 原始测试题目 |
| `test.noisy.inputs.jsonl` | 扰动后的测试题目 |
| `test.gold.jsonl` | 评测用真实标签 |
| `taxonomy.json` | 可预测的末级标签及层级路径 |
| `predictions.template.jsonl` | 待填写的预测模板 |

程序还会保存划分和文件校验信息，请保留整个目录。对比不同模型时使用同一测试包。

语义指标需要生成一次向量缓存，之后可离线评测：

```bash
python -m pip install -e ".[semantic]"
python -m kpt_test build-embeddings --taxonomy data/da20k/taxonomy.json --output data/da20k/embeddings.json
```

默认模型为 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。XES3G5M 将路径中的 `da20k` 改为 `xes3g5m`。已有 `embeddings.json` 时无需重复生成。

## 3. 准备预测文件

输入为 UTF-8 JSONL，一行一道题：

```json
{"qid":"1001","text":"在等差数列中，a1=2，公差为3，求a10。"}
```

将预测模板复制到测试包目录之外，填入结果，保存为 `predictions.jsonl`：

```json
{"qid":"1001","labels":["101","102"]}
```

- 每行只包含 `qid` 和 `labels`，每题预测一个或多个末级标签 **ID**，ID 见 `taxonomy.json`。
- 必须覆盖全部测试题；按 `qid` 对齐，行和标签的顺序不限。
- 不允许重复题目、重复标签、空预测或未知标签。

用同一模型和推理配置对 `test.noisy.inputs.jsonl` 再预测一次，保存为相同格式的 `noisy_predictions.jsonl`。扰动默认以 0.35 的概率删除非空白字符、交换相邻字符。预测时只读取测试输入，不使用真实标签或答案。

## 4. 运行评测

```bash
python -m kpt_test evaluate --data-dir data/da20k --predictions predictions.jsonl --noisy-predictions noisy_predictions.jsonl --output outputs/report.json --details outputs/details.jsonl --require-complete
```

XES3G5M 只需将 `--data-dir` 改为 `data/xes3g5m`。默认读取测试包内的 `embeddings.json`。

终端显示主要结果，`report.json` 保存完整报告，`details.jsonl` 保存逐题结果（`--details` 可省略）。报告包括：

- **严格准确率、宽松准确率**：分别要求全部标签相同、至少命中一个标签；作业门槛为 **90%、95%**。
- **Precision、Recall、Micro-F1、Macro-F1**：Macro-F1 对完整末级标签集合平均，未出现类别按 0 计。
- **层级和语义 P/R/F1**：按层级关系或语义相似度进行一对一最优匹配。
- **交互容错性**：扰动后的 Precision / 原始 Precision；原始 Precision 为 0 时返回 `null`。

没有扰动预测时可省略 `--noisy-predictions`；没有语义缓存时可加 `--skip-semantic`。此时去掉 `--require-complete`，报告会标记指标不完整，缺失指标为 `null`。

更多参数见 `python -m kpt_test evaluate --help`。运行自动测试：

```bash
python -m unittest discover -s tests -v
```

评估流程参考 [SGPE/scripts](https://github.com/Colentine/SGPE/tree/main/scripts)，指标按作业说明实现。
