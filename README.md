# 知识点预测评测

支持 **DA-20K 和 XES3G5M**。你需要提供模型预测出的知识点，程序将它们与真实知识点对比，输出分数。

整个流程是：**原始数据 → 程序生成待测题目 → 你的模型预测 → 程序评分**。下面按这个顺序操作。

## 第一步：安装

需要 Python 3.10 或更新版本。在终端依次执行，后续命令都在 `kpt_test` 目录下运行：

```bash
git clone https://github.com/Colentine/kpt_test.git
cd kpt_test
python -m pip install -e .
```

## 第二步：让程序生成待预测的题目

下载原始数据后，选择对应数据集的命令执行一次。将命令中的文件路径替换为实际路径。

**DA-20K**：[数据下载](https://figshare.com/s/2be2eb2c06d00a9e4349)。输入题目表、题目与知识点的关联表、知识点表，格式见[原始数据说明](docs/prepare_data.md)。

```bash
python -m kpt_test prepare --dataset da20k --questions data/raw/da20k/questions.json --links data/raw/da20k/links.json --labels data/raw/da20k/knowledge.json --output-dir data/test
```

**XES3G5M**：从[官方仓库](https://github.com/ai4ed/XES3G5M)下载数据，输入其中的两个元数据文件。

```bash
python -m kpt_test prepare --dataset xes3g5m --questions data/raw/XES3G5M/metadata/questions.json --labels data/raw/XES3G5M/metadata/kc_routes_map.json --output-dir data/test
```

执行成功后，程序自动创建 `data/test` 文件夹并写入以下文件。**这些文件由程序生成，你只需保留它们。**

| 生成的文件 | 里面是什么 | 用来做什么 |
| --- | --- | --- |
| `test.inputs.jsonl` | 原始测试题目的 ID 和题干 | 交给你的模型预测知识点 |
| `test.noisy.inputs.jsonl` | 同一批题目，随机删字或交换相邻字符 | 再预测一次，检查模型能否容忍输入错误 |
| `test.gold.jsonl` | 每道测试题真正对应的知识点 ID | 评分程序用来核对预测，不作为模型输入 |
| `taxonomy.json` | 所有允许预测的知识点 ID、名称和层级 | 确认模型输出的标签 ID |
| `train.jsonl` / `valid.jsonl` | 带知识点标签的训练题 / 验证题 | 训练或调整模型时使用 |
| 其他文件 | 预测模板、题目划分和文件校验信息 | 保留原样，供程序使用 |

`.jsonl` 表示“一行一条 JSON 记录”。例如，原始题目文件中的一行是：

```json
{"qid":"1001","text":"解方程：3x+5=20。"}
```

对应的删字版本可能是下面这样，**题目 ID 保持一致**：

```json
{"qid":"1001","text":"解方程：3x+5=2。"}
```

这里的 `data/test` 就是存放上述文件的普通目录。默认按 8:1:1 划分训练、验证、测试题；已有自己的划分时，按[原始数据说明](docs/prepare_data.md)指定，确保与已有预测一致。重复评测直接复用该目录；重新生成时需选择一个空目录。

## 第三步：用你的模型预测，保存两个结果文件

| 让模型读取 | 将模型预测保存为（放在仓库根目录） |
| --- | --- |
| `data/test/test.inputs.jsonl` 中的原始题目 | `predictions.jsonl` |
| `data/test/test.noisy.inputs.jsonl` 中的删字、调换相邻字序题目 | `noisy_predictions.jsonl` |

两次使用同一模型和推理设置。**上一步生成的是题目文件，这一步需要你提供的是预测结果文件。**

两个预测文件都保存为 UTF-8 编码，每行一道题，格式如下（ID 仅作示意）：

```json
{"qid":"1001","labels":["101","102"]}
{"qid":"1002","labels":["103"]}
```

- `qid`：原题的 ID，所有测试题都要预测，每题只写一行。
- `labels`：模型预测的一个或多个最细一级知识点 ID，从 `taxonomy.json` 中选择，填写 ID 而非名称或概率。
- 两个文件的题目 ID 相同，各自填对应输入的实际预测。不得有空数组、重复标签或未知标签，行顺序不限。

已有预测结果时，按上述格式整理即可；题目 ID 和标签 ID 必须与第二步生成的数据一致。

## 第四步：生成语义评分需要的文件

这一步用于比较知识点含义是否接近。程序下载作业指定的语义模型，生成 `data/test/embeddings.json`，只需执行一次：

```bash
python -m pip install -e ".[semantic]"
python -m kpt_test build-embeddings --taxonomy data/test/taxonomy.json --output data/test/embeddings.json
```

默认模型为 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。已有对应的 `embeddings.json` 时跳过本步，后续评分可离线运行。

## 第五步：运行评分，查看结果

此时，两个预测文件在仓库根目录，其余生成的文件在 `data/test`。复制执行：

```bash
python -m kpt_test evaluate --data-dir data/test --predictions predictions.jsonl --noisy-predictions noisy_predictions.jsonl --output outputs/report.json --details outputs/details.jsonl --require-complete
```

运行成功后，终端显示主要结果，并自动保存两个文件：

| 结果文件 | 内容 |
| --- | --- |
| `outputs/report.json` | 全部指标和是否达到作业准确率要求 |
| `outputs/details.jsonl` | 每道题的预测标签、真实标签和得分 |

终端里 `evaluation_complete: true` 表示所需评测材料齐全；`accuracy_thresholds_passed: true` 表示严格准确率 ≥ 90%、宽松准确率 ≥ 95%。例如 `strict_accuracy: 0.92` 就是严格准确率 92%。

报告还包含 Precision、Recall、Micro/Macro-F1、层级和语义 P/R/F1、交互容错性。Macro-F1 对全部末级标签平均；交互容错性为删字、调换相邻字序后的 Precision / 原始 Precision，原始 Precision 为 0 时返回 `null`。

想先确认能否运行，可直接使用[已准备好题目和预测的合成示例](examples/README.md)。
