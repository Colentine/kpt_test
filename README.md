# 知识点预测评测

仓库已包含 **DA-20K 和 XES3G5M 的题目、真实知识点标签、固定划分及评分向量**。克隆后直接使用，无需另外下载数据或语义模型。预测只使用文本，包含题干、选项和公式，不使用图片。

## 第一步：安装

需要 Python 3.10 或更新版本。在终端依次执行：

```bash
git clone https://github.com/Colentine/kpt_test.git
cd kpt_test
python -m pip install -e .
```

后续操作都在 `kpt_test` 仓库目录下进行。

## 第二步：选择数据集，读取题目

下面的文件**已经在仓库中**。选择一个数据集，直接读取对应文件：

| 用途 | DA-20K | XES3G5M |
| --- | --- | --- |
| 原始测试题目 | [test.inputs.jsonl](datasets/da20k/test.inputs.jsonl) | [test.inputs.jsonl](datasets/xes3g5m/test.inputs.jsonl) |
| 删字、调换字序后的同一批题目 | [test.noisy.inputs.jsonl](datasets/da20k/test.noisy.inputs.jsonl) | [test.noisy.inputs.jsonl](datasets/xes3g5m/test.noisy.inputs.jsonl) |
| 可预测的知识点 ID 和名称 | [taxonomy.json](datasets/da20k/taxonomy.json) | [taxonomy.json](datasets/xes3g5m/taxonomy.json) |
| 待填写的预测模板 | [predictions.template.jsonl](datasets/da20k/predictions.template.jsonl) | [predictions.template.jsonl](datasets/xes3g5m/predictions.template.jsonl) |

`.jsonl` 就是每行一条 JSON。例如题目文件中的一行：

```json
{"qid":"45289","text":"题干及必要的选项、公式文本"}
```

`qid` 是题目 ID，`text` 是交给模型的内容。两个题目文件的 ID 相同，其中 `noisy` 版本用来检查输入有小错误时，预测是否稳定。

需要训练模型时，同目录下的 `train.jsonl`、`valid.jsonl` 分别为训练集、验证集，含真实知识点标签。测试题的真实标签 `test.gold.jsonl` 只供评分使用，不传给预测模型。

## 第三步：预测并保存两个结果文件

用同一个模型和推理设置，分别对上一步的原始题目、删字调序后的题目预测。将结果放在仓库根目录：

| 预测哪份题目 | 保存的文件名 |
| --- | --- |
| `test.inputs.jsonl` | `predictions.jsonl` |
| `test.noisy.inputs.jsonl` | `noisy_predictions.jsonl` |

可复制所选数据集的预测模板，填入每题的预测知识点 ID。两个结果文件都保存为 UTF-8 编码，每行如下（标签 ID 仅作格式示意）：

```json
{"qid":"45289","labels":["3","4"]}
```

- 每题只写 `qid` 和 `labels`，必须包含全部测试题，每题一行，顺序不限。
- `labels` 填 `taxonomy.json` 中一个或多个知识点 **ID**，不填名称或概率。
- 不允许空预测、重复题目、重复标签或未知标签。
- 两个结果文件各自填写对应题目输入的实际预测。

## 第四步：复制命令评分

**评测 DA-20K：**

```bash
python -m kpt_test evaluate --data-dir datasets/da20k --predictions predictions.jsonl --noisy-predictions noisy_predictions.jsonl --output outputs/da20k.report.json --details outputs/da20k.details.jsonl --require-complete
```

**评测 XES3G5M：**

```bash
python -m kpt_test evaluate --data-dir datasets/xes3g5m --predictions predictions.jsonl --noisy-predictions noisy_predictions.jsonl --output outputs/xes3g5m.report.json --details outputs/xes3g5m.details.jsonl --require-complete
```

选择与预测文件对应的命令即可，程序自动读取仓库内的真实标签及评分向量，并创建 `outputs` 目录。

## 第五步：查看结果

终端直接显示分数：

- `strict_accuracy`：严格准确率，要求 ≥ 0.90。
- `loose_accuracy`：宽松准确率，要求 ≥ 0.95。
- `accuracy_thresholds_passed: true`：上述两项准确率都达标。
- `evaluation_complete: true`：全部评测所需材料齐全。

`outputs/数据集名.report.json` 保存完整报告，包含 Precision、Recall、Micro/Macro-F1、层级和语义 P/R/F1、交互容错性；`outputs/数据集名.details.jsonl` 保存逐题预测、真实标签及得分。

Macro-F1 对完整标签集合平均。交互容错性为扰动后 Precision / 原始 Precision，原始 Precision 为 0 时返回 `null`。

## 仓库中的数据

| 数据集 | 训练题 | 验证题 | 测试题 | 预测标签数 |
| --- | ---: | ---: | ---: | ---: |
| DA-20K 文本版 | 18,024 | 2,253 | 2,253 | 427 |
| XES3G5M 文本版 | 6,121 | 765 | 766 | 865 |

DA 来自提供的 `mathdata-main.zip`，按官方 427 标签范围保留 22,530 道题；XES 保留官方 7,652 道题。图片已移除，部分“如图”题目可能因此缺少信息。来源、筛选记录及纯文本限制见[数据说明](datasets/README.md)。

只想先试跑，可使用[合成示例](examples/README.md)。重新生成数据的步骤见[数据重建说明](docs/prepare_data.md)。
