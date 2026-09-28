# 知识点预测评测

支持 **DA-20K 和 XES3G5M**。按下面的格式准备预测结果，再依次完成四步，即可得到完整评测报告。

## 先准备好这两个预测文件

| 文件名 | 内容 |
| --- | --- |
| `predictions.jsonl` | 对 `test.inputs.jsonl` 中原始题目的预测 |
| `noisy_predictions.jsonl` | 同一模型对 `test.noisy.inputs.jsonl` 中扰动题目的预测 |

两个文件都使用 **UTF-8 JSONL** 格式，每行一道题，只写题目 ID 和预测标签 ID：

```json
{"qid":"1001","labels":["101","102"]}
{"qid":"1002","labels":["103"]}
```

填写规则：

- `qid` 与测试输入中的题目 ID 一致，所有测试题都要有预测，每题只写一行。
- `labels` 填一个或多个末级知识点 **ID**，可用 ID 见测试包的 `taxonomy.json`。
- 不填知识点名称或概率；不允许空数组、重复标签和未知标签。行顺序不限。
- 扰动预测需要对扰动后的题目重新推理，不能直接复制原始预测。

**还需要与预测对应的完整测试包**，其中包含真实标签、标签体系和语义缓存，程序用它们计算分数。没有测试包时，先按[准备测试包](docs/prepare_data.md)生成，再对包内题目预测。已有预测应使用生成预测时的同一测试划分。

## 第一步：下载代码并安装

需要 Python 3.10 或更新版本。打开终端，依次执行：

```bash
git clone https://github.com/Colentine/kpt_test.git
cd kpt_test
python -m pip install -e .
```

**后续命令都在 `kpt_test` 仓库根目录执行。**

## 第二步：把文件放到指定位置

在仓库根目录新建 `data` 文件夹，将完整测试包文件夹复制进去并命名为 `test`；将两个预测文件放到仓库根目录。

摆放后应当是这样：

```text
kpt_test/
├── predictions.jsonl
├── noisy_predictions.jsonl
└── data/
    └── test/
        ├── test.inputs.jsonl
        ├── test.noisy.inputs.jsonl
        ├── test.gold.jsonl
        ├── taxonomy.json
        ├── embeddings.json
        ├── splits.json
        └── manifest.json
```

测试包里的其他文件也保留。**DA-20K、XES3G5M 都使用上述摆放方式，每次评测一个数据集。** 程序自动识别包内的数据集信息，不需要改代码或修改包内文件。

## 第三步：复制命令，开始评测

```bash
python -m kpt_test evaluate --data-dir data/test --predictions predictions.jsonl --noisy-predictions noisy_predictions.jsonl --output outputs/report.json --details outputs/details.jsonl --require-complete
```

命令会检查预测文件，计算全部指标，并自动创建 `outputs` 文件夹。已有测试包和语义缓存时，评测无需联网或加载预测模型。

## 第四步：查看结果

运行成功后，终端直接显示结果，重点看：

| 终端字段 | 怎么看 |
| --- | --- |
| `evaluation_complete` | 为 `true` 表示评测所需材料齐全，全部指标已处理 |
| `strict_accuracy` | 严格准确率，如 `0.92` 表示 92%，要求 ≥ 90% |
| `loose_accuracy` | 宽松准确率，要求 ≥ 95% |
| `accuracy_thresholds_passed` | 为 `true` 表示上述两项准确率都达标 |

详细结果保存在：

- **`outputs/report.json`**：完整报告，包含准确率、Precision、Recall、Micro/Macro-F1、层级和语义 P/R/F1、交互容错性，以及逐类统计。
- **`outputs/details.jsonl`**：逐题结果，可查看每题预测、真实标签和得分。

Macro-F1 对全部末级标签取平均，未出现类别按 0 计。交互容错性为扰动后 Precision / 原始 Precision；原始 Precision 为 0 时，该比值显示为 `null`。

## 报错时检查这里

| 提示 | 处理方式 |
| --- | --- |
| `No such file` / 找不到文件 | 对照第二步检查位置，确认终端当前在仓库根目录 |
| `qid mismatch` | 预测题目与测试集不一致，按测试输入补齐缺失题目、去掉额外题目 |
| `unknown/non-leaf label IDs` | 改用 `taxonomy.json` 中的末级标签 ID |
| `at least one label` / `duplicate` | 检查空预测、重复题目或重复标签 |
| `Semantic cache required` | 缺少 `embeddings.json`，按[准备测试包](docs/prepare_data.md)生成 |
| `hash mismatch` | 测试包文件被修改或混用了版本，恢复与预测对应的完整测试包 |

想先试跑可使用[仓库自带示例](examples/README.md)。数据转换和语义缓存生成见[准备测试包](docs/prepare_data.md)。
