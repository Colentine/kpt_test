# 原始数据格式与题目划分

本说明补充 [README](../README.md) 第二步所需的原始文件格式。`prepare` 命令读取这些文件，生成待预测的题目、真实知识点标签和评分所需信息，统一保存在 `data/test`。

先完成 README 第一步的代码下载与安装，以下命令均在仓库根目录执行。

## 1. 转换数据（二选一）

默认按题目 ID 以 **8:1:1** 划分训练集、验证集和测试集，随机种子为 42。输出到 `data/test`，该目录须不存在或为空；对比不同模型时复用该目录。

### DA-20K

#### 文件从哪里来

| 数据 | 来源 | 说明 |
| --- | --- | --- |
| 完整题目及真实知识点标签 | 官方仓库链接的 [DA-20K 原始数据](https://figshare.com/s/2be2eb2c06d00a9e4349) | 未随本仓库提供；下载包内的具体文件名和字段尚未核验 |
| 知识点 ID、名称及父类 | 官方仓库的 [DA-20k-labels-with-father.json](https://github.com/xuqiang124/atmk_system/blob/master/DA-20k-labels-with-father.json) | 点击 Raw 后保存 JSON；这是已核验可读取的 427 个标签的元数据 |
| 可试跑的合成题目 | 本仓库 [examples/raw/da20k.questions.json](../examples/raw/da20k.questions.json) | 文件实际存在，仅用于演示，不是完整 DA-20K 数据 |

之前文档中的 `data/raw/da20k/questions.json`、`links.json`、`knowledge.json` 都是人为约定的示意路径，不代表官方下载包包含这些文件。`prepare` 只读取输入文件，不负责下载或创建它们。

#### 使用自己的真实题目文件

最简单的输入是一个 JSON 数组，每条记录同时包含题目 ID、题干和真实知识点 ID：

```json
[
  {"id":1,"content":"题干文本","labels":[3,4]},
  {"id":2,"content":"另一道题的题干文本","labels":[8]}
]
```

以上内容是字段示例，实际填写真实数据。知识点 ID 必须与标签表一致。若已有文件符合这个格式，直接传入它的路径；否则需先按原始文件字段整理。当前没有通用的原始格式转换脚本，仅修改文件名不能完成格式转换。

命令模板（将大写占位符替换为实际文件路径）：

```bash
python -m kpt_test prepare --dataset da20k --questions "YOUR_QUESTIONS_JSON" --labels "YOUR_LABELS_JSON" --output-dir data/test
```

`YOUR_QUESTIONS_JSON` 是上述题目 JSON；`YOUR_LABELS_JSON` 是已保存的官方 `DA-20k-labels-with-father.json`。

官方另有预处理版 `.h5` 和 `.pkl` 文件，它们不能直接传给这个 JSON 读取入口。来源见[官方数据说明](https://github.com/xuqiang124/atmk_system#dataset)。

#### 已有题目与标签分表时

也支持 SGPE 风格的三个 JSON 数组：

| 命令参数 | 所需内容 | 最小记录示例 |
| --- | --- | --- |
| `--questions` | 题目 ID 与题干 | `{"id":1,"text_processed":"判断两个集合的关系。"}` |
| `--links` | 题目 ID 与知识点 ID 的关联，一题可有多条 | `{"qid":1,"label_id":3}` |
| `--labels` | 知识点 ID、名称及父节点关系 | `{"id":3,"name":"子集","uuid":"u3","parent_uuid":"u1"}` |

每个 JSON 文件是上述记录的数组。知识点表需要包含父节点记录；根节点的 `parent_uuid` 为空字符串。

```bash
python -m kpt_test prepare --dataset da20k --questions "YOUR_QUESTIONS_JSON" --links "YOUR_LINKS_JSON" --labels "YOUR_KNOWLEDGE_JSON" --output-dir data/test
```

大写占位符分别替换为题目表、题目标签关联表和知识点表的真实路径。题目内嵌标签也接受 `label_ids` 字段。

#### 只想先验证程序能运行

下面的命令使用本仓库实际存在的合成数据，可以直接执行。输出到单独的演示目录，不作为真实数据实验：

```bash
python -m kpt_test prepare --dataset da20k --questions examples/raw/da20k.questions.json --links examples/raw/da20k.links.json --labels examples/raw/da20k.knowledge.json --splits examples/raw/splits.json --output-dir outputs/da20k-format-demo
```

### XES3G5M

从[官方仓库的 Download 部分](https://github.com/ai4ed/XES3G5M)下载数据，使用题目元数据：

- `metadata/questions.json`：题目 ID 到题目信息的映射，包含题干 `content` 和知识点路径 `kc_routes`。
- `metadata/kc_routes_map.json`：知识点 ID 到知识点路径的映射。

```bash
python -m kpt_test prepare --dataset xes3g5m --questions data/raw/XES3G5M/metadata/questions.json --labels data/raw/XES3G5M/metadata/kc_routes_map.json --output-dir data/test
```

已有自己的题目划分时，在转换命令末尾加 `--splits splits.json`，文件格式如下。三个集合必须不重叠并覆盖全部输入题目，测试集不能为空：

```json
{"train":["1","2"],"valid":["3"],"test":["4","5"]}
```

## 2. 生成语义评分文件

首次生成需安装额外依赖并联网下载模型：

```bash
python -m pip install -e ".[semantic]"
python -m kpt_test build-embeddings --taxonomy data/test/taxonomy.json --output data/test/embeddings.json
```

默认使用作业指定的 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。缓存生成一次即可，后续评测直接读取 `embeddings.json`。

## 3. 对测试题生成预测

现在 `data/test` 中已有待预测的题目和评分所需文件。用自己的模型分别读取：

- `test.inputs.jsonl`：原始题目，预测保存为 `predictions.jsonl`。
- `test.noisy.inputs.jsonl`：扰动题目，预测保存为 `noisy_predictions.jsonl`。

两次推理使用同一模型和配置，预测格式见 [README](../README.md)。`test.gold.jsonl` 仅用于评分，不作为模型输入。其他文件保留原样。

按 README 第三步保存预测结果，再执行第五步的评分命令。

评估流程参考 [SGPE/scripts](https://github.com/Colentine/SGPE/tree/main/scripts)，指标按作业说明实现。转换参数可通过 `python -m kpt_test prepare --help` 查看。
