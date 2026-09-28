# 准备测试包

已有完整测试包时，直接按 [README](../README.md) 评测。本说明用于第一次从原始数据生成测试包，生成后应使用包内测试题进行预测。

先完成 README 第一步的代码下载与安装，以下命令均在仓库根目录执行。

## 1. 转换数据（二选一）

默认按题目 ID 以 **8:1:1** 划分训练集、验证集和测试集，随机种子为 42。输出到 `data/test`，该目录须不存在或为空；对比不同模型时复用同一个包。

### DA-20K

下载：[原始数据](https://figshare.com/s/2be2eb2c06d00a9e4349) · [官方仓库](https://github.com/xuqiang124/atmk_system)

准备题目表、题目与标签关联表、知识点表，格式参考 [examples/raw](../examples/raw)，将命令中的路径替换为实际路径：

```bash
python -m kpt_test prepare --dataset da20k --questions data/raw/da20k/questions.json --links data/raw/da20k/links.json --labels data/raw/da20k/knowledge.json --output-dir data/test
```

若题目已包含 `labels` 或 `label_ids` 数组，可省略 `--links`。`--labels` 也支持官方 `DA-20k-labels-with-father.json`。

### XES3G5M

从[官方仓库的 Download 部分](https://github.com/ai4ed/XES3G5M)下载数据，使用题目元数据：

```bash
python -m kpt_test prepare --dataset xes3g5m --questions data/raw/XES3G5M/metadata/questions.json --labels data/raw/XES3G5M/metadata/kc_routes_map.json --output-dir data/test
```

已有自己的题目划分时，在转换命令末尾加 `--splits splits.json`，文件格式如下。三个集合必须不重叠并覆盖全部输入题目，测试集不能为空：

```json
{"train":["1","2"],"valid":["3"],"test":["4","5"]}
```

## 2. 生成语义缓存

首次生成需安装额外依赖并联网下载模型：

```bash
python -m pip install -e ".[semantic]"
python -m kpt_test build-embeddings --taxonomy data/test/taxonomy.json --output data/test/embeddings.json
```

默认使用作业指定的 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。缓存生成一次即可，后续评测直接读取 `embeddings.json`。

## 3. 对测试题生成预测

现在 `data/test` 已是完整测试包。用自己的模型分别读取：

- `test.inputs.jsonl`：原始题目，预测保存为 `predictions.jsonl`。
- `test.noisy.inputs.jsonl`：扰动题目，预测保存为 `noisy_predictions.jsonl`。

两次推理使用同一模型和配置，预测格式见 [README](../README.md)。`test.gold.jsonl` 仅用于评分，不作为模型输入。其他文件保留原样。

返回 README 第二步检查文件位置，再执行第三步的评测命令。

评估流程参考 [SGPE/scripts](https://github.com/Colentine/SGPE/tree/main/scripts)，指标按作业说明实现。转换参数可通过 `python -m kpt_test prepare --help` 查看。
