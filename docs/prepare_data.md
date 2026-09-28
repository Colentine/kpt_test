# 数据重建说明

日常评测直接按 [README](../README.md) 使用 `datasets/da20k` 或 `datasets/xes3g5m`。这些目录已包含真实题目、固定划分和语义评分向量。

下面用于核对数据来源或重新生成文件，所需源数据也已保存在仓库中。

## 1. 源文件位置

| 数据集 | 仓库中的实际文件 |
| --- | --- |
| DA 原始题目、完整知识树、题目标签关联表 | `datasets/sources/da20k/mathdata-main.zip` |
| DA 官方 427 标签表 | `datasets/sources/da20k/DA-20k-labels-with-father.json` |
| XES 原始题目元数据 | `datasets/sources/xes3g5m/questions.json` |
| XES 知识点 ID 与名称映射 | `datasets/sources/xes3g5m/kc_routes_map.json` |

DA 压缩包内真实文件名为 `math_questions_content.json`、`math_questions_knowledge.json`、`math_questions_knowledgetag.json`。重建程序直接读取 ZIP，无需手动解压或改名。

## 2. 重建文本数据

在仓库根目录执行，输出目录须不存在或为空：

```bash
python -m kpt_test.bundle --dataset da20k --output-dir outputs/rebuilt-da20k
python -m kpt_test.bundle --dataset xes3g5m --output-dir outputs/rebuilt-xes3g5m
```

重建程序执行文本清理、DA 标签范围筛选、XES 路径解析、固定 8:1:1 划分，以及删字调序题目生成；不会下载或读取图片。DA 的 MathML 公式转为 LaTeX 文本，分数、上下标、矩阵等结构保留。

随机种子为 42。每个输出目录包含 `provenance.json`，列出源文件摘要、保留/排除题数及图片引用移除情况。DA 被排除的题目 ID 和被过滤的标签均有记录。具体规则见[数据说明](../datasets/README.md)。

原样重建后，直接将 `datasets/对应数据集/embeddings.json` 复制到新目录，即可离线评测。如果修改了知识点体系或路径，则需重新生成语义向量。

## 3. 需要重新生成语义向量时

本仓库的向量由 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 生成，模型版本固定为 `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`。

```bash
python -m pip install -e ".[semantic]"
python -m kpt_test build-embeddings --taxonomy outputs/rebuilt-da20k/taxonomy.json --output outputs/rebuilt-da20k/embeddings.json --revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42
```

XES 将路径中的 `rebuilt-da20k` 改为 `rebuilt-xes3g5m`。这一步首次运行需要下载语义模型；使用仓库已有向量时不需要。

## 4. 验证

```bash
python -m unittest discover -s tests -v
```

测试包含真实数据的文件校验、划分隔离、纯文本检查、XES 865 标签解析、MathML 转换以及离线评分。测试中用真实标签作为预测的用例仅验证评分程序，不代表任何模型的实际成绩。
