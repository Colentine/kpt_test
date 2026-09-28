# 已准备好的纯文本数据

本目录随仓库分发，使用者无需另外下载数据或语义模型。`da20k/`、`xes3g5m/` 可直接传给 `evaluate --data-dir`。

## 数据来源与范围

| 数据 | 来源 | 仓库保留内容 |
| --- | --- | --- |
| DA 原始数据 | 提供的 `mathdata-main.zip`；ZIP 注释中的提交为 `e72248b3547ffda0bde23dca66660ba136d8fd06` | [原始 ZIP](sources/da20k/mathdata-main.zip)、转换后的文本数据 |
| DA 427 标签体系 | [atmk_system 官方标签表](https://github.com/xuqiang124/atmk_system/blob/master/DA-20k-labels-with-father.json) | [标签表原文件](sources/da20k/DA-20k-labels-with-father.json) |
| XES3G5M | [官方仓库](https://github.com/ai4ed/XES3G5M)提供的 [Google Drive 下载](https://drive.google.com/file/d/1eFiIYyh5O2V90RA0brammGH6EpHvPDQe/view) | `metadata/questions.json`、`metadata/kc_routes_map.json` 及转换后的文本数据 |
| 语义向量 | [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | 两个数据集各自的 `embeddings.json` |

XES 下载包约 373 MB，解压内容约 8 GB。知识点预测只需要题目和知识点元数据，因此仓库不收录作答序列、图片和原包内预计算的其他模型向量。下载归档的 SHA-256 为 `62d145bd995248f78726a0b6ab69612cf418e3118d0fe01bfe6f4c61b5072f73`。

保留 XES 官方仓库的 [MIT 许可与版权声明](licenses/XES3G5M-LICENSE)。DA 原始 ZIP 内只有说明文件和三个 JSON 表，没有附带 LICENSE；这里记录原始来源，未将第三方数据重新声明为本项目所有。

## DA-20K 的处理规则

提供的 ZIP 含 22,641 道题、648 个知识树节点和 42,701 条题目标签关联。本作业采用官方的 **427 个目标标签及其父类关系**作为固定预测词表：

- 17 道题没有任何标签，排除。
- 94 道题的标签全部不在这 427 个目标标签中，排除。
- 116 道保留题含有词表外标签，仅保留属于目标词表的标签。
- 最终保留 **22,530 道题**。这一数目来自本次提供的原始包及明确的筛选规则，与论文中的 22,498 题版本不完全一致。

题干的 HTML 转为文本，MathML 转为 LaTeX；移除图片和“查看解析”等页面页脚。只使用题干及选项，不将答案或解析放入模型输入。

全部排除题目及过滤标签列于 [da20k/provenance.json](da20k/provenance.json)，源文件校验值也记录在其中。这里使用官方两层父类表评估层级接近度，未混用原始 648 节点树的其他层级。

## XES3G5M 的处理规则

官方元数据包含 **7,652 道题、1,175 个知识树节点**。每道题的 `kc_routes` 给出它关联的路径，取每条路径末端的 ID 作为预测标签，共 **865 个**。

一个节点可能在某条路径末端出现，也在另一条路径中作为祖先出现；按原数据的路径末端定义保留标签，不按整张图“无子节点”筛选。339 个预测标签具有多条已出现的路径，全部保留。层级配对分数取路径组合的最高分，再进行题目内的一对一最优匹配。

原始映射中有三组名称只相差首尾空格，但 ID 不同。处理时保留原始 ID 和名称，不把这些类别合并。语义编码包含名称及所有路径，以 ` | ` 分隔路径。

题干与选项合并为纯文本，移除图片占位符；答案、解析及作答序列不进入输入。保留全部 7,652 道题，统计见 [xes3g5m/provenance.json](xes3g5m/provenance.json)。

## 纯文本版本的限制

按本次要求，不使用图片。DA 的 3,860 道保留题、XES 的 2,221 道题原本包含图片引用。这些题的文本仍保留，“如图”等表述可能因此缺少上下文。

去图后的文本可能相同：DA 有 1 组重复文本，XES 有 544 组（比唯一文本数多 678 条记录）；其中分别有 1 组、79 组对应不同的真实标签集合。这些记录仍按原始题目 ID 保留，不应把纯文本模型的成绩等同于使用完整图文的成绩。

## 固定划分与评分资源

题目按 `SHA256(42:qid)` 排序，以 8:1:1 划分；同一个题目 ID 不跨集合。保留不同 ID 的相同文本，因此该划分不保证文本去重。`splits.json` 记录每个集合的 ID，`manifest.json` 记录文件 SHA-256。比较模型时应使用相同文件版本。

| 数据集 | 训练 | 验证 | 测试 |
| --- | ---: | ---: | ---: |
| DA-20K 文本版 | 18,024 | 2,253 | 2,253 |
| XES3G5M 文本版 | 6,121 | 765 | 766 |

`embeddings.json` 的模型 revision 固定为 `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`。向量依据知识点名称和路径生成，维度为 384；源数据未提供定义时不补造定义。基础 NumPy/SciPy 依赖即可读取这些向量离线评分。

原始数据再生成方式见[重建说明](../docs/prepare_data.md)。
