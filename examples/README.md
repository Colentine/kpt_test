# 可运行的合成示例

这里的 6 道题、知识体系及预测是人工编写的流程示例，不属于真实 DA-20K / XES3G5M 数据。
`raw/` 展示 DA 分表格式及 XES 官方元数据结构；`da20k/`、`xes3g5m/` 中已准备好题目、真实知识点标签和语义评分文件。
两个目录表示同一组概念，但分别保留 DA 示例的 101–105 与 XES 示例的 0–4 标签 ID。

每个目录包含 1 道训练题、1 道验证题、4 道测试题。预测示例包含多标签、错标、漏标及扰动后的不同预测。
这些预测是手写演示数据，不能用作模型鲁棒性实验结论。

语义向量使用真实模型 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 生成，
固定模型 revision：`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`。评测只读缓存，不联网下载模型。

安装依赖后，在仓库根目录复制以下命令，即可离线试跑完整评测：

```bash
python -m kpt_test evaluate --data-dir examples/da20k --predictions examples/da20k.predictions.jsonl --noisy-predictions examples/da20k.noisy.predictions.jsonl --output outputs/da20k.report.json --require-complete
```

将命令中的 `da20k` 全部替换为 `xes3g5m`，即可运行另一个示例。预期结果：

| 指标 | 两个示例的结果 |
| --- | ---: |
| Strict Accuracy | 0.50 |
| Loose Accuracy | 0.75 |
| Precision / Recall / Micro-F1 | 0.60 |
| Macro-F1 | 0.466666667 |
| 层级 P/R/F1 | 0.68 |
| 语义 P/R/F1 | 约 0.911417082 |
| 原始 Precision | 0.60 |
| 扰动 Precision | 0.75 |
| 交互容错性 | 1.25 |

重新验证转换（输出到新目录）：

```bash
python -m kpt_test prepare --dataset da20k --questions examples/raw/da20k.questions.json --links examples/raw/da20k.links.json --labels examples/raw/da20k.knowledge.json --splits examples/raw/splits.json --output-dir outputs/prepared-da20k
python -m kpt_test prepare --dataset xes3g5m --questions examples/raw/xes3g5m.questions.json --labels examples/raw/xes3g5m.kc_routes_map.json --splits examples/raw/splits.json --output-dir outputs/prepared-xes3g5m
```
