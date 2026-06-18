---
language:
- vi
- en
license: cc-by-sa-4.0
task_categories:
- table-question-answering
- text-generation
tags:
- text-to-sql
- vietnamese
- spider
- nl2sql
pretty_name: ViSpider
size_categories:
- 10K<n<100K
---

# ViSpider — Vietnamese Spider Benchmark

ViSpider is a Vietnamese translation of the [Spider](https://yale-lily.github.io/spider) Text-to-SQL benchmark (Yu et al., EMNLP 2018).

## Splits

| Split | Items |
|-------|------:|
| Train | 8,659 |
| Dev   | 1,034 |
| Test  | 2,147 |
| **Total** | **11,840** |

## Schema

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (`spider-{split}-XXXXX`) |
| `db_id` | string | Database identifier |
| `question` | string | Original English question |
| `query` | string | Gold SQL query |
| `sql_class` | string | SQL complexity class (`JOIN`, `AGG_ONLY`, `NESTED`, …) |
| `question_vi` | string | Vietnamese translation |
| `translation_method` | string | `human` · `gpt` · `oss` |

## Translation breakdown (train)

| Method | Items | Source |
|--------|------:|--------|
| `human` | 1,299 | Human annotators |
| `gpt`   | 2,165 | GPT few-shot translation |
| `oss`   | 5,195 | Fine-tuned Qwen2.5-7B translator |

## Related

- Translator model: [hoadm/qwen25-spider-translator-vi](https://huggingface.co/hoadm/qwen25-spider-translator-vi)
- Sister dataset: [hoadm/vibird](https://huggingface.co/datasets/hoadm/vibird)
- Source code: [hoadm-net/hitl-dataset-translation](https://github.com/hoadm-net/hitl-dataset-translation)

## Citation

```bibtex
@inproceedings{yu2018spider,
  title     = {Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author    = {Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and Zhang, Zilin and Radev, Dragomir},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  year      = {2018},
  url       = {https://aclanthology.org/D18-1425}
}
```

## License

[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
