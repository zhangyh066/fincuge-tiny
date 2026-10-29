# 数据说明

`data/fincuge/` 下的 JSON 由 FinCUGE 原始数据转换而来，已转成 alpaca 三字段格式（instruction / input / output），clone 后可直接训练。

## 来源

- 原始任务与数据：BBT-Fin / FinCUGE（arXiv 2302.09432）
- 指令化版本：FinCUGE-Instruction（HuggingFace `Maciel/FinCUGE-Instruction`）
- 新闻任务用的 7 类版本由 `prepare_newscat7.py` 从 15 类合并得到

## 文件

| 文件 | 条数 | 用途 |
|---|---|---|
| `emotion_train.json` / `emotion_eval.json` | 16,157 / 2,020 | 情绪三分类，答案保留原始 15 种句式 |
| `emotion_vb_train.json` / `emotion_vb_eval.json` | 16,157 / 2,020 | 同上，但答案只剩标签词（Verbalizer 对照实验） |
| `newscat7_train.json` / `newscat7_eval.json` | 7,071 / 884 | 新闻七分类 |
| `entity_train.json` / `entity_eval.json` | 11,752 / 1,464 | 事件主体抽取 |
| `newscat_train.json` / `newscat_eval.json` | 7,071 / 884 | 新闻 15 类原始版本，未使用，保留备查 |

## 版权

数据版权归 BBT-Fin / FinCUGE 原项目所有，放在本仓库仅为便于研究复现，请遵循原项目许可，勿作商用；如原作者认为不妥，提出后会立即移除。
