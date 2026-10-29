# 候选语料来源

`data/candidates/` 和 `data/alpaca_zh_51k.json` 是早期做数据筛选时下载的第三方语料，**三个正式任务都不用它们**（正式训练只用 `data/fincuge/*.json`）。因为体积大、且各自有独立许可，所以没有放进仓库。需要复现筛选过程时按下面的地址自己下。

| 本地文件 | 是什么 | 规模 | 去哪里找 |
|---|---|---|---|
| `alpaca_zh_51k.json` | 中文 Alpaca 51k 通用指令集 | 17.8 MB / 51k 条 | https://github.com/ymcui/Chinese-LLaMA-Alpaca 的 `data/alpaca_data_zh_51k.json` |
| `coig_pc.jsonl` | COIG-PC 抽样，含 task_type / domain 细粒度标签 | 4.4 MB | https://huggingface.co/datasets/BAAI/COIG-PC |
| `finance_coig_pc.jsonl` | 上面那份按金融域筛出来的子集 | 1.6 MB | 同上，按 `domain` 过滤 |
| `mba_baike.jsonl` | 经管 / 百科类问答（字段与 COIG-PC 一致，推测是它的经管域子集） | 65.9 MB | 同上；同类替代见 https://huggingface.co/datasets/m-a-p/COIG-CQIA |
| `cail_valid.parquet` | CAIL2018 法研杯判决预测（罪名、法条、刑期） | 10.3 MB / 17,131 条 | https://github.com/china-ai-law-challenge/CAIL2018 |
| `fincuge_train.parquet` | FinCUGE 全任务指令版，含 task / desc 字段 | 27.3 MB / 123,137 条 | https://huggingface.co/datasets/Maciel/FinCUGE-Instruction （原始任务见 https://github.com/ssymmetry/BBT-FinCUGE-Applications） |
| `fincuge_eval.parquet` | 同上，评估切分 | 3.3 MB / 15,167 条 | 同上 |
| `lima_train.jsonl` | **不是数据**：LIMA 是受限数据集，需要先在页面上同意条款，本地这个文件只是"Access to dataset GAIR/lima is restricted"的提示（119 字节） | — | https://huggingface.co/datasets/GAIR/lima |

## 许可提醒

- COIG-PC：默认 Apache-2.0，但每个子数据集可能有自己的许可，以子数据集声明为准
- CAIL2018：学术研究用途
- LIMA：受限访问，非商用
- FinCUGE / FinCUGE-Instruction：版权归 BBT-Fin 项目，研究用途

拿这些数据自己再分发前，请先看各自页面的许可条款。
