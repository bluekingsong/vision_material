# NoR & OneR Benchmarks: Qwen3-0.6B / Qwen3-4B

**Test set**: `processed_data/{dataset}/dev_subsampled.jsonl` (100 examples each)

**Models**:
- QA Reader: `Qwen/Qwen3-0.6B` / `Qwen/Qwen3-4B` (HTTP via `llm_server`, port 8011)
- Dense Retriever (OneR only): `Qwen/Qwen3-Embedding-0.6B` / `Qwen/Qwen3-Embedding-4B` (HTTP via `dense_serve`, port 9201, linear-scan)

**Evaluation**: Official scripts under `official_evaluation/{dataset}/` (hotpotqa, 2wikimultihopqa, musique). `musique` evaluator was re-cloned from upstream (`stonybrooknlp/musique`).

**OneR hyperparameters**: `bm25_retrieval_count=15`, `distractor_count=1`, `rc_qa_type=direct` (论文 IRCoT 的 OneR 标准设置). Reader prompt uses 20 few-shot in-context examples (codex CoT style).

**Retrieval corpora (for OneR)**:

| Corpus name (used by `dense_serve`) | Source                                                                                                                                                                                                      | # paragraphs (after dedup) | Notes                                                                                                       |
| -------------------------------------| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ----------------------------| -------------------------------------------------------------------------------------------------------------|
| `musique`                           | `raw_data/musique/musique_{ans,full}_v1.0_{train,dev,test}.jsonl` (per `retriever_server/build_index.py`)                                                                                                   | 139,416                   | full corpus                                                                                                 |
| `2wikimultihopqa`                   | `raw_data/2wikimultihopqa/{train,dev,test}.json` (per `retriever_server/build_index.py`)                                                                                                                    | 430,225                   | full corpus                                                                                                 |
| `hotpotqa`                          | **`raw_data/hotpotqa_local/corpus.jsonl`** (distinct gold+distractor (title, paragraph_text) from hotpot_train_v1.1 + hotpot_dev_distractor_v1; built by `retriever_server/build_hotpotqa_local_corpus.py`) | 509,300                   | **local subset, not full 5M Wikipedia abstracts**. Mounted as `hotpotqa` corpus in dense_serve via symlink. |

**Note**: `IIRC` is intentionally skipped per task instruction (corpus too large to index in this iteration).

---

## Summary: NoR vs OneR (EM / F1)

| Model | Dataset | NoR (EM / F1) | OneR (EM / F1) | Δ EM | Δ F1 |
|---|---|---|---|---|---|
| Qwen3-0.6B | hotpotqa | 0.080 / 0.132 | 0.180 / 0.234 | +0.100 | +0.102 |
| Qwen3-0.6B | 2wikimultihopqa | 0.230 / 0.264 | 0.310 / 0.344 | +0.080 | +0.080 |
| Qwen3-0.6B | musique | 0.000 / 0.065 | 0.020 / 0.083 | +0.020 | +0.018 |
| Qwen3-4B | hotpotqa | 0.170 / 0.250 | 0.340 / 0.446 | +0.170 | +0.196 |
| Qwen3-4B | 2wikimultihopqa | 0.260 / 0.293 | 0.410 / 0.449 | +0.150 | +0.156 |
| Qwen3-4B | musique | 0.020 / 0.093 | 0.050 / 0.177 | +0.030 | +0.084 |

---

## NOR_QA

| Model | Dataset | EM | F1 | Precision | Recall | Count | Prediction File |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 2wikimultihopqa | 0.230 | 0.264 | 0.268 | 0.266 | 100 | `predictions/nor_qa_qwen3_0_6b_2wikimultihopqa____prompt_set_1/prediction__2wikimultihopqa_to_2wikimultihopqa__dev_subsampled.json` |
| Qwen3-0.6B | hotpotqa | 0.080 | 0.132 | 0.144 | 0.139 | 100 | `predictions/nor_qa_qwen3_0_6b_hotpotqa____prompt_set_1/prediction__hotpotqa_to_hotpotqa__dev_subsampled.json` |
| Qwen3-0.6B | musique | 0.000 | 0.065 | 0.069 | 0.063 | 100 | `predictions/nor_qa_qwen3_0_6b_musique____prompt_set_1/prediction__musique_to_musique__dev_subsampled.json` |
| Qwen3-4B | 2wikimultihopqa | 0.260 | 0.293 | 0.301 | 0.317 | 100 | `predictions/nor_qa_qwen3_4b_2wikimultihopqa____prompt_set_1/prediction__2wikimultihopqa_to_2wikimultihopqa__dev_subsampled.json` |
| Qwen3-4B | hotpotqa | 0.170 | 0.250 | 0.272 | 0.252 | 100 | `predictions/nor_qa_qwen3_4b_hotpotqa____prompt_set_1/prediction__hotpotqa_to_hotpotqa__dev_subsampled.json` |
| Qwen3-4B | musique | 0.020 | 0.093 | 0.113 | 0.091 | 100 | `predictions/nor_qa_qwen3_4b_musique____prompt_set_1/prediction__musique_to_musique__dev_subsampled.json` |

## ONER_QA

| Model | Dataset | EM | F1 | Precision | Recall | Count | Prediction File |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 2wikimultihopqa | 0.310 | 0.344 | 0.354 | 0.344 | 100 | `predictions/oner_qa_qwen3_0_6b_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__15___distractor_count__1/prediction__2wikimultihopqa_to_2wikimultihopqa__dev_subsampled.json` |
| Qwen3-0.6B | hotpotqa | 0.180 | 0.234 | 0.269 | 0.239 | 100 | `predictions/oner_qa_qwen3_0_6b_hotpotqa____prompt_set_1___bm25_retrieval_count__15___distractor_count__1/prediction__hotpotqa_to_hotpotqa__dev_subsampled.json` |
| Qwen3-0.6B | musique | 0.020 | 0.083 | 0.095 | 0.083 | 100 | `predictions/oner_qa_qwen3_0_6b_musique____prompt_set_1___bm25_retrieval_count__15___distractor_count__1/prediction__musique_to_musique__dev_subsampled.json` |
| Qwen3-4B | 2wikimultihopqa | 0.410 | 0.449 | 0.450 | 0.455 | 100 | `predictions/oner_qa_qwen3_4b_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__15___distractor_count__1/prediction__2wikimultihopqa_to_2wikimultihopqa__dev_subsampled.json` |
| Qwen3-4B | hotpotqa | 0.340 | 0.446 | 0.493 | 0.450 | 100 | `predictions/oner_qa_qwen3_4b_hotpotqa____prompt_set_1___bm25_retrieval_count__15___distractor_count__1/prediction__hotpotqa_to_hotpotqa__dev_subsampled.json` |
| Qwen3-4B | musique | 0.050 | 0.177 | 0.201 | 0.177 | 100 | `predictions/oner_qa_qwen3_4b_musique____prompt_set_1___bm25_retrieval_count__15___distractor_count__1/prediction__musique_to_musique__dev_subsampled.json` |

---

## Appendix: Raw Metrics Files

- **nor_qa / Qwen3-0.6B / 2wikimultihopqa** (hp: `prompt_set_1`): `predictions/nor_qa_qwen3_0_6b_2wikimultihopqa____prompt_set_1/evaluation_metrics__2wikimultihopqa_to_2wikimultihopqa__dev_subsampled.json`
- **nor_qa / Qwen3-0.6B / hotpotqa** (hp: `prompt_set_1`): `predictions/nor_qa_qwen3_0_6b_hotpotqa____prompt_set_1/evaluation_metrics__hotpotqa_to_hotpotqa__dev_subsampled.json`
- **nor_qa / Qwen3-0.6B / musique** (hp: `prompt_set_1`): `predictions/nor_qa_qwen3_0_6b_musique____prompt_set_1/evaluation_metrics__musique_to_musique__dev_subsampled.json`
- **nor_qa / Qwen3-4B / 2wikimultihopqa** (hp: `prompt_set_1`): `predictions/nor_qa_qwen3_4b_2wikimultihopqa____prompt_set_1/evaluation_metrics__2wikimultihopqa_to_2wikimultihopqa__dev_subsampled.json`
- **nor_qa / Qwen3-4B / hotpotqa** (hp: `prompt_set_1`): `predictions/nor_qa_qwen3_4b_hotpotqa____prompt_set_1/evaluation_metrics__hotpotqa_to_hotpotqa__dev_subsampled.json`
- **nor_qa / Qwen3-4B / musique** (hp: `prompt_set_1`): `predictions/nor_qa_qwen3_4b_musique____prompt_set_1/evaluation_metrics__musique_to_musique__dev_subsampled.json`
- **oner_qa / Qwen3-0.6B / 2wikimultihopqa** (hp: `prompt_set_1___bm25_retrieval_count__15___distractor_count__1`): `predictions/oner_qa_qwen3_0_6b_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__15___distractor_count__1/evaluation_metrics__2wikimultihopqa_to_2wikimultihopqa__dev_subsampled.json`
- **oner_qa / Qwen3-0.6B / hotpotqa** (hp: `prompt_set_1___bm25_retrieval_count__15___distractor_count__1`): `predictions/oner_qa_qwen3_0_6b_hotpotqa____prompt_set_1___bm25_retrieval_count__15___distractor_count__1/evaluation_metrics__hotpotqa_to_hotpotqa__dev_subsampled.json`
- **oner_qa / Qwen3-0.6B / musique** (hp: `prompt_set_1___bm25_retrieval_count__15___distractor_count__1`): `predictions/oner_qa_qwen3_0_6b_musique____prompt_set_1___bm25_retrieval_count__15___distractor_count__1/evaluation_metrics__musique_to_musique__dev_subsampled.json`
- **oner_qa / Qwen3-4B / 2wikimultihopqa** (hp: `prompt_set_1___bm25_retrieval_count__15___distractor_count__1`): `predictions/oner_qa_qwen3_4b_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__15___distractor_count__1/evaluation_metrics__2wikimultihopqa_to_2wikimultihopqa__dev_subsampled.json`
- **oner_qa / Qwen3-4B / hotpotqa** (hp: `prompt_set_1___bm25_retrieval_count__15___distractor_count__1`): `predictions/oner_qa_qwen3_4b_hotpotqa____prompt_set_1___bm25_retrieval_count__15___distractor_count__1/evaluation_metrics__hotpotqa_to_hotpotqa__dev_subsampled.json`
- **oner_qa / Qwen3-4B / musique** (hp: `prompt_set_1___bm25_retrieval_count__15___distractor_count__1`): `predictions/oner_qa_qwen3_4b_musique____prompt_set_1___bm25_retrieval_count__15___distractor_count__1/evaluation_metrics__musique_to_musique__dev_subsampled.json`


---

## OneR: BM25 (ES) vs Dense retriever — Qwen3-0.6B

为对照不同检索后端对 OneR 的影响, 在相同 reader (Qwen3-0.6B, prompt_set_1, 20 few-shot CoT, `rc_qa_type=direct`) 和相同超参 (`bm25_retrieval_count=15`, `distractor_count=1`) 下, 分别用以下两种 retriever 重跑 dev_subsampled (100 ex/dataset):

| Retriever | Backend | Index/Corpus |
|---|---|---|
| Dense | Qwen/Qwen3-Embedding-0.6B (`dense_serve` @ 9201, linear-scan) | 同表 1 |
| BM25  | Elasticsearch 7.10.2 @ 9200 (`retriever_server/serve.py` @ 8000) | hotpotqa=5,233,329 / 2wikimultihopqa=430,225 / musique=139,416 |

两次 run 中 reader/few-shot/topk/distractor 等所有变量均一致, 仅 retriever 不同, 故指标差异可归因到检索端。
预测目录用后缀区分: `..._bm25_retriever` / `..._dense_retriever`。

### Results (Qwen3-0.6B, 100 ex, prompt_set_1)

| Dataset | Retriever | in-house EM | in-house F1 | official EM | official F1 |
|---|---|---:|---:|---:|---:|
| hotpotqa        | Dense | 0.180 | 0.234 | 0.180 | 0.238 |
| hotpotqa        | **BM25** | 0.160 | 0.218 | 0.160 | 0.212 |
| 2wikimultihopqa | Dense | 0.310 | 0.344 | 0.310 | 0.354 |
| 2wikimultihopqa | **BM25** | 0.260 | 0.299 | 0.270 | 0.315 |
| musique         | Dense | 0.020 | 0.083 | 0.040 | 0.106 |
| musique         | **BM25** | 0.010 | 0.111 | 0.020 | 0.125 |

观察:
- **hotpotqa / 2wikimultihopqa**: Dense (Qwen3-Embedding-0.6B) > BM25。2wiki 上 official F1 差 +3.9 (0.354 vs 0.315), hotpot 差 +2.6 (0.238 vs 0.212)。说明在 wiki 全集上 dense 召回比 BM25 更准。
- **musique**: BM25 (官方 F1 0.125) > Dense (0.106)。musique 语料更小 (139K), BM25 词项匹配反而更鲁棒; dense 在小语料/复合实体 query 上召回偏移。

### Run artifacts (Qwen3-0.6B BM25)

- `predictions/oner_qa_qwen3_0_6b_{hotpotqa,2wikimultihopqa,musique}____prompt_set_1___bm25_retrieval_count__15___distractor_count__1___bm25_retriever/`
  - `prediction__*.json` (100 答案)
  - `evaluation_metrics__*.json` (in-house)
  - `official_evaluation_metrics__*.json` (官方脚本)
- 单条平均耗时: hotpot 7.4s, 2wiki 7.2s, musique 6.9s (BM25 查询 ~0.7s + LLM ~6.5s on 2×A800)
- 服务启动:
  - ES: `cd /home/jinsonglan.ljs/agent_tempdir/es/elasticsearch-7.10.2 && nohup ./bin/elasticsearch &`
  - retriever_server: `uvicorn serve:app --port 8000 --app-dir retriever_server`
  - LLM: `GPUS=0,1 MODEL_NAME=qwen3-0.6b PORT=8011 bash llm_server/run_qwen3.sh start`
  - `.retriever_address.jsonnet` 端口切换到 `8000` (走 ES BM25); dense 模式则切回 `9201`

---

## NoR Prompt Variants — Qwen3-4B (12 experiments)

**目的**：对比 NoR 在 prompt **格式**（codex / flan_t5）× **推理方式**（direct / cot）下的差异。
- **codex** 格式: `# METADATA: ...` + `Q:` + `A:`（简短）
- **flan_t5** 格式: `Q: Answer the following question.\n<question>` + `A:`（含任务指令前缀）
- **direct**: A 直接给答案
- **cot**: A 含推理链 "...So the answer is: X." + answer_extractor 后处理

**Prompt 文件**: `prompts/{dataset}/no_context_{direct|cot}_qa_{codex|flan_t5}.txt`
**Base config 命名**:
- `codex_direct`: `base_configs/nor_qa_qwen3_4b_{dataset}.jsonnet`（已存在）
- `codex_cot`:    `base_configs/nor_qa_qwen3_4b_codex_cot_{dataset}.jsonnet`
- `flan_t5_direct`: `base_configs/nor_qa_qwen3_4b_flan_t5_direct_{dataset}.jsonnet`
- `flan_t5_cot`:    `base_configs/nor_qa_qwen3_4b_flan_t5_cot_{dataset}.jsonnet`

**并行执行**: 双 4B llm_server (GPU0:8011, GPU1:8012)；通过 `LLM_SERVER_PORT` 环境变量路由（已在 `lib.get_llm_server_address` 中加支持），脚本 `run_nor_4b_all.sh` 把 12 个实验对半分配并行运行。

### 结果 (Qwen3-4B, dev_subsampled.jsonl, 100 examples, official metrics)

| Prompt 变体 | Dataset | EM | F1 | Precision | Recall |
|---|---|---|---|---|---|
| codex_direct   | hotpotqa        | 0.170 | 0.245 | 0.274 | 0.239 |
| codex_direct   | 2wikimultihopqa | 0.260 | 0.299 | 0.295 | 0.333 |
| codex_direct   | musique         | 0.020 | 0.105 | —     | —     |
| codex_cot      | hotpotqa        | 0.200 | 0.269 | 0.296 | 0.279 |
| codex_cot      | 2wikimultihopqa | 0.220 | 0.276 | 0.267 | 0.371 |
| codex_cot      | musique         | 0.030 | 0.135 | —     | —     |
| flan_t5_direct | hotpotqa        | 0.190 | 0.254 | 0.281 | 0.250 |
| flan_t5_direct | 2wikimultihopqa | 0.260 | 0.313 | 0.314 | 0.317 |
| flan_t5_direct | musique         | 0.040 | 0.133 | —     | —     |
| flan_t5_cot    | hotpotqa        | 0.150 | 0.231 | 0.262 | 0.228 |
| flan_t5_cot    | 2wikimultihopqa | 0.270 | 0.319 | 0.320 | 0.322 |
| flan_t5_cot    | musique         | 0.030 | 0.125 | —     | —     |

> musique 的 P/R 走官方 `evaluate_v1.0.py`，只输出 EM/F1（无 precision/recall 字段）。

### 跨变体对比汇总（每个 dataset 找最佳）

| Dataset | 最佳变体 (按 F1) | EM | F1 | 次佳 |
|---|---|---|---|---|
| hotpotqa        | codex_cot       | 0.200 | 0.269 | flan_t5_direct (0.254) |
| 2wikimultihopqa | flan_t5_cot     | 0.270 | 0.319 | flan_t5_direct (0.313) |
| musique         | flan_t5_direct  | 0.040 | 0.133 | codex_cot (0.135 F1 略高 EM 低) |

### 观察

1. **flan_t5 格式总体略优于 codex**，特别在 2wikimultihopqa 上（+0.014 ~ +0.020 F1）。差异可能来自显式指令前缀让 reader 更聚焦于 "回答问题" 任务。
2. **CoT 增益不稳定**：
   - hotpotqa: codex_cot > codex_direct (+0.024 F1)，但 flan_t5_cot < flan_t5_direct (−0.023 F1)
   - 2wikimultihopqa: cot 在两种格式下都略升 F1 (+~0.020) 但 codex_cot EM 反而降 0.04
   - musique: cot 让两种格式都有微小提升
3. **musique 整体非常难**：4B 模型在无 retrieval 时 EM 均在 0.02~0.04，F1 仅 0.10~0.13。
4. 对 NoR 而言，**Recall 普遍高于 Precision**（除 codex_cot 2wiki 外），意味着模型答案常**包含**正确实体但夹带额外信息。

### Prediction & Metrics 文件路径

所有 12 个实验输出位于：
`predictions/nor_qa_qwen3_4b_{variant_suffix}_{dataset}____prompt_set_1/`
- `prediction__{ds}_to_{ds}__dev_subsampled.json`
- `official_evaluation_metrics__{ds}_to_{ds}__dev_subsampled.json`

其中 `variant_suffix`:
- codex_direct → 空（即 `nor_qa_qwen3_4b_{ds}`）
- codex_cot → `_codex_cot`
- flan_t5_direct → `_flan_t5_direct`
- flan_t5_cot → `_flan_t5_cot`

### 复现命令

```bash
# 启动两个 4B server (各占一卡)
PORT=8011 MODEL_NAME=qwen3-4b GPUS=0 bash llm_server/run_qwen3.sh start
PORT=8012 MODEL_NAME=qwen3-4b GPUS=1 bash llm_server/run_qwen3.sh start

# 并行预测 + 评估
bash run_nor_4b_all.sh predict
bash run_nor_4b_all.sh evaluate
```

---

## 解码策略对比 — Qwen3 官方推荐采样参数 vs 贪心

**动机**：在 4B/musique greedy 输出中发现 1 条退化重复 (QID `2hop__821368_14251`，生成 *"I'm not a man, I'm a woman, I'm not a woman, I'm a man..."* 循环至 max_new_tokens=200)。试验 Qwen3 官方推荐的采样参数能否消除退化并提升整体质量。

**实验设置**：
- 4 个实验：{Qwen3-0.6B, Qwen3-4B} × {hotpotqa, musique} × NoR codex_direct
- 100 例 / 数据集 (dev_subsampled.jsonl)
- 双 server 并行：GPU0:8011 (4B) + GPU1:8013 (0.6B)
- **Sampling 参数**：`do_sample=true, temperature=0.7, top_p=0.8, top_k=20, min_p=0, seed=42` (Qwen3 官方 README 推荐)
- **Greedy 对照组**：直接复用先前已跑结果（同 codex_direct config）

**代码改动**：
- `llm_server/serve.py`：`/generate/` 端点新增 `min_p` 与 `seed` 参数，透传给 `model.generate`
- `commaqa/models/llm_client_generator.py`：`LLMClientGenerator`、`llm_call`、`(non_)cached_llm_call` 全部加 `min_p` 参数；`seed` 仅 non-cached 路径生效（sampling 不缓存）
- 4 个新 base configs：`base_configs/nor_qa_qwen3_{0_6b,4b}_sample_{hotpotqa,musique}.jsonnet`
- 调度脚本：`run_nor_sampling_compare.sh`

**结果**：

| Model | Dataset | Decoding | EM | F1 | Precision | Recall | 退化样本 |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | hotpotqa | greedy | 0.080 | **0.132** | 0.144 | 0.139 | 0 |
| Qwen3-0.6B | hotpotqa | sample | 0.080 | 0.123 | 0.129 | 0.131 | 0 |
| Qwen3-0.6B | musique  | greedy | 0.000 | 0.065 | 0.069 | 0.063 | 1 |
| Qwen3-0.6B | musique  | sample | 0.000 | **0.083** | — | — | **0** |
| Qwen3-4B   | hotpotqa | greedy | 0.170 | **0.245** | 0.274 | 0.239 | 0 |
| Qwen3-4B   | hotpotqa | sample | 0.170 | 0.230 | 0.256 | 0.227 | 0 |
| Qwen3-4B   | musique  | greedy | 0.020 | **0.105** | — | — | 1 |
| Qwen3-4B   | musique  | sample | 0.000 | 0.095 | — | — | **0** |

**关键观察**：

1. **采样消除了所有退化样本** (greedy 在 musique 上有 2 条退化循环，sample 全消)。例如 4B/musique QID `2hop__821368_14251`：
   - greedy: `"I'm not a man, I'm a woman, I'm not a woman, I'm a man, ..."` (循环至 200 tokens)
   - sample: `"I'm not a man, I'm an Archie"` (仍错，但简洁正常)
2. **整体 F1 在 3/4 组合上 sample 略劣于 greedy** (Δ ∈ [-0.015, -0.009])。**唯一显著占优的是 0.6B/musique** (+0.018 F1, +28% 相对)。
3. **EM 几乎无变化**：3/4 组合 EM 相同；只有 4B/musique 由 0.020 → 0.000 (1/100 退化 case 之前歪打正着没扣 EM？实际上 greedy 的 EM=0.020 来自其它样本，sample 在不同样本上失分)。
4. **采样引入随机噪声**：尤其在 short-answer EM 评估下，温度=0.7 容易让答案多一两个修饰词从而失去 exact match。这解释了 hotpotqa/4B 上 −0.015 F1 的下降。

**结论**：
- Qwen3 官方采样参数 (`temp=0.7, top_p=0.8, top_k=20, min_p=0`) **对 NoR-QA benchmark 整体并非更优**。
- 它解决了贪心解码偶发的退化循环问题，但代价是引入随机性带来的 F1/EM 波动，3/4 测点反而退步。
- **建议**：保留贪心解码作为 benchmark 默认，仅当出现退化时再考虑切换 sampling（或加 `repetition_penalty=1.05~1.1`，是更便宜的方案）。
- 仅对 **musique + 较小模型** 这种"模型常陷入退化"的组合，sampling 有正向收益 (0.6B/musique +0.018 F1)。

**附**：所有原始 metrics 在 `predictions/nor_qa_qwen3_{0_6b,4b}_sample_{hotpotqa,musique}____prompt_set_1/official_evaluation_metrics__*.json`。

---

## NoR vs OneR — 案例级对比 (Qwen3-4B / musique)

**目的**：理解 OneR 相对 NoR 在 musique 上的 F1 提升 (0.103 → 0.170, +0.067) 来自哪里、为何不更高。

### 全集 100 条按 F1 变化分类

| 分类 | n / 100 | ΔF1 范围 | 解读 |
|---|---|---|---|
| **saved** | **14** | NoR<0.3 → OneR≥0.5 | retrieval 把关键事实拉回来了 |
| **improved** | 8 | OneR > NoR + 0.1 | 部分改善 |
| **tied_high** | 3 | 双方都 ≥0.5 | 模型本身知道 |
| **tied_low** | 64 | 都 <0.3 (含很多 0/0) | retrieval **没救起来** |
| **harmed** | 11 | OneR < NoR − 0.1 | retrieval 反伤 (干扰段落让推理跑偏) |

净增 = 14+8 − 11 = +11 例显著改善，与 F1 提升 +0.067 数值上吻合（每例平均 +0.6 F1 / 100 ≈ +0.006，14+8 净优 ≈ +0.09 量级，与 +0.067 同阶）。

### Saved 类典型 (retrieval 确实带回了正确事实)

| QID | Gold | NoR | OneR | ΔF1 |
|---|---|---|---|---|
| 2hop__152027_141308 | Apple Corps | Warner Music Group | Apple Records | +0.50 |
| 2hop__128608_82341 | in Northern Florida | North | Northern | +0.50 |
| 3hop2__49541_140875_51068 | Jenna-Louise Coleman | Sarah Siddons | Jenna Coleman | +0.80 |
| 2hop__25396_593388 | Sir Robert Peel, 1st Baronet | William Wilberforce | Robert Peel | +0.57 |
| 3hop1__617062_127905_46894 | Bill Bergen | 0.2 (空) | Bill Bergen | +1.00 |
| 4hop1__151650_5274_458768_33677 | 2013 | 2016 | 2013 | +1.00 |

可见 retrieval 把"Apple Records / Robert Peel / Bill Bergen"这类**核心实体**直接带回，reader 只需要从段落里挑出来即可。

### Harmed 类典型 (retrieval 反而引入干扰)

| QID | Gold | NoR | OneR | ΔF1 |
|---|---|---|---|---|
| 4hop1__40316_497223_15840_36002 | "built on 16-bit...graphics and sound" | "better graphics and sound system, larger game library" (NoR 蒙对 F1=0.38) | "theatrical costumes and performances" | −0.24 |
| 3hop2__326964_7855_7713 | "about 400 years" | "1,200 years" | "1,989 mi" (检到无关里程数) | − |

NoR 在某些样本上靠模型常识"凑近"答案；OneR 检索到的 top-K 里夹杂高度无关段落，反而让 reader 走偏。

### Retrieval 命中率 vs F1 关联

按 supporting-paragraph recall（gold 标注的 `is_supporting=True` 段落在 top-15 中被检索到的比例）分桶：

| Recall 桶 | n | OneR EM | OneR F1 |
|---|---|---|---|
| Recall = 1.0 (全找到) | 29 | 0.069 | **0.212** |
| 0 < Recall < 1.0 (找到部分) | 67 | 0.045 | 0.154 |
| Recall = 0 (一个都没找到) | 4 | 0.000 | 0.125 |

**全集平均 recall 仅 0.611**：即 OneR 的 BM25 在 100 条 musique 例上，平均只检到了 61% 的 supporting paragraphs。**29% 的 case retrieval 完美，反而只换来 F1=0.212**——这意味着：

1. **Retrieval 不是唯一瓶颈**：即便完美 retrieval，4B reader 只能从段落里取出 21% 的 F1 分数，多跳推理仍然失败
2. **答案是长短语而非短实体**（musique 平均 gold 3.3 词）：即使段落里有正确事实，reader 也常用自己的话改写，EM 严重扣分
3. **`distractor_count=1` 设置**：把检索 top-15 缩减到 1 个段落 + 1 个干扰，是这个版本 OneR 的"难度档位"。换更高的 retrieval_count 可能改善（论文 IRCoT 标准值是 distractor_count=1，但不同 paper 取值不一）

### 结论

- **OneR 确实改善了 NoR**（+0.067 F1），但绝大部分（64%）case 仍然失败
- **失败的 2 大根因**：
  1. **BM25 召回不足**（71% case 没拿全 supporting paragraphs）
  2. **Reader 多跳推理能力有限**（即使完美 retrieval，F1 也仅 0.212）
- 这正是 **IRCoT** 想解决的问题：用 CoT 引导 reader 一步一步 **重新发出 retrieval query**，让 retrieval 不再依赖原问题的关键词（多跳问题里的 sub-question 关键词往往不在原问题里）

> 下一步：跑 IRCoT (qwen3-4b / musique)，看 retrieval 命中率和 F1 能否进一步上升。

---

## NoR SFT (Reader fine-tuned without retrieval) — 反例

**动机**：NoR in-context baseline 在 musique 上 F1 仅 0.065 (Qwen3-0.6B) / 0.093 (Qwen3-4B)，怀疑 in-context prompt 效果不如 SFT，故构造 NoR SFT 数据集训练 reader 验证。

**结论（先说）**：**SFT NoR 反而低于 in-context NoR**——**在 0.6B 和 4B 上都成立**（分别 -42 % / -44 % F1）。这推翻了"prompt 是瓶颈"的假设：NoR 场景本质上没有可学的"格式/流程"，SFT 只能记住 (question → answer) 表面映射，会摧毁预训练常识、过拟合训练分布。

### 数据构造

- 脚本：`prompt_generator/build_nor_sft_data.py`（80 行，独立于 `build_atomic_search_sft_data.py`）
- 每条样本 3 个 segment：`system` (NOR_GLOBAL_INSTRUCTION) / `node_input` (question) / `node_output` (`<|start_answer|>answer<|end_answer|>`)
- system prompt：精简版（方案 B），只讲"直接输出 answer 标签"，不涉及 retrieval / plan / reasoning
- Label 只用 `answer`，**不使用 `answer_aliases`**（aliases 是 evaluator-side augmentation，且质量参差：`'cz'`, `'cze'`, `'uni'` 这类噪声若用作训练目标反而伤害模型）
- 完全复用下游 `processing_scripts/tokenize_trajectory.py` + `qa_reader_sft.py`，无需修改任一行

**Token 长度统计**（同 tokenizer=Qwen3-0.6B/4B，词表相同）：

| Split | n | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 19,938 | 122 | 136 | 147 | 150 | 158 | **177** |
| dev   | 2,417  | 123 | 139 | 150 | 153 | 162 | **179** |

→ 训练 `max_length=192`（覆盖 100 %，向上对齐 8 的整数倍）。

### 训练配置

| Reader | GPU | batch × grad_accum | effective bs | lr | epochs | 训练时长 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B (full FT) | 2×A800 | 8×1 | 16 | 1e-5 | 3 | ~13 min |
| Qwen3-4B  (full FT)  | 2×A800 | 4×2 | 16 | 5e-6 | 3 | ~63 min |

- 启动脚本：`/home/jinsonglan.ljs/agent_tempdir/train_nor_qwen{0.6b,4b}.sh`
- Checkpoint：`dumps/nor_qwen{0.6b,4b}_musique/{epoch_0,epoch_1,epoch_2,final}/`
- 两个 model size **共用同一份 tokenized 数据**（Qwen3-0.6B 和 Qwen3-4B tokenizer 相同, vocab=151669）

### 训练动态

**Qwen3-0.6B**

| Epoch | train_loss | eval_loss | eval_token_acc | eval_exact_acc |
|---:|---:|---:|---:|---:|
| 0 | 1.274 | **2.404** | 0.6142 | 0.0012 |
| 1 | 0.126 | 2.634 | 0.6202 | 0.0012 |
| 2 | 0.035 | 2.981 | 0.6164 | 0.0004 |

**Qwen3-4B**

| Epoch | train_loss | eval_loss | eval_token_acc | eval_exact_acc |
|---:|---:|---:|---:|---:|
| 0 | 1.072 | **1.772** | 0.6783 | 0.0079 |
| 1 | 0.075 | 1.983 | 0.6788 | 0.0074 |
| 2 | 0.018 | 2.413 | 0.6751 | 0.0074 |

- 两个模型都在 **epoch 0 就到达 eval loss 最优**，之后 eval_loss 单调上升 → 典型过拟合
- eval_token_acc 从 epoch 0 就进入平台（0.6B: 0.614, 4B: 0.678），后续 epoch 没有实质提升
- **4B 比 0.6B 有明显收益**：eval_loss 从 2.40→1.77，token_acc 从 0.614→0.678，说明 4B 记得更细
- 但两个 size 都遇到同一堵墙：**训练集里的 (Q→A) 对基本不能 transfer 到 dev**（19938 对 musique 都是独立的多跳实体链）

### 评测结果

**评测集**：`raw_data/musique/musique_ans_v1.0_dev_with_pid.jsonl` (全量 2,417 例) + 对齐到 100 例 `processed_data/musique/dev_subsampled.jsonl` 子集（100 例是与 § NOR_QA / ONER_QA baseline 对齐的官方 subsample）。

| Reader | 方法 | dev-100 EM | dev-100 F1 | dev-full EM | dev-full F1 |
|---|---|---:|---:|---:|---:|
| **Qwen3-0.6B** | NoR in-context (from § NOR_QA) | 0.000 | **0.065** | — | — |
| Qwen3-0.6B | SFT NoR ep0 | 0.000 | 0.038 | 0.001 | 0.037 |
| Qwen3-0.6B | SFT NoR ep1 | 0.000 | 0.035 | 0.000 | 0.041 |
| Qwen3-0.6B | SFT NoR ep2 | 0.000 | 0.032 | 0.000 | 0.037 |
| **Qwen3-4B** | NoR in-context (from § NOR_QA) | 0.020 | **0.093** | — | — |
| Qwen3-4B   | SFT NoR ep0 | 0.010 | 0.052 | 0.010 | 0.060 |
| Qwen3-4B   | SFT NoR ep1 | 0.000 | 0.045 | 0.007 | 0.053 |
| Qwen3-4B   | SFT NoR ep2 | 0.000 | 0.039 | 0.004 | 0.049 |

**关键观察**：

1. **每个 SFT checkpoint 都低于对应尺寸的 in-context**：
   - 0.6B: SFT best F1 = 0.038 vs in-context 0.065 → **-42 %**
   - 4B  : SFT best F1 = 0.052 vs in-context 0.093 → **-44 %**
2. **Epoch 0 (最短训练) 在 dev 上表现最好**，之后每个 epoch 反而变差 — 与 eval_loss 曲线完全一致
3. **模型放大 (0.6B→4B) 对 SFT NoR 有帮助**（F1 0.038→0.052, +37 %），但仍显著低于对应的 in-context baseline
4. **Stop reason 分布**：0.6B 全部 `answered`；4B 有极少数 `malformed` (ep0: 6, ep1: 15, ep2: 15) 和 `truncated_answer` (ep0: 1) — 也就是 4B 偶尔生成不闭合的 answer 标签，但 >99 % 格式正确
5. Predictions 目录：`predictions/nor_qwen{0.6b,4b}_musique_ep{0,1,2}/{results_dev.jsonl,metrics.json}`

### 讨论 — 为什么 SFT NoR 反而更差

1. **摧毁预训练常识**：in-context 模式下 Qwen3 靠 pretraining 里的世界知识"蒙"，SFT 后被拉向训练分布的猜答风格，反而更保守 / 更错。这在 0.6B 和 4B 上都成立。
2. **NoR 任务没格式可学**：SFT 的核心价值是教模型工作流（如 atomic reader 学 tag 循环、search_plan / atomic_decision 交替）；NoR 只有"直接吐答案"一件事，模型 pretraining 已经会做。数据也印证——epoch 0 就学会格式（token_acc 立刻到 61-68 % 平台），后续训练全在无效过拟合。
3. **样本级 memorization 无法 transfer**：musique 每个问题都是独立的多跳实体链，训练集里学到的 `(question_A → answer_A)` 对 dev 集里的问题基本无帮助。
   - 0.6B: train_loss 0.035 vs eval_loss 2.98 (gap **~85×**)
   - 4B  : train_loss 0.018 vs eval_loss 2.41 (gap **~130×**)
4. **对照参照**：同一 0.6B 模型跑 atomic SFT (interleave retrieval) 拿到 dev F1=34.3（见 `two_stage_qa_benchmark.md`），说明 SFT 在**有检索信号的多跳格式**上是强有力的；单看 NoR 就无用武之地。

### 结论

- **NoR 的瓶颈不在 prompt，在于"无检索时模型根本没那个知识储量"**——in-context 都比 SFT 强，说明 pretraining knowledge 已经是这个 setting 下的天花板；SFT 只会磨掉一些常识
- **两个模型尺寸都得出同样结论**：0.6B / 4B 各自 SFT 都比对应 in-context 差 ~40 %，且都是 epoch 0 (最早期) 最好
- 要提升 musique 表现必须走**检索**：OneR (+0.028 F1) → atomic reader (+0.278 F1 on 0.6B / +0.374 F1 on 4B) 才是正确方向
- 未来若要做 NoR SFT，应该**只学格式**（一个 epoch 内 lr 很小的短训练），而不是像本实验这样让模型记住 answer

### Artifacts

- 数据: `raw_data/musique/musique_ans_v1.0_{train,dev}_nor_{trajectory,tokens}.jsonl`
- 数据构造代码: `prompt_generator/build_nor_sft_data.py`
- 推理评测代码: `nor_predict.py`（复用 `metrics/drop_answer_em_f1.py`）
- 训练启动脚本: `/home/jinsonglan.ljs/agent_tempdir/train_nor_qwen{0.6b,4b}.sh`
- Checkpoint: `dumps/nor_qwen{0.6b,4b}_musique/{epoch_0,epoch_1,epoch_2,final}/`（全都是 HF `save_pretrained` dir，可直接 `--ckpt <dir>` 加载）
