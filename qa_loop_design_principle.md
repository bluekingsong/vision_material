# QA Loop 接口与设计原则

## 1. 文档目标

本文定义 plan-guided multi-hop QA 的运行时循环，以及 `main_function` 和 `retrieval_function` 两个逻辑服务之间的接口。

本文的直接读者是需要实现 QA loop 调用脚本的开发者或 Agent。实现者应当仅依赖这里定义的职责边界和输入输出协议，不应把训练数据内部字段误当成运行时接口。

核心原则：

1. **Main function 负责思考和控制流程**：规划、生成/改写 search intent、判断证据是否充足、继续检索或给出答案。
2. **Retrieval function 负责执行检索并校验结果**：消费 search intent，检索候选 passages，并通过生成式 relevance judgement 选出真正相关的 passage。
3. 两个 function 通过两种对象连接：
   - Main → Retrieval：`search_intent`
   - Retrieval → Main：经过相关性校验的 `observation`
4. 两个 function 在逻辑上可以由两个不同 LLM/服务实现；当前项目将两种能力训练到同一个模型中，但调用层仍必须保持功能隔离。
5. Main 不应直接看到未经校验的 top-k 检索结果；它只能看到 retrieval function 返回的 validated observation。

---

## 2. 总体 QA Loop

一个典型的多跳执行过程如下：

```text
question
   │
   ▼
main_function(search_plan)
   │  输出 plan + search_intent_0
   ▼
retrieval_function(search_intent_0)
   │  检索 top-k + relevance judgement
   ▼
validated observation_0
   │
   ▼
main_function(atomic_decision)
   │  输出 next search_intent_1，或者 final answer
   ▼
retrieval_function(search_intent_1)       （若继续搜索）
   │
   ▼
validated observation_1
   │
   └──────── 重复，直到 main 输出 answer 或达到循环上限
```

伪代码：

```python
main_history = initialize(question)

main_output = call_main(task="search_plan", history=main_history)
append_main_output(main_history, main_output)

for hop in range(max_hops):
    if main_output.action == "answer":
        return main_output.answer

    search_intent = main_output.search_intent
    observation = call_retrieval(search_intent)
    append_observation(main_history, observation)

    main_output = call_main(task="atomic_decision", history=main_history)
    append_main_output(main_history, main_output)

raise MaxHopsExceeded(...)
```

注意：第一次 `search_intent` 由 `search_plan` 调用产生；后续 `search_intent` 由 `atomic_decision` 调用产生。

---

## 3. Main Function

### 3.1 职责

Main function 是 QA agent 的高层控制器，负责：

- 理解原始问题；
- 生成 atomic search plan；
- 生成第一个 search intent；
- 维护并利用完整 main trajectory；
- 阅读经过 relevance 校验的 observations；
- 判断当前证据是否足够；
- 证据不足时生成下一条或改写后的 search intent；
- 检索失败时参考之前的 search intent 和失败 observation 做 query rewrite；
- 证据完整时输出最终答案。

Main function **不负责**：

- 执行向量/树检索；
- 阅读未经校验的 top-k candidates；
- 在多个候选 passage 中选择相关 PID；
- 根据检索 score 直接做证据判断。

### 3.2 Main 的上下文

Main history 应保留下列信息：

1. 原始问题；
2. search plan；
3. 每一轮已经生成的 search intent；
4. 每一轮 retrieval function 返回的 observation；
5. 之前的 atomic decision/reasoning。

尤其不能删除旧的 search intent。query rewrite 需要知道“上一轮搜索了什么”和“上一轮为什么没有获得相关结果”。

### 3.3 初始调用：`search_plan`

#### 输入

```text
Current task: search_plan

Original question:
{question}
```

运行时可以通过独立 system prompt 告诉模型 main function 的职责，但不要把 system/global instruction 当作 trajectory 的普通历史段反复追加。

#### 输出

```text
<|start_atomic_search_plan|>
{atomic plan}
<|end_atomic_search_plan|>
<|start_atomic_search_intent|>
{first search intent}
<|end_atomic_search_intent|>
```

解析规则：

- `atomic_search_plan` 是面向整个多跳问题的计划；
- `atomic_search_intent` 是当前只解决一个缺失事实的检索意图；
- 第一次调用必须产生 intent，随后调用 retrieval function；
- 不应在 plan 阶段提前输出最终答案。

示例：

```text
<|start_atomic_search_plan|>
1. Identify the person who is a member of The Bruce Lee Band.
2. Determine which record label this person started.
<|end_atomic_search_plan|>
<|start_atomic_search_intent|>
Find out who is a member of The Bruce Lee Band.
<|end_atomic_search_intent|>
```

### 3.4 后续调用：`atomic_decision`

#### 输入

当前节点输入为：

```text
Current task: atomic_decision
```

但模型实际上下文必须包含此前完整 main history，例如：

```text
Current task: search_plan
Original question: ...

<plan and first intent>

Observation:
pid: ...
title: ...
text: ...

Current task: atomic_decision
```

#### 输出分支 A：继续检索

```text
<|start_atomic_search_reasoning|>
{why evidence is incomplete / why the query should be rewritten}
<|end_atomic_search_reasoning|>
<|start_atomic_search_intent|>
{next or rewritten search intent}
<|end_atomic_search_intent|>
```

判断输出包含完整的 `atomic_search_intent` 标签后，loop 应继续调用 retrieval function。

失败后的 query rewrite 示例：

```text
<|start_atomic_search_reasoning|>
The previous retrieval did not return evidence identifying a band member, so I need a more direct query.
<|end_atomic_search_reasoning|>
<|start_atomic_search_intent|>
Bruce Lee Band members
<|end_atomic_search_intent|>
```

#### 输出分支 B：停止并回答

```text
<|start_atomic_search_reasoning|>
{why the validated evidence is sufficient}
<|end_atomic_search_reasoning|>
<|start_answer|>
{final answer}
<|end_answer|>
```

判断输出包含完整的 answer 标签后，loop 应立即停止并返回 answer，不得再次调用 retrieval function。

`atomic_decision_answer` 中不得输出旧格式：

```text
Support pids: ...
```

该行已从 SFT 数据移除。原因是历史数据中的 PID 曾经过 uint64 → signed int64 迁移，旧 support PID 可能不正确，而且最终回答接口不需要暴露 support PID。

### 3.5 Main 输出的严格校验

调用脚本至少应校验：

- `search_plan` 输出同时有 plan 和 intent，且没有 answer；
- `atomic_decision` 恰好选择一种动作：
  - 有 intent、无 answer：继续检索；
  - 有 answer、无 intent：结束；
- 同时有 intent 和 answer：非法输出；
- 两者都没有：非法输出；
- 标签缺失或未闭合：非法输出，可重试模型调用；
- 不应依赖自然语言 reasoning 来决定动作，动作由结构化标签决定。

---

## 4. Retrieval Function

### 4.1 职责

Retrieval function 接受 main function 已经生成的一个 atomic `search_intent`，负责：

1. 将 search intent 编码成 retrieval query；
2. 从 corpus 检索多个候选 passages；
3. 把 search intent 和所有候选 passages 一起交给 relevance judgement；
4. 输出一个相关 passage 的 PID，或者 `none`；
5. 将 judgement 转换成标准 observation 返回给 main function。

Retrieval function **不负责**：

- 生成初始 search intent；
- 改写 search intent；
- 制定多跳计划；
- 判断整个问题是否已有足够证据；
- 直接生成最终问题答案。

### 4.2 Retrieval 输入

逻辑接口：

```json
{
  "search_intent": "Find out who is a member of The Bruce Lee Band."
}
```

训练用 doc-centric 数据中的等价字段名是：

```json
{
  "query_text": "Find out who is a member of The Bruce Lee Band.",
  "retrieval_id": [
    "graph::obs::success::0",
    "graph::obs::fail::0::0"
  ]
}
```

说明：

- `query_text` 就是 search intent；
- doc-centric 数据按 `(target document, query_text)` 去重；
- `retrieval_id` 是列表，因为同一个 query text 可能来自多次 retrieval call，例如一条成功轨迹和一条模拟失败轨迹；
- `retrieval_id` 仅用于数据追踪、调试和关联 SFT 样本，不应参与 query embedding，也不是运行时必需参数。

### 4.3 Retrieval Task

检索器根据 search intent 从全量 corpus 中返回 top-k candidates：

```python
candidates = retriever.search(search_intent, top_k=k)
```

每个 candidate 至少包含：

```json
{
  "pid": -435270577653286901,
  "title": "The Bruce Lee Band",
  "text": "The Bruce Lee Band ..."
}
```

PID 要求：

- JSON 中使用整数类型，不使用字符串；
- 当前数据使用 signed int64 形式，允许负数；
- 不要把负 PID 再转回 uint64；
- relevance judgement 输出的 PID 必须能在本次 candidates 中精确匹配。

### 4.4 Relevance Judgement 输入

Relevance judgement 是**针对一次检索返回的多个候选 passages 做一次联合判断**，不是每个 candidate 单独调用一次。

推荐 prompt 格式：

```text
Current task: relevance_judgement

Search intent:
{search_intent}

Retrieved candidate passages:
[Candidate 1]
pid: {pid_1}
title: {title_1}
text: {text_1}

[Candidate 2]
pid: {pid_2}
title: {title_2}
text: {text_2}

...
```

候选顺序应与 retriever 返回顺序一致。`Candidate N` 仅用于阅读；最终选择必须使用 PID，而不是 candidate index。

### 4.5 Relevance Judgement 输出

若某个 candidate 满足 search intent：

```text
RelevancePid:-435270577653286901
```

若所有 candidates 都不满足：

```text
RelevancePid:none
```

不要使用旧的标签格式：

```text
<|start_relevance_judgement|>
...
<|end_relevance_judgement|>
```

解析建议：

```python
prefix = "RelevancePid:"
value = output.strip().removeprefix(prefix).strip()
if value == "none":
    relevant_pid = None
else:
    relevant_pid = int(value)
```

严格校验：

- 输出应只有一条 `RelevancePid:...`；
- 非 `none` 值必须是整数；
- 该整数必须属于本轮 candidates；
- 若输出 PID 不属于 candidates，应视为 relevance 调用失败，而不是从 corpus 另行读取该 PID；
- 若有多个看起来相关的候选，当前接口仍只选择一个最符合 search intent 的 PID。

### 4.6 Retrieval Function 输出：Validated Observation

Retrieval function 不把 top-k candidates 全部返回给 main。它根据 `RelevancePid` 构造唯一的 observation。

#### 相关结果

```text
Observation:
pid: -435270577653286901
title: The Bruce Lee Band
text: The Bruce Lee Band (or B. Lee Band) is ...
```

建议服务层同时提供结构化对象：

```json
{
  "pid": -435270577653286901,
  "title": "The Bruce Lee Band",
  "text": "The Bruce Lee Band (or B. Lee Band) is ..."
}
```

#### 无相关结果

```text
Observation:
pid: null
title: 
text: No relevant result was found.
```

对应结构化对象：

```json
{
  "pid": null,
  "title": "",
  "text": "No relevant result was found."
}
```

Observation 是 retrieval → main 的唯一返回接口。无论成功还是失败，它在 main history 中都属于输入上下文，不计算 main SFT loss。

### 4.7 Retrieval 服务建议返回值

为了方便线上调试，retrieval 服务可以返回比 main 所需更多的 metadata：

```json
{
  "search_intent": "...",
  "candidates": [
    {"pid": 1, "title": "...", "text": "...", "score": 0.9}
  ],
  "relevance_raw_output": "RelevancePid:1",
  "relevance_pid": 1,
  "observation": {
    "pid": 1,
    "title": "...",
    "text": "..."
  },
  "observation_text": "Observation:\npid: 1\ntitle: ...\ntext: ..."
}
```

但调用 main function 时，只应追加 `observation_text`（或其等价结构化渲染），不要追加完整 candidates、score 或 relevance prompt。

---

## 5. 两个 Function 的边界

| 方向 | 对象 | 产生者 | 消费者 | 是否进入 Main History |
|---|---|---|---|---|
| Main → Retrieval | `search_intent` | Main | Retrieval | 是，作为 main 输出保留 |
| Retriever → Judge | top-k candidates | Retrieval 内部 | Relevance judgement | 否 |
| Judge → Retrieval wrapper | `RelevancePid:<pid/none>` | Relevance judgement | Retrieval wrapper | 否 |
| Retrieval → Main | validated observation | Retrieval | Main | 是，且无 SFT loss |
| Main → Caller | final answer | Main | QA loop caller | 是，循环终止 |

需要避免的错误实现：

1. 让 retrieval function 自己生成 search intent；
2. main function 直接读取 top-k passages；
3. relevance judge 对每个 candidate 分别做二分类，而不是联合选择 PID；
4. relevance 失败时把 rank-1 passage 当作 observation；
5. 把 `RelevancePid:none` 直接追加到 main history，而不是转换成标准失败 observation；
6. query rewrite 时丢掉上一条 search intent；
7. 使用旧 target PID 或把 signed int64 PID 转成 uint64。

---

## 6. 训练数据格式

转换脚本：

```text
~/GenRet/to/process_trajectory.py
```

输入：

```text
train_trajectory.jsonl
dev_trajectory.jsonl
```

主要输出：

```text
train.qg.full.jsonl   # retrieval 的 doc-centric train data
dev.jsonl             # retrieval 的 doc-centric dev data
train_sft.jsonl       # main + retrieval relevance SFT
dev_sft.jsonl         # main + retrieval relevance SFT
corpus_lite.jsonl
corpus_lite.json
doc_id_list.json
```

### 6.1 Doc-centric Retrieval 数据

训练文件每个文档一行：

```json
{
  "doc_id": -435270577653286901,
  "title": "The Bruce Lee Band",
  "doc_text": "The Bruce Lee Band ...",
  "query_list": [
    {
      "query_text": "Find out who is a member of The Bruce Lee Band.",
      "retrieval_id": [
        "2hop__...::obs::success::0",
        "2hop__...::obs::fail::0::0"
      ]
    },
    {
      "query_text": "Bruce Lee Band members",
      "retrieval_id": [
        "2hop__...::obs::success_after_repair::0::0"
      ]
    }
  ]
}
```

规则：

- `doc_id` 是整数；
- query 挂在其 gold target document 下；
- 同一文档内按 `query_text` 去重；
- 重复 query 的来源合并到 `retrieval_id` 列表；
- `query_list` 不保存冗余的 `target_pid` 或 `retrieval_task`。

### 6.2 Main Function SFT

一行是一条 main trajectory：

```json
{
  "task": "main_function",
  "graph_id": "...",
  "path_id": "...",
  "path_type": "positive",
  "segments": [
    {
      "segment_type": "node_input",
      "text": "Current task: search_plan\n\nOriginal question:\n...",
      "loss": false
    },
    {
      "segment_type": "node_output",
      "text": "<|start_atomic_search_plan|>...",
      "loss": true
    },
    {
      "segment_type": "retrieval_observation",
      "pid": 123,
      "title": "...",
      "passage_text": "...",
      "text": "Observation:\npid: 123\ntitle: ...\ntext: ...",
      "loss": false
    }
  ]
}
```

规则：

- node output 计算 NTP/SFT loss；
- node input 和 observation 不计算 loss；
- global instruction 不逐样本存储，避免重复浪费 token；
- 成功 observation 只包含被 relevance judge 选中的 passage；
- 失败 observation 使用标准 `No relevant result was found.`；
- answer reasoning 中不含 `Support pids:` 行。

### 6.3 Retrieval Relevance SFT

一次 retrieval call 对应一条联合 relevance judgement SFT：

```json
{
  "task": "retrieval_function",
  "graph_id": "...",
  "retrieval_id": "...",
  "segments": [
    {
      "segment_type": "relevance_input",
      "text": "Current task: relevance_judgement\n\nSearch intent:\n...\n\nRetrieved candidate passages:\n[Candidate 1]...",
      "loss": false
    },
    {
      "segment_type": "relevance_output",
      "text": "RelevancePid:123",
      "loss": true
    }
  ]
}
```

失败样本输出：

```json
{
  "segment_type": "relevance_output",
  "text": "RelevancePid:none",
  "loss": true
}
```

注意：`train_sft.jsonl` / `dev_sft.jsonl` 同时包含 `main_function` 和 `retrieval_function`。训练 loader 必须依据 `task` 和 segment loss mask 正确处理。

---

## 7. 推荐的服务 API

### 7.1 Main 服务

请求：

```json
{
  "task": "search_plan",
  "history": [
    {
      "segment_type": "node_input",
      "text": "Current task: search_plan\n\nOriginal question:\n..."
    }
  ]
}
```

或后续调用：

```json
{
  "task": "atomic_decision",
  "history": [
    {"segment_type": "node_input", "text": "..."},
    {"segment_type": "node_output", "text": "..."},
    {"segment_type": "retrieval_observation", "text": "Observation:\n..."},
    {"segment_type": "node_input", "text": "Current task: atomic_decision"}
  ]
}
```

响应建议统一解析为：

```json
{
  "raw_output": "...",
  "action": "search",
  "search_plan": "... or null",
  "reasoning": "... or null",
  "search_intent": "...",
  "answer": null
}
```

或：

```json
{
  "raw_output": "...",
  "action": "answer",
  "search_plan": null,
  "reasoning": "...",
  "search_intent": null,
  "answer": "Asian Man Records"
}
```

### 7.2 Retrieval 服务

请求：

```json
{
  "search_intent": "Bruce Lee Band members",
  "top_k": 5
}
```

响应：

```json
{
  "relevance_pid": -435270577653286901,
  "observation": {
    "pid": -435270577653286901,
    "title": "The Bruce Lee Band",
    "text": "The Bruce Lee Band ..."
  },
  "observation_text": "Observation:\npid: -435270577653286901\ntitle: The Bruce Lee Band\ntext: The Bruce Lee Band ..."
}
```

失败响应：

```json
{
  "relevance_pid": null,
  "observation": {
    "pid": null,
    "title": "",
    "text": "No relevant result was found."
  },
  "observation_text": "Observation:\npid: null\ntitle: \ntext: No relevant result was found."
}
```

---

## 8. Loop 终止、重试与异常处理

调用脚本应设置：

- `max_hops`：防止 main 永久继续搜索；
- main 输出格式重试上限；
- relevance 输出格式重试上限；
- retrieval 服务超时；
- 空 candidate list 的处理；
- 重复 intent 检测（用于日志或防循环，但不要未经策略直接拒绝 query rewrite）；
- 完整 trace 日志。

推荐行为：

1. Retriever 返回空 candidates：直接构造失败 observation，无需让 judge 幻想 PID；
2. Judge 返回不在 candidates 中的 PID：重试 judgement；重试仍失败则返回失败 observation并记录错误；
3. Main 输出非法 action：重试 main；
4. 达到 `max_hops`：显式返回执行失败，或调用一个受控的 forced-answer 分支；不要悄悄把最后一个 passage 当答案；
5. Retrieval 返回失败 observation 后，仍由 main 决定是否 rewrite、换下一跳或终止。

建议每轮记录：

```json
{
  "hop": 0,
  "main_raw_output": "...",
  "search_intent": "...",
  "candidate_pids": [1, 2, 3],
  "relevance_raw_output": "RelevancePid:2",
  "observation": {"pid": 2, "title": "...", "text": "..."}
}
```

这些 trace 只用于调试和评估，不应全部注入下一轮 main context。

---

## 9. 最终实现检查清单

实现 QA loop 时请逐项确认：

- [ ] 初始 main 调用使用 `search_plan`；
- [ ] search plan 同时产生第一条 search intent；
- [ ] 后续 main 调用使用 `atomic_decision`；
- [ ] 旧 search intents 保留在 main history 中；
- [ ] retrieval function 只消费 intent，不生成或 rewrite intent；
- [ ] relevance judgement 一次看到本轮全部 candidates；
- [ ] relevance 输出严格为 `RelevancePid:<int>` 或 `RelevancePid:none`；
- [ ] 非 none PID 必须属于本轮 candidates；
- [ ] main 只看到 validated observation，不看到 top-k candidates；
- [ ] observation 总是包含 `pid/title/text`；
- [ ] 失败 observation 的 pid 为 null，text 为固定失败文本；
- [ ] observation 不计算 main SFT loss；
- [ ] answer 输出后立即停止循环；
- [ ] PID 始终使用 signed int64 JSON integer；
- [ ] 不输出或依赖旧的 `Support pids:`；
- [ ] global instruction 由服务配置一次，不在 trajectory 中重复；
- [ ] 设置 max hops、超时、非法格式重试和 trace 日志。
