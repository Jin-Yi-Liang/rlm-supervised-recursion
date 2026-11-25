# Planning-Only RLM 开发报告

`experiment/planning-only-thinking` 基于 `a95c0c8e58bd3456ae68f04e21d5071f35328d14`（`codex/supervised-rlm-v1`）。Root 保留版本化全局计划，Child 使用 Thinking 在各自的持久化 REPL 中求解。Runtime 在结构校验通过后直接持久化结果并签发回执。

## 系统架构

```text
用户问题与原始数据
        │
        ▼
SessionRuntime ──► Root Agent（Thinking ON、持久化 REPL）
                           │              ▲
                           │              │ 回执、进度与结果分页
                           ▼              │
              PlanManager ────────► 结果回执与 Task Ledger
              版本化 Plan / Task            ▲
                      │                    │ 发布
                      ▼                    │
         rlm_query / rlm_query_batched      │
                      │                    │
                      ▼                    │
              Child Agent ────────► SubmissionRegistry
              Thinking ON           提交变量、确定性校验
              独立持久化 REPL ◄────── 结构错误反馈
                                           │
                                           ▼
                                  ArtifactStore / Commit JSON

Root 与 Child ──► 共享 AgentLoop ──► 模型后端
       │                 │
       └───────────────► wire / agents / events / metrics
```

Root 继续创建、复制、提交和修订计划草稿，定义任务范围、数据引用、必需任务、验收描述和输出结构。计划修订保留版本检查，变更已完成任务会使旧回执失效。Root 在所有当前必需任务均有有效回执后才能提交全局结果。

任务间的数据依赖沿用基线的 Root 调度方式。Root 等待前置任务回执，读取结果后用 `store_input()` 生成下游任务输入，再派发下游任务。基线的 `TaskSpec` 没有显式依赖边或自动拓扑调度。本版验证了这一依赖传递过程。

## 执行流程和状态

| 阶段 | Supervised RLM v1 | Planning-Only RLM |
| --- | --- | --- |
| Root 规划 | 提交版本化计划，按任务 ID 派发 | 沿用版本化计划和单个、批量派发工具 |
| Child 求解 | Thinking OFF，私有 REPL，多轮执行 | Thinking ON，沿用同一套 REPL 和 Agent Loop |
| 结果处理 | 结构校验通过后创建 Supervisor 会话，按审核结论处理 | 结构校验通过后存储结果并发布回执 |
| 处理非法提交 | 程序化错误反馈或 Supervisor 驳回后再次推进 Child | 程序化错误直接回到当前 Child 会话 |
| 汇总 | Root 读取通过审核的回执 | Root 读取通过确定性校验的回执 |

```text
基线
PLANNED → RUNNING → SUBMITTED → REVIEWING → ACCEPTED
                        REVIEWING → REPAIRING → RUNNING
                        REVIEWING → PLAN_ISSUE / REVIEW_INCOMPLETE

本版
PLANNED → RUNNING → SUBMITTED → ACCEPTED
             │          └→ FAILED（回执发布失败）
             ├→ RUNNING（结构校验失败，继续当前会话）
             └→ FAILED（Child 步数、全局预算或执行异常）
```

`CANCELLED` 和 `SUPERSEDED` 继续用于取消及计划修订。预算耗尽会保留失败原因。Root 耗尽预算会产出带原因的 `INCOMPLETE`。

### 子任务提交调用链

1. `SubAgentDispatcher.run_child()` 在 Child 的 REPL 注册 `submit_candidate(result_var, evidence_vars=None, summary="")`。REPL 先登记提交意图，成功执行完整代码块后返回被引用变量的快照。
2. `SubmissionRegistry.finalize_submission()` 检查变量存在且可序列化为 JSON，并校验输出类型、必需字段和 `expected_count`。非法结果会反馈具体错误，Task 继续处于 `RUNNING`。
3. 校验通过后，Runtime 在计划锁内验证任务与候选版本，冻结结果 Artifact 和显式提交的可选证据，写入 Candidate Manifest，将 Task 推进到 `SUBMITTED`。
4. Runtime 将回执与 `ACCEPTED` ledger 记录写到同一份原子提交文件，再发布给内存中的计划状态。同一提交重入会返回已有回执。回执只能读取当前计划仍有效的结果。
5. Root 通过 `rlm_query()` 或批量结果拿到回执，经 `read_accepted_result()` 读取可分页的 Artifact。最终提交检查必需任务、活跃任务、回执归属和版本。

## 文件变更

| 操作 | 文件 | 变化与原因 |
| --- | --- | --- |
| 修改 | `rlm/supervised_rlm/__init__.py` | 更新包说明，保留 `SessionRuntime`、`RunConfig` 和 `QwenClient` 的导入入口 |
| 修改 | `rlm/supervised_rlm/runtime.py` | 装配提交注册表，导出有效回执 |
| 修改 | `rlm/supervised_rlm/root_agent.py` | 从提交注册表读取回执，用任务和候选版本校验完成条件，记录 Root 步数上限 |
| 修改 | `rlm/supervised_rlm/plan_manager.py` | 清除审核状态，把结果发布交给持有回执的提交注册表 |
| 修改 | `rlm/supervised_rlm/schemas.py` | 清除审核角色、状态、数据类型和预算。回执关联提交 ID 与任务版本 |
| 修改 | `rlm/supervised_rlm/subagent.py` | 沿用单个与批量派发、持久化会话和预算控制，改为直接提交 |
| 修改 | `rlm/supervised_rlm/submission.py` | 复用输出校验，集中持久化 Candidate、Artifact 与回执 |
| 修改 | `rlm/supervised_rlm/repl.py` | 沿用 REPL RPC，通过内部工具传递当前提交引用的变量名 |
| 修改 | `rlm/supervised_rlm/repl_worker.py` | 只冻结提交引用的变量，去除审核变量目录适配，保留现有 Python 执行与隔离机制 |
| 修改 | `rlm/supervised_rlm/prompts.py` | 清除 Root 和 Child 的审核说明以及 Supervisor 系统提示，保留计划、分片和汇总提示 |
| 修改 | `rlm/supervised_rlm/client.py` | 两角色共用 Thinking 配置，保留模型正文与推理字段分离，记录正文及截断反馈 |
| 修改 | `rlm/supervised_rlm/metrics.py` | 移除审核事件和聚合逻辑，补充提交、失败、推理、缓存、截断和步数指标。旧分析字段只在 JSON 导出层保留空值或零 |
| 修改 | `examples/supervised_v1_smoke.py` | 保留 Qwen smoke 命令及其预算，改测两个直接提交结果的求和 |
| 新增 | `examples/planning_only_case12.py` | 复用历史 Case 12 选择和计分，接入 DeepSeek Flash 的 Thinking 参数与同一套分词器 |
| 新增 | `tests/test_planning_only.py` | 保留计划、REPL、预算及指标的测试，补充直接提交、回执一致性和 HTTP Mock |
| 删除 | `rlm/supervised_rlm/supervisor.py` | 移除 Supervisor Agent Loop、独立 REPL 和审核工具注册 |
| 删除 | `rlm/supervised_rlm/review_gate.py` | 移除 LLM 审核结果校验和审核驱动的发布入口 |
| 删除 | `rlm/supervised_rlm/state_broker.py` | 移除 Supervisor 专用的候选、快照与轨迹读取接口 |
| 删除 | `rlm/core/supervised.py`、`rlm/core/supervised_prompts.py` | 移除早期 Pilot 的另一套监督实现和提示词。Pilot 历史实现可从原实验分支获取 |
| 删除 | `examples/supervised_oolong32k.py` | 移除早期 Pilot 的运行入口 |
| 删除 | `tests/test_supervised.py`、`tests/test_supervised_v1.py` | 替换绑定已删除审核链路的回归测试 |
| 修改 | `README.md` | 更新本分支使用入口 |
| 删除 | `docs/supervised-oolong.md`、`docs/supervised-rlm-v1.md` | 移除本分支已退役运行入口的说明 |
| 新增 | `docs/planning-only-rlm.md` | 记录实现、验证与 Case 12 复现命令 |

## Thinking、预算与日志

Qwen 向 `/tokenize` 和 Chat Completions 发送 `chat_template_kwargs.enable_thinking=true`。DeepSeek Flash 通过 Chat Completions 发送 `thinking.type=enabled`，本地 tokenizer 也使用 Thinking 模式。HTTP Mock 直接检查了两种后端的序列化正文，覆盖 Root、单个 Child 和批量 Child。[DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/)

模型回复的 `reasoning_content` 保存在 `response.json`。REPL 仅解析模型正文中的完整代码块。正文为空或达到 `finish_reason=length` 且没有完整代码块时，记录本轮并在既有步数预算内继续。每次成功响应保留 `meta.json`、`request.json`、`response.json` 和 `model-output.txt`。网络或 API 失败保留请求与错误文件。消息和执行轨迹写入 `agents/`，任务状态写入事件流和最终 ledger。

Case 12 沿用 `deepseek-flash`、32,768 token 上下文、4,096 token 输出上限、Root 与 Child 各 20 步、单并行 Child、480 次模型调用上限、2,000,000 token 全局预算、每个 Python cell 1,024 次工具调用上限和 3,600 秒截止时间。发送的采样参数仍为 temperature 0.6、top-p 0.95、presence penalty 0.0。DeepSeek 对这些参数的实际处理随 Thinking 模式而定。[DeepSeek 输入输出参数](https://api-docs.deepseek.com/guides/thinking_mode/#input-and-output-parameters)

`metrics.json` 提供总计和按角色统计的模型调用、输入输出 token、推理 token、缓存命中与未命中 token、耗时、截断、迭代上限和任务数量。模型服务未提供的数据保留为 `null`。另行记录工具执行成功率、提交尝试和结构合规率。结构合规率以进入结构校验的提交为分母，Python cell 中止单列。历史汇总所需的 Supervisor 调用数和审核计数固定为零。

## 验收

| 验收项 | 验证方式 |
| --- | --- |
| 全局计划与任务依赖 | 脚本模型创建及修订计划，拒绝过早结束，从前置任务回执构造下游输入 |
| 模型请求中的 Thinking | Qwen 与 DeepSeek 的 HTTP Mock 同时覆盖 Root、单个和批量 Child |
| Child 独立、多轮 REPL | Child 在多轮中读写变量，另一个 Child 看不到该变量，历史保留工具反馈 |
| 合法提交 | 直接写 Artifact、回执与 ledger，验证分页、重复派发幂等、计划修订使旧回执失效 |
| 非法提交 | 实测缺少变量、类型不符、缺少字段、数量不符和 Python cell 失败，Child 在同一会话收到错误 |
| 失败和预算 | 验证步数、全局 token 和模型调用上限。模拟回执写入失败，必需任务无法完成 |
| 结果汇总 | 两个独立 Child 和后续聚合任务经 Root 得出最终结果 `15` |
| Supervisor 残留 | 静态检查运行期无审核模块引用或审核状态。Mock 事件只有 Root 和 Child，兼容汇总中 Supervisor 为零 |

本地定向回归为 **45 通过**。除 Docker 的两个测试文件以外，仓库测试为 **283 通过，8 跳过**。此次修改涉及的 Python 文件通过 Ruff 检查和格式检查。全库 Ruff 报告两处基线已有的训练模块导入排序问题。

```bash
.venv/bin/python -m pytest -q tests/test_planning_only.py tests/test_rlm_query.py
.venv/bin/python -m pytest -q tests \
  --ignore=tests/test_docker_repl_persistent.py --ignore=tests/test_docker_repl_robustness.py
```

## 复现 OOLONG-Pairs 32K Case 12

在集群 `adviser-jyl-qwen35` 命名空间的 `rlm-harness` Pod 内，从项目目录运行。复用历史 benchmark 虚拟环境及已保存的 Case 12 数据选择器、模型输入、分词器和计分代码。

```bash
cd /workspace/rlm-supervised-recursion
baseline=runs/oolong_pairs_32k-supervised-rlm-v1-20261124T043045
encoder=runs/oolong_pairs_32k-deepseek-flash-case12.79zqPW
legacy=/workspace/rlm-repro/experiments/rlm-4lengths-5090-128k-20261110T102957Z/32768/test_oolong_pair_32768.py
benchmark_python=/workspace/rlm-repro/benchmark-venv/bin/python
export PYTHONPATH="$PWD:$encoder/deps"

"$benchmark_python" -m examples.planning_only_case12 \
  --historical-runner "$baseline/run_oolong_pairs_32k_supervised_rlm_v1.py" \
  --legacy "$legacy" --tokenizer-dir "$encoder" --dry-run
```

实际运行前，通过私密环境变量注入 `DEEPSEEK_API_KEY`，然后在同一 shell 中运行以下命令。

```bash
out="runs/oolong_pairs_32k-planning-only-thinking-case12-$(date -u +%Y%m%dT%H%M%SZ)"
"$benchmark_python" -m examples.planning_only_case12 \
  --historical-runner "$baseline/run_oolong_pairs_32k_supervised_rlm_v1.py" \
  --legacy "$legacy" --tokenizer-dir "$encoder" --out "$out"
```

入口校验 Case 12 问题与上下文和历史输入逐项相等，再将问题与上下文交给 Runtime。原始 scorer 只在 Runtime 产出最终答案后运行。逐例状态、pair F1 和 exact match 保存在 `case-12/result.json`，完整运行记录保存在 `case-12/supervised/`。前置检查不发模型请求。

## 限制

- `ACCEPTED` 仅表示程序化校验通过。自然语言验收描述仍随任务交给 Child。端到端语义质量由独立 scorer 衡量。
- Thinking 会消耗既有 4,096 token 输出额度，真实截断频率和 Case 12 得分待实验验证。
- 实验入口复用集群 PVC 中的历史 selector 和 tokenizer。运行时 API 允许传入其他实现同一协议的模型后端。
