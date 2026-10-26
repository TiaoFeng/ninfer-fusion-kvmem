# `tools/` · 工具清单（2026-10-20 补入的三件）

> 来源：`E:\infer-build\kv-stream\` 开发树同字节拷贝（未改内容；逐件 SHA256 相同）。
> 用途：补上 `docs/修复清单-20261019.md` §1.4 提到的 `agg_gateway.py` 与 8 GB 档相关脚本，使修复项可从仓内复核/复现。
> 每行只写脚本自带 usage/argparse 里真实存在的参数（不编 flag）。

| 文件 | 干什么 | 谁用 | 需引擎在跑？ | 入口命令（取自脚本 argparse） |
|---|---|---|---|---|
| `agg_gateway.py` | 超池题面的分片/聚合网关：按整结构单元（空行分段）打包、按池容量定片数、`--max-shard-tokens` 封顶；单个结构单元 > 池时走 `unit_broken` 拒答（`sum_method="unit_broken"`、`low_confidence=true`），不回落模型 reduce。附 `--selftest`（`partition_exact`、`unit_broken` 等离线自检） | 修复 1.4 的复核者；超池题面验收 | **是**：`--selftest` 除外（其余要 POST 到 `--base-url` 的 `/v1/chat/completions`） | `python tools/agg_gateway.py --prompt-file <材料文件> --question <问题> --pool-tokens <池 token 数> [--max-shard-tokens 32768] [--base-url http://127.0.0.1:8080] [--model default]`；离线自检：`python tools/agg_gateway.py --selftest` |
| `kv_plan.py` | 27B ternary 的 KV 池规划（纯算术）：按 `--vram-gib` / `--quant` / `--context` 算池页数与 engine MiB；带 `PAGE_BYTES` 表（`k8v4` 1,646,592 B / `nvfp4` 1,179,648 B / `rk2v4-e8` 884,736 B）与 `--selftest` | 8 GB/小显存档配参者；`--host-kv-mib` 下界核算 | **否**：纯算术（stdlib only） | `python tools/kv_plan.py --vram-gib 16 --quant ptq1 --context 262144 [--host-kv-mib 2816] [--desktop-mib 300] [--reserve-mib 500]`；自检：`python tools/kv_plan.py --selftest` |
| `accept_8g.py` | 8 GiB 档端到端验收：6 个 streaming case 走聊天网关，用引擎日志、网关日志、GPU 采样与主机空闲内存判 GREEN/RED；退出码 0=GREEN、1=RED，不重试、不改参数 | 8 GB 档回归/交付验收 | **是**：需要网关与引擎在跑（默认 `--gateway http://127.0.0.1:8081`） | `python tools/accept_8g.py --gateway http://127.0.0.1:8081 --engine-log <引擎日志> --gateway-log <网关日志> --filler <filler.txt> [--pool 15616] [--engine-max-mib 7442] [--host-min-gib 20]` |

> 说明：`accept_8g.py` 的默认日志/filler 路径是开发机路径（`E:\infer-build\kv-stream\…`），换机时必须用上面的参数覆盖。
> 三件都是 stdin/stdout CLI，没有新增依赖（Python 标准库）。

---

# `tools/` · 2026-10-26 补入的三件：Bonsai 2 27B 的 PTQ1_0 转换链

> 来源：本机实际跑通的一次转换整理而来（文本塔取 llama.cpp 的 ternary GGUF，Vision 取同一发布方的 `mmproj-*.gguf`）。
> 为什么需要：引擎自带的 `bonsai2_27b_ternary` recipe 只吃 PrismML 的 **PQ2_0** GGUF（`qwen35.block_count = 64`），
> 且 Vision 取自 Qwen3.8-27B safetensors checkpoint；社区手上的文件是 **PTQ1_0**（ggml type 143），
> MTP 合并版还多一个 `blk.64`，Vision 只有 mmproj。三件脚本补的就是这两处差距，**不需要下载 checkpoint**。
> 三件都是标准库实现（转换那支要求 `--python` 指向装了 CUDA torch 的解释器，因为它要在 GPU 上量化）。

| 文件 | 干什么 | 谁用 | 需引擎在跑？ | 入口命令（取自脚本 argparse） |
|---|---|---|---|---|
| `make_hf_config_bonsai27b.py` | 从两个 GGUF 的 metadata 反推 **config-only 模型目录**：`config.json`、`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`、`generation_config.json`、`preprocessor_config.json`、`video_preprocessor_config.json`。`SafetensorsSource` 明确接受「无权重目录」，所以 `--model` 只要这一份 | 自己拼 `python -m tools.convert` 的人 | **否**（只读 GGUF 元数据并写 JSON） | `python tools/make_hf_config_bonsai27b.py --text-gguf <PTQ1_0>.gguf --mmproj-gguf <mmproj>.gguf --out <目录>` |
| `recipe_bonsai27b_ptq1.py` | 转换 recipe：放行 PTQ1_0 与 `blk.64`；Vision 走 `qwen35_vision_name_map` 从 mmproj 取；`blk.64` 的 MTP 头按存储的 ggml 块型（Q6_K/Q8_0）原样搬入（`--components text,vision,mtp`），走 `import_encoded`、不重新量化；并补上 `pipeline.convert` 漏传的 artifact `identity` | 用 `tools.convert` 转这条模型线的人 | **否**（由转换器调用，不能单独运行） | 作为参数：`--recipe <此文件>:bonsai2_27b_ternary`（引擎源码根需在 `PYTHONPATH` 上） |
| `convert_bonsai27b_ptq1.py` | 一键驱动：校验引擎源码树、探测 `--python` 的 torch/CUDA、调上一支生成 config、再调 `tools.convert` 并流式打印进度 | 想一条命令出制品 / 复现实验的人 | **否**（要引擎**源码树**，不要服务在跑） | `python tools/convert_bonsai27b_ptq1.py --engine-src src-tree/fusion-engine-src --text-gguf <PTQ1_0>.gguf --mmproj-gguf <mmproj>.gguf --out <产物>.ninfer --python <装了torch的python>`（另有 `--reuse-config`、`--dry-run`、`--components`、`--name`） |

> 用 **BF16** mmproj，别用 Q8_0：artifact 的 Vision 存 Q4/Q5/Q6/Q8，Q8_0 源会被解码后再量化一次。
> 三个「写错就启动失败」的细节已写在脚本注释里：added token 必须与 `model.vocab` 不相交；
> `<think>` / `</think>` 必须是 `special: false`（否则引擎报 "cannot present the canonical thinking control suffix"）；
> `pre_tokenizer` 的 ByteLevel 旗标与 `model` 的不变量不能动。
> 实测（12 GB 显卡）：产物 7.15 GB、`--components text,vision`；引擎起来后
> `capacity | KV 131,072 tokens, rk4v4-e8 | runtime 2.38 GiB`，66,528 token 题面问中段事实 39 s 答对，
> 同一会话第二轮 `cached 66,523`（prefill 208 ms）、视觉描述正确。
> MTP 头已接线（2026-10-26）：`--components text,vision,mtp` 会把 `blk.64` 的 15 个参数原样搬进制品
> （`mtp/input_projection` + 一层 full-attention + 三条 nextn 归一化），约 +335 MiB；`blk.64.nextn.embed_tokens.weight`
> （Q4_K，715 MB）**不写入** —— 引擎的 MTP 头固定复用文本 token 表（`bind_mtp`）。块号由
> `gguf_blocks.MTP_BLOCK` 推导，不写死 `blk.64`；`--source mtp=` **会被拒绝**（MTP 头只从
> `--source ternary=` 的那份 GGUF 读；若传入的本身是 PTQ1_0 GGUF，还会更早在 gguf-py 解析阶段报
> `143 is not a valid GGMLQuantizationType`），`--components` 缺省时按 GGUF 是否含该头自动选择。
> artifact 的 `weights_id` 已把 component 列表算进去：同一份 GGUF 产出的 text+vision 与
> text+vision+mtp 不再共用 identity。
> 制品多了 `mtp` 组件后，`ninfer-serve` 要加 `--spec mtp` 才启用量子投机解码。
> 实测（12 GB 显卡、temp 0、输出 400 token、93 token 题面）：`--spec none` 50.0 tok/s；
> `--draft-tokens 1` 70.4（接受率 57.7%）、**`2` 91.1（61.8%，最佳）**、`3` 72.8（31.4%）、`4` 68.3；
> `--mtp-attention-window 64` 反而掉到 75.2。MTP 另占 ~0.14 GiB runtime（131,072 池：有 vision 时只剩 300 MiB）。

