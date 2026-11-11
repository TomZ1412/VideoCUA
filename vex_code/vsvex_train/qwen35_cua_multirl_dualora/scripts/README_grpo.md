# SFT / GRPO 训练

`bash scripts/train.sh --train_type sft ...` 保持原有 SFT 行为；默认也是 SFT。
`--train_type grpo` 使用单轮 vLLM generate、组内归一化 advantage 和 fused GRPO loss。
Trainer 位于 `qwenvl/train/trainer.py` 与 `grpo_trainer.py`，训练入口只负责准备和选择。

## 参数与一次更新

- `--num_generations N`：每个 prompt 的 completion 数，默认 8。
- `--grpo_response_batch_size K`：一次 forward/backward 共享同一 prompt 的 response 数，
  默认 1（原逐条路径），必须为正整数。8 条 response 配 3 时处理为 **3、3、2**；
  不要求整除，也不跨 prompt/生成组拼批。设为 2 或 4 可分别降到每组 4 或 2 次。
- `--bs B --accum_steps A`：每卡每次 optimizer update 的名义 rollout 预算为 `B*A`。
  在一次 generate 中生成这些 rollout，然后按 `grpo_response_batch_size` 分批 forward/backward。
  Trainer 外层 accumulation 固定为 1；不会再次乘以 A。
- 每节点 L 卡时，prompt 数为 `L*B*A/N`，要求能够整除。
- `--tp_size T`（同 `--vllm_tensor_parallel_size`）：生成的 tensor parallel 大小，须整除 L。
  一个节点有 `L/T` 个生成实例。TP 组内使用相同 prompt 顺序和 seed，输出只收集一次。
  生成完成并完成 dynamic sampling 筛选后，在节点内按完整 prompt 组分配；
  同一视频的 N 条 completion 始终归一张训练卡，不再拆到多卡。
- `--max_output_token` 默认 16384。`--vllm_max_model_len` 是 prompt + completion 的上下文容量，默认 155000。
- `--vllm_gpu_memory_utilization` 默认 0.45；它必须覆盖生成权重、KV cache 和多模态峰值。
  大模型可加大 TP 或调整该值。`--vllm_enforce_eager` 可关闭生成端图编译。
- `--shard_model` 使用 FSDP2 `reshard_after_forward=true`；35B-A3B 测试使用此配置。

### 训练阶段共享视频前缀

K>1 时，模型内的序列布局为一份 prompt 加 K 个独立 response。视频/音频编码器、
每层前缀投影和 MLP 只处理一次 prompt。全注意力通过 FlashAttention 的因果 Q/KV
长度偏移共享前缀；线性注意力通过 FLA 的可微 final/initial state 和卷积尾状态分支。
每条 response 沿用同样的多模态 RoPE 起点，不能看到其他 response。
前缀梯度会从所有分支回传，不 detach，不跨 optimizer step 缓存。
状态只活在单个 decoder 调用中，activation checkpointing 重算时重新创建。
FSDP2 的 decoder 包装、reshard、LoRA、FLA、FlashAttention 3 和 Liger fused loss 均保留。

这会将每组 8 次前缀计算降为 ceil(8/K) 次，不能保证整体训练同倍提速：
每条 response 对 prompt 的注意力仍需要计算；当前全注意力为各分支拼接 K/V，
K 越大，response 激活和临时 K/V 越多。response 右侧补齐到该批最长长度，
loss 只统计实际 token；长度为 1 的 response 也支持。显存不足时降低 K。
共享路径要求 FlashAttention 2/3。K=1 保持原计算路径，便于基准对照。

各 rank 在训练前按实际 **microbatch 数** 对齐，空闲卡用单条零 advantage 补齐，
不把有效尾批强行补到 K 条。DAPO 仍按全局有效 token 数、GRPO 仍按全局有效
completion 数归一化，动态采样丢弃项与对齐批不计入分母。

2026-10-02 验证：4 个 CPU batching 测试、单卡音视频数值/分支隔离测试，以及
2/8 卡 FSDP2 测试通过。8 卡 BF16 梯度对照的整体相对误差约 1.1%，存在 kernel
分块与求和顺序引入的浮点差异，不保证逐 bit 相同。
使用 9B audioalign checkpoint-7630 + checkpoint-7986 LoRA 和 CinePile 数据，
在原命令上仅追加 `--grpo_response_batch_size 3`，已完成两次真实 optimizer 更新。
保留 1024 帧、176400 pixels、8192 输出上限、TP=1、num_generations=8、bs=8、
dynamic sampling、FSDP2 reshard、activation checkpointing、FLA、FA3 和 Liger。
两步平均 prompt 约 83.5K / 83.6K，loss 为 0.3122 / 0.3874，梯度范数为
0.1112 / 0.1411；有效 rollout 为 120 / 64，无 OOM。
完整验证日志位于 `output/grpo_shared_prefix_validation/`。

GRPO 始终使用模型原生 `<think>... </think>` 格式。JSON 中缺省/true/false 和
命令行 `--thinking false` 都不会关闭 GRPO thinking。SFT 不传 `--thinking` 时仍遵循
每条样本的字段；显式传参可统一设置。GRPO 移除参考答案后构造 prompt，不做多轮定位或工具调用。

## Reward 与 dynamic sampling

仅保留三个 reward，`--reward_weights` 顺序为 length,format,accuracy，默认 `0.05,1,1`：

| reward | 定义 |
|---|---|
| length | 长度不超过 `max_output_token / 2` 时为 0；超过后为 `-(completion_tokens - max_output_token / 2) / (max_output_token / 2)`，下限 -1 |
| format | 一次 `</think>`，非空思考与非空最终答案，没有额外 thinking/tool 标签 |
| accuracy | 比较 `</think>` 后答案与 reference；选项字母支持 `E`、`E.` 及其后解释，其余答案使用规范化后的精确匹配 |

模型模板已经包含 `<think>` 开头，不要求 completion 再输出一次。
组内使用总体标准差归一化，epsilon 为 1e-4；全相同 reward 的 advantage 为 0。
每个 rollout 只用于当前一次更新，beta=0，不创建额外 reference model。
vLLM sampling logprob 与训练策略 logprob 分开：PPO clipping 使用 detached 当前训练策略，
vLLM 差异使用 token importance correction，最大值由 `--vllm_importance_sampling_cap` 控制（默认 2）。
词表投影按 token 分块，避免为整个 16K completion 分配完整词表 logits。

`--loss_type dapo|grpo` 控制 loss 的归一化方式，默认 `dapo`：按本次 optimizer
更新中所有 rank、所有累积 rollout 的有效 completion token 总数归一化。
动态采样丢弃的 rollout 和为对齐 forward/backward 次数添加的 padding 不计入分母。
原有训练命令无需修改即可使用 DAPO；追加 `--loss_type grpo` 恢复原来的
每条 completion 内先按 token 平均、再对全局有效 completion 平均的行为。
该参数仅切换 loss 归一化，dynamic sampling、epsilon 和 reward 等仍由各自参数控制。

`--importance_sampling_level token|sequence` 控制策略 loss 的比率粒度，默认 `token`。
`token` 使用每个 token 的 `exp(logp - old_logp)`；`sequence` 使用整条 completion
有效 token 的 `exp(mean(logp - old_logp))`（GSPO 式比率）。例如在训练命令中追加
`--importance_sampling_level sequence`。这不改变独立的 vLLM token importance correction。
当前每批 rollout 只更新一次，old policy 是当前策略的 detach，故两种比率的前向值
均为 1；sequence 模式通过整条 completion 的平均 correction 缩放各 token 梯度，
而 token 模式使用各 token 自己的 correction。没有 correction 差异时，两种模式的
loss 和梯度一致。实现保留 token 分块，以该单次更新条件下等价的一阶梯度计算
sequence loss，不把分块后的单个 token 误当成完整序列。

`--dynamic_sampling` 按上一步全局接受率自适应增加候选 prompt。设每节点目标组数为
`B = local_size * bs * accum_steps / num_generations`，接受率为 `r`，补采系数
`f = --dynamic_sampling_oversample_factor`（默认 1.2），使用参考仓库的缺额估算：
`候选数 = B + ceil((B/r - B) * f)`。第一步使用
`--dynamic_sampling_initial_accept_rate`（默认 0.5）。例如 B=16 时，r=0.5 采36组，
r=0.4 采45组，r=1 回到16组。全局接受率在本步生成结果汇总后更新所有 rank 的 sampler。
下一步计划在本步生成前先用旧接受率预测；新接受率出来后修订数量。扩容保留原候选
前缀及其节点/TP leader 归属，已完成和正在进行的媒体读取可复用；缩容清掉多余缓存。

每节点候选数最多为 `floor(B * --dynamic_sampling_max_factor)`，该系数默认 8，
接受率为零时直接使用这个上限。按长视频内存容量可调低这个上限；上限只限制采样量，
不保证一定达到目标有效 batch。每步仍只有一轮真实生成，不会循环补采到满。
epoch/调度步数沿用基础 prompt 计划，补充候选从同模态、邻近长度的数据中选取，
不同 step 的补充候选可能重复；同一步候选池足够时不额外重复，池太小时补齐。

沿用参考仓库的 accuracy 筛选：丢弃全对或全错的组，保留部分正确组的完整 N 条输出。
长度/格式 reward 仍参与 advantage，但不会使全错组因为长度不同而被接纳。
所以实际更新 rollout 数可以变化。分发后**在任何训练 forward 之前**全局通信，
确定各卡统一的 forward/backward 次数。短缺但非空的卡复用自己的最后一个视频，以零 advantage 补齐；
完全空闲的卡使用本节点最短 prompt 的一个零 advantage completion，以保持 FSDP 模态分支一致；
loss 按全局有效 rollout 总数归一化，补齐项不参与分母。全局无有效样本时清空梯度，
Adam momentum / weight decay 不更新参数；这次候选批仍消耗一个调度 step。
日志包括三个 reward、有效 rollout 总数、接受率、每卡 forward/backward 次数。
dynamic sampling 还记录本步候选数 `sampled_prompts_per_node`、目标数
`target_prompts_per_node` 和接受率更新后的下一步候选数 `next_prompts_per_node`。

## 长度缓存

`--length_cache` 复用 `scripts/cache_token_lengths.py` 的 version 2 video_token_map。
GRPO 按 **prompt** 分组，再按节点分配，不能按最终 rollout 或普通 SFT rank batch 来排。
例如 64 卡、每节点 8 卡、B=A=1、N=8 时，每一步先从长度相近的 8 个 prompt 中
给每节点一个，节点生成的 8 条全部归一张卡，全球共 64 条有效 rollout。
其余卡需要各做 8 次零权重 forward/backward，建议让每节点有效 prompt 数至少达到卡数，
以减少空闲卡补齐开销。名义 rollout 预算和有效 loss 分母不变。
B/A 变大时每节点增加 prompt 数。`--length_grouping_window` 控制候选排序窗口。
没有 cache 时仍按相同拓扑调度，只是不具备视频长度信息；缓存缺项沿用已有平均长度回退。
每个 global step 先保证音视频分支相同，再在该模态内按长度分组；避免 FSDP 对不同模块发起不匹配的 collectives。每个 epoch 使用一致 seed 排序，尾批补齐。这个 dataloader 不再交给 Accelerate 二次切片。

训练分配使用生成后实际 `prompt_length * completion 数` 估计负载，其中 prompt_length
包含展开后的视频 token，也包含文本/音频 token；不再按单条 completion 长度拆组。
先尽量平衡各卡有效 completion 数，再平衡 prompt token 总量；原生成卡的预计负载
在最小可选负载的 110% 内时优先原地保留。这是减少移动的启发式，不保证全局最优。
重复 dataset index 的候选整批绑定到同一卡，避免尾批/补采重复再次复制视频；
因此重复候选较多时，各卡有效条数差异可能大于 N。组内连续训练，媒体只准备一次。

生成前按长度选候选仍能减少极端差异，但 dynamic sampling 改变各节点的保留集合，
不能保证逐次 forward/backward 长度对齐。新的均衡发生在筛选之后，以完整组为单位。
例如 L=N=8、每节点 64 个不同有效视频时，各卡约处理 8 个视频的 64 条输出，
训练媒体的跨卡重复份数可由 8 降到 1。这不意味着整体峰值 CPU/GPU 内存一定降八倍：
TP 生成输入、下一步候选预读、模型/优化器、激活和空闲卡 padding 仍有各自开销。
生成阶段媒体在生成返回后立即释放，避免一直持有到训练分配完成。

所有视频读取、解码和预处理都由每 rank 一个持久的 8 线程后台 worker 执行，不在训练
主线程调用 `prepare()`。生成阶段只在 TP leader 准备媒体；训练阶段所有 rank 使用同一个
后台 worker，最多并行预读 8 个 prompt，连续同 prompt 复用数据。

跨 step 流水线：本步开始生成前，按旧接受率安排下一步基础及预计补采候选；生成结束
后按新接受率追加或缩减下一步候选，后台读取与本步训练重叠。首次启动仍需等待首批
准备，缓存未读完时消费端也会等待。当前训练输入的读取优先于下一步的预测性读取。

预读不设条数或字节上限，后台持续读取，直到下一轮已确定的候选全部准备好。
计划缩减时释放不再需要的缓存；不会继续预读尚未安排的更远轮次。
训练结束或异常时回收全部线程，不另起进程复制整份数据集。外层 DataLoader 只传计划键，
仍是 `num_workers=0`；`dataloader_num_workers` 不控制实际媒体 worker。

## 生成时 CPU offload

大数据通信使用 NCCL：TP 输入中的 CPU 视频/音频张量按 64 MiB 分块暂存到 GPU 后
广播，接收端还原到 CPU；不把整批媒体同时放上 GPU。各 TP leader 本地计算 reward 和筛选，
只广播组索引、来源、条数和 prompt 长度等规划元数据。保留组的 IDs/logprobs/advantage
按计划通过 NCCL 点对点、64 MiB 分块传到唯一目标卡；原地组不通信，丢弃组不广播。
媒体张量不参与训练 rollout 交换，目标卡通过后台 worker 读取自己的视频。
本仓库 GRPO 显式创建的通信组统一使用 NCCL，包括输入元数据、TP 控制信号和同步等待。
CPU 媒体张量经 CUDA 分块中转；vLLM 依赖内部仍可能创建自己的 Gloo CPU 组。
日志分别输出媒体准备、prompt 广播、实际生成和 rollout 交换耗时，便于定位大 batch 的等待。

`--grpo_offload_model true`（默认）按如下顺序执行：

1. FSDP2 模型 `.to('cpu')`，清理 CUDA cache。
2. 所有训练 rank 通过 NCCL barrier后唤醒 vLLM 权重，再通过一次 NCCL barrier。
   每次只把一个参数以 `non_blocking=True` 恢复到 CUDA 并通过 DTensor `full_tensor()` 收集，送给 vLLM。
   LoRA 也逐层收集 A/B 并合并到推理权重，不在训练模型上执行 merge/unmerge。
   所有 rank 完成 CUDA 权重加载并通过 NCCL barrier后，才进入生成阶段。
3. vLLM generate，然后 sleep(level=2) 释放推理权重与 KV cache。
4. FSDP2 模型移回原 CUDA device，forward/backward/optimizer step。

这是参数 offload，optimizer 状态仍留在 GPU。传 `--grpo_offload_model false` 可保持
训练模型在 GPU，只清理 cache。

`--sync_vllm true`（默认）恢复参考仓库的生成等待保活：提前完成真实生成的 TP 实例，
在独立 NCCL 通信组的异步 barrier 等待期间重复执行 24K token 的文本 dummy prefill，
每次只生成 1 token；上下文容量不足 24001 时，dummy 长度降为 `max_model_len - 1`。
等待从单独的 CUDA stream 发起，结束前不让生成 stream 等待它。
TP > 1 时由 leader 通过 CUDA 控制张量广播继续/退出决定，保证组内 generate 调用一致；
TP = 1 时直接轮询完成状态。
所有实例完成真实生成后，各组结束当前 dummy 并退出，随后 sleep、恢复训练模型。
真实生成与评测端一样开启 `enable_prefix_caching=True`，使同一 prompt 的
`num_generations` 个请求能够复用已经计算的公共前缀。缓存按 block 命中；首批并发、
未对齐的尾部及缓存淘汰仍可能导致部分重复 prefill，不保证严格只计算一次。
每次同步策略权重后清空 prefix cache，避免复用旧权重的 KV/SSM 状态。
dummy 输出不参与 reward、rollout 统计或训练。每轮 dummy 使用唯一 `cache_salt`，
从第一个 block 起隔离缓存，确保每轮都重新完整 prefill；计数跨等待调用保留，
TP 组内一致，且不消耗训练/采样 RNG。这个机制用于填充等待时间，不提高
有效样本吞吐，退出时可能多等待当前 dummy 完成。

开启时，没有真实 prompt 的 TP 实例也会加载当前权重并参与 dummy 生成。
传 `--sync_vllm false` 可关闭此行为；关闭时空实例保持睡眠，但其训练 rank
仍参加参数收集与训练通信。此机制依赖当前独立 ExternalDP 布局，不适用于
启用跨实例 vLLM 原生 DP/EP 的配置。

参考实现：TRL main commit `a7c34f363a8716473a0f15378621a3358b994417` 的
[`vllm_generation.py`](https://github.com/huggingface/trl/blob/a7c34f363a8716473a0f15378621a3358b994417/trl/generation/vllm_generation.py)
和 [`grpo_trainer.py`](https://github.com/huggingface/trl/blob/a7c34f363a8716473a0f15378621a3358b994417/trl/trainer/grpo_trainer.py)。
这里复用本仓库自定义音视频 processor/模型，而非直接套用 TRL 的标准图片数据入口。

### 权重同步边界

CPU 参数以 `non_blocking=True` 恢复到 CUDA，避免逐参数拷贝主动阻塞主线程。
FSDP2 的 `full_tensor()` 仍要求所有训练 rank 参与，与 vLLM TP 大小无关。
媒体准备、vLLM 权重唤醒和权重加载完成的边界通过 NCCL barrier 对齐；
空闲推理 rank 也参与。通信组超时沿用 `ddp_timeout`（秒）。

## 35B 模型的历史验证命令

在每台 worker 上执行同一命令（平台设置 ARNOLD 多机变量）。LoRA checkpoint 会先合并为
本次运行的 immutable base，然后 `--use_lora` 创建新的训练 adapter。

```bash
bash scripts/train.sh \
  --train_type grpo --shard_model --use_lora \
  --model /path/to/checkpoint \
  --model_base /path/to/checkpoint \
  --lora_ckpt /path/to/checkpoint \
  --dataset /path/to/grpo_CinePile_gemini35_washed.json \
  --max_frames 1024 --max_pixels 176400 --interval 0.1 \
  --max_output_token 16384 --tp_size 8 \
  --num_generations 8 --bs 1 --accum_steps 1 \
  --vllm_gpu_memory_utilization 0.8 --vllm_enforce_eager \
  --dynamic_sampling --max_steps 2 --save_steps 2 --warmup_ratio 0 \
  --run_name grpo_35b_validation --report_to none --no_hdfs_upload
```

以下是整组分配改动前的历史验证记录，不能用来证明当前分配实现的端到端结果。
上述命令曾在单机 8 张 H100 80GB 上完成两步真实训练，进程退出码为 0。
使用指定的模型、base 与 checkpoint-7986 LoRA，TP=8、FSDP2 reshard、CPU offload、
dynamic sampling，以及上述完整视频与输出上限设置。两步均接纳 8/16 条 rollout，
每卡执行一次 forward/backward；梯度范数分别为 0.3852、0.4089，loss 均有限。
实际平均 prompt 长度约 83.5K / 85.3K，平均 completion 长度约 1325 / 818；
这些是本次样本的实际长度，不表示每条样本都达到了配置上限。

`output/grpo_35b_validation/checkpoint-2` 和最终目录都已保存 380 个 LoRA 张量，
全部有限，190 个 LoRA B 张量全部非零；两份保存文件的 SHA256 一致。
日志见 `output/grpo_validation/real_35b_final.log`，张量校验见同目录
`checkpoint_verification.json`。环境为 PyTorch 2.11.0、Transformers 5.17.0、
vLLM 0.26.0、Accelerate 1.15.0、Liger 0.8.3、PEFT 0.21.0。
退出时有未显式调用 `destroy_process_group` 的 NCCL 警告，所有进程已退出。

另已通过两步 SFT 回归、小模型 TP=1/2 与累积梯度测试、空接受批次测试，以及
FSDP2 CPU/GPU 往返和不均匀 rollout 的三步梯度/AdamW 参考对照。
64 卡的长度分组仅做了调度模拟，尚未做实际多机 64 卡训练。复现方式见
[`tests/README.md`](../tests/README.md)。

## 整组分配回归验证

```bash
python -m unittest discover -s tests -p 'test_grpo_group*.py'
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 tests/grpo_group_distributed.py
```

CPU 测试覆盖整组唯一归属、重复 index、长度负载、原地保留和 1/2/8/64 rank 调度。
GPU 测试使用合成 generation，运行真实 NCCL 分块传输、trainer 筛选/分配/补齐、
后台媒体读取和小模型 FSDP2 backward，并对照全局有效样本梯度；包含 TP=1/2/8、
全空批次、少于卡数的组，以及单机模拟两个节点（含整节点零接受）。
它不替代真实 vLLM 长视频训练或多机网络验证。

本次改动已通过上述 4 个 CPU 测试和 6 种 8 卡 GPU 场景，分布式脚本退出码为 0。
当前环境显式销毁 NCCL 通信组会等待不退出，在不含 GRPO 的两卡 FSDP Linear
基线中也可复现；回归脚本沿用生产 trainer 的解释器回收方式，退出时有 NCCL
未显式 destroy 的警告。尚未重新执行真实 vLLM 长视频训练或实测峰值内存。
