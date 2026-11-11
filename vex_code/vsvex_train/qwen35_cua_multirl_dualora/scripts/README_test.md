# 多机多实例评测

入口：`bash scripts/test.sh ...`。默认每台机器 8 张 GPU，启动 **8 / tp_size** 个独立 vLLM OpenAI 服务。
`tp_size=1` 为 8 个实例，`tp_size=2` 为 4 个实例；各实例的 GPU 不重叠。

## Arnold 启动

与 `scripts/train.sh` 一样，让平台在**每个 worker 上执行相同命令**：

```bash
bash scripts/test.sh \
  --model ./output/qwen35_35ba3b_debug \
  --input /path/to/test.json \
  --run_name qwen35_moe_eval \
  --tp_size 2 \
  --max_frames 1024 --max_pixels 176400 --interval 0.1 \
  --max_model_len 155000 --max_tokens 2048 \
  --concurrency 4 \
  --no_think
```

自动读取 `ARNOLD_WORKER_NUM`（机器数）和 `ARNOLD_ID`（机器 rank）。
平台负责在各节点启动脚本；各节点独立读取输入、切片并管理本节点实例，无需节点间通信或等待。
所有机器都需有本仓库和当前推理依赖，并能访问内容及顺序一致的完整输入 JSON、模型和媒体路径。
各机器使用相同评测参数、相同机器数和不同的 rank；脚本不跨机器校验参数。

非 Arnold 环境，在每台机器分别执行相同命令，加上：

```bash
--nnodes 4 --node_rank 0
```

其他机器把 `node_rank` 改为 1、2、3。各机器可独立启动和重试，模型 TP 通信只在各自机器内发生。

1. 每台机器独立读取完整输入 JSON，按原顺序均分为 N 份，份数之间最多相差一条。
2. 取本机 rank 对应的连续切片，保存到该机器的 `rank{i}/input.json`。
3. 每台机器准备模型，必要时先在 CPU 上合并一次，再启动各个 GPU 实例。
4. 遍历本机切片，按视频分组，优先将大组分给当前待处理样本数最少的实例；每个实例按 `--concurrency` 并发逐条调用 OpenAI API。空闲请求槽可从其他实例接手尚未发送的任务。
5. 按原始样本顺序写入一个 JSON，并上传本机器的结果文件到 HDFS。

空分片也输出并上传 `[]`，但不会加载模型。单机默认 `nnodes=1`，处理全部输入。

## 模型和 LoRA 合并

完整 checkpoint 只传 `--model`。`--input`、`--model`、`--model_base` 和
`--lora_ckpt` 均支持直接传入 `hdfs://...` 路径；JSON 中的 `video` 也支持 HDFS 文件。
每台机器独立下载完整 JSON 后取本机切片，在启动实例前下载模型及可选的 base/LoRA，
本机所有实例共用下载缓存。视频在调用对应样本时下载。
缓存目录由 `--cache_dir` 指定（默认 `output/test/.cache`），已完成的同一 URI 下载会复用。
本地模型必须是已有目录。

例如，在各 worker 执行（将路径替换为实际 HDFS 地址）：

```bash
bash scripts/test.sh \
  --input /path/to/checkpoint \
  --model /path/to/checkpoint \
  --cache_dir /tmp/qwen35_eval_cache \
  --run_name qwen35_hdfs_eval --tp_size 2 \
  --max_frames 1024 --max_pixels 176400 --max_tokens 2048 --no_think
```

需要合并 LoRA 时追加 `--lora_ckpt hdfs://.../checkpoint`，必要时再传
`--model_base hdfs://.../base`；各参数的模型角色见下文。

显式指定基础模型、可选的 tokenizer/processor 来源和 LoRA checkpoint：

```bash
bash scripts/test.sh \
  --model /path/to/full_audio_base \
  --model_base /path/to/processor_fallback \
  --lora_ckpt /path/to/checkpoint-3000 \
  --input /path/to/test.json --run_name qwen35_lora_eval \
  --tp_size 2 --max_frames 1024 --max_pixels 176400 --no_think
```

这里沿用 `scripts/merge_base_model.py` 的语义：`model` 提供完整底座权重，`model_base`
提供缺失的 tokenizer/processor 文件，`lora_ckpt` 提供 LoRA 和音频/Q-Former 等训练权重。

也支持直接把 adapter checkpoint 传给 `--model`：

```bash
--model /path/to/checkpoint-3000 --model_base /path/to/full_audio_base
```

此时 `model_base` 作为完整底座；没传时读取 `adapter_config.json` 中的 `base_model_name_or_path`。
模型合并日志输出到 stdout；默认合并到 `rank{i}/merged_model`，可用 `--merge_dir`
指定其他位置。使用文件锁及完整性标记复用已经完成的相同合并，输入变化则拒绝复用。
HDFS 下载缓存按 URI 区分，适用于不可变的 checkpoint 路径。

## Thinking 与生成参数

**仅由命令行决定，忽略输入 JSON 中的 `thinking` 字段。**

- `--think` / `--think true`：开启。
- `--no_think` / `--think false`：关闭；默认关闭。
- 服务使用 `--reasoning-parser qwen3`，将思考过程与最终答案分开保存。
- `max_tokens` 包含 reasoning 和最终回答。开启 thinking 时根据任务增加，例如 `--max_tokens 8192`。
  若仅返回 reasoning 而没有最终回答，该条记录标记为失败，同时保留 reasoning。达到长度上限时可增加 max_tokens；若已经返回 stop，则应检查 checkpoint 是否训练了 thinking 格式。

默认生成采样与示例 `example_call.py` 一致：

| 模式 | temperature | top_p | top_k | presence_penalty |
|---|---:|---:|---:|---:|
| 不 think | 0.7 | 0.8 | 20 | 1.5 |
| think | 1.0 | 0.95 | 20 | 1.5 |

可用 `--temperature`、`--top_p`、`--top_k`、`--presence_penalty` 覆盖。
确定性解码可传 `--temperature 0 --top_p 1 --top_k -1 --presence_penalty 0`。
默认 `--max_num_batched_tokens 16384`，对照不同入口时保持此参数一致。

## 输入与结果

输入为 JSON 数组，沿用现有 `video` / `tos_key`、`conversations` 格式。
与参考调用脚本一致，使用第一个 human/user 提问，第一个 gpt/assistant 内容作为 ref。
发送请求时移除文本中的 `<image>` / `<video>`，媒体以单独 `video_url` 传入；输出中的 prompt
保持原始对象。每条样本对应一个视频；多轮对话和多视频样本不在本脚本的评测格式范围内。

- `video`：支持本地文件、HTTP(S)、file://、HDFS 路径；本地媒体通过 file:// 由本机服务读取。
- 只有 `tos_key` 时，复用仓库已有 TOS client 下载到临时文件，调用完成后删除临时文件。
- 本地 JSON 的相对视频路径按该 JSON 所在目录解析；HDFS JSON 中应使用绝对媒体路径。
- 默认读取音轨；样本 `use_audio: false` 可禁用该条音频，`--no_audio` 可禁用全部音频。
- `--max_frames` 和 `--max_pixels` 沿用 Dataset 语义：帧数上限与总像素预算参数。
  它们不代表强制重复到固定帧数，也不代表严格的单帧像素限制。

输出示例：

```json
[
  {
    "tos_key": null,
    "video": "/path/to/9.mp4",
    "prompt": {"from": "human", "value": "<image>\nxxx"},
    "ref": "B",
    "pred": "B.",
    "reasoning": "..."
  }
]
```

`pred` 是最终回答原文，不做选项字母提取。没有 reasoning 时使用空字符串。
连续请求失败时仍保留该样本，额外写入 `error`，`pred` 为空；脚本上传结果后以非零状态退出。

本地文件默认保存到：

```text
output/test/<run_name>/test_result_rank0.json
output/test/<run_name>/rank0/input.json
output/test/<run_name>/rank0/plan.json
output/test/<run_name>/rank0/progress.jsonl
output/test/<run_name>/rank0/summary.json
```

评测进度、模型合并日志和最终 HDFS 路径输出到 stdout；vLLM 服务及其子进程的 stdout/stderr
统一送到 stderr。Python 警告和异常也保留在 stderr。关闭 Python 输出缓冲。
本机进度每处理 10 条、发生失败或全部完成时输出；多个实例的服务日志会在 stderr 交错显示。
可用 `bash scripts/test.sh ... >eval.log 2>vllm.log` 分别保存两路输出。
每 30 秒及结束时打印各实例的 `pending`（尚未发送）、`active`（正在处理，包括媒体加载及重试）、
`done`（本次已完成请求，含失败）和 `stolen`（从其他实例接手的样本数，可包含再次转移的任务）。
据此可区分任务分配不均与请求本身较慢；`active` 不等于 GPU 上正在执行的序列数。
`progress.jsonl` 仍逐条落盘用于续跑，保留全局索引、执行实例 `replica`、请求耗时 `seconds`、
token 用量、停止原因和错误信息。
最终 JSON 按原始顺序排列。`--output_dir` 修改本地父目录，脚本仍会在其下增加 run_name。

**默认自动上传**到：

```text
/path/to/checkpoint
```

上传成功后，每台机器在结束时向 stdout 打印 `HDFS result directory: hdfs://.../<run_name>/`。
使用现有 `cruise.utilities.hdfs_io`，无需 hdfs CLI。上传失败会保留本地文件并非零退出。
`--hdfs_root` 可修改父目录，`--no_hdfs_upload` 仅保存在本地。

## 续跑与调试

相同命令增加 `--resume`，跳过成功记录，重试失败/未完成记录，并重新上传完整结果。
输入或影响生成的参数变化会拒绝续跑，需改用新的 run_name。已全部完成的 rank 不启动模型。
不传 `--resume` 时，不覆盖同名已运行的评测。

`--dry_run` 只执行本机切片和写入启动计划，不合并模型、不占 GPU、不上传结果。
可先使用它检查机器 rank、GPU 分组及端口。

默认使用 8 张 GPU。小规模验证可设置 `--gpus_per_node 2 --tp_size 1`（两个实例），
或 `CUDA_VISIBLE_DEVICES=2,3,4,5 --gpus_per_node 4 --tp_size 2`（两个 TP=2 实例）。
必须满足 `gpus_per_node % tp_size == 0`。

其他参数：`--base_port`（默认 8000，实例依次使用后续端口）、`--startup_timeout`、
`--request_timeout`、`--retries`、`--gpu_memory_utilization`（默认 0.9）、`--enforce_eager`。
每个实例均为本机独立服务；完成或异常退出时清理其 API/engine/TP 子进程。

## Prefix cache 与视频分组

评测服务默认显式开启 `--enable-prefix-caching`。本机切片中的相同视频（或相同
`tos_key`）优先交给同一个 TP 实例，组内按原顺序入队，尽量复用视频对应的公共前缀。
本地相对路径、绝对路径和本地 file:// URL 会规范化后分组；不同远程 URL 不做内容去重。
各组初始按待处理样本数从大到小分配给当前负载最小的实例。某实例本地队列耗尽后，
从剩余队列最长的实例优先接手完整的未开始视频组；若只剩已开始的组，则接手其中
最大组约一半的未发送请求。已经发送的请求不会迁移或重复提交；每个实例仍最多保留
`--concurrency` 个并发请求。转移视频到新实例可能增加一次媒体处理和缓存预热开销。
这样避免视频长度、生成长度及媒体 I/O 差异导致部分实例提前闲置；最后仅剩在途请求时
仍可能有 GPU 空闲，不能保证整个运行期间 GPU 利用率始终满载。调度仅在本机实例之间进行。
续跑先跳过成功项，再对剩余任务重新分组均衡；兼容旧版静态分组生成的续跑记录。
最终结果仍按原始输入顺序保存。原命令无需新增参数；已运行的进程需重启并加 `--resume`。
生成和采样参数保持原值；调度及批次变化仍可能影响生成文本，不保证逐字复现旧运行。
缓存由各实例独立维护，实际命中取决于公共前缀、媒体内容和缓存容量；并发首批请求可能尚未命中。
AV loader 使用 CPU 预处理缓存；AVProcessor 同时接入 vLLM 的共享内存传输缓存，见下文。

评测默认令每个实例的 `max_num_seqs` 等于 `--concurrency`（默认 4），并传给 vLLM。
例如 `--concurrency 16` 会同时设置 `max_num_seqs=16`；可用 `--max_num_seqs` 单独覆盖。

启动时启用多模态显存 profiling：以接近上下文长度上限的视觉输入及一个音频块
测量编码器峰值和 encoder cache，再由 vLLM 分配 KV cache。日志输出 `AV profiling input`
和实际可用 KV cache 大小。视觉编码保持整段执行。

## 媒体预处理加速

HTTP AV loader 在每个媒体线程内复用 Dataset、tokenizer 和 processors，避免每个问题
重新加载。每个 API 服务进程默认保留最多 **4 GiB** 的 CPU 预处理结果，按 LRU 淘汰；
四个 TP=2 实例的缓存预算合计为 16 GiB，不包括正在解码或请求持有的张量。
缓存键包含视频字节内容的 SHA-256、模型路径、完整媒体采样参数及音频开关。
同一视频的并发首次请求共享一次计算；超过缓存预算的视频只与正在等待的请求共享，
计算失败不会留在缓存中。模型文件在服务生命周期内应保持不变。

这些优化沿用原有 Dataset 的抽帧、缩放、音频处理及 token 构造，不调整 thinking、生成
参数、视频分组或最终 JSON 字段和顺序。媒体仍需由 vLLM 读取/下载到字节后查缓存；
这个缓存不消除所有远程 I/O，也不替代 GPU encoder/KV cache。

AVProcessor 还使用 vLLM 原生的 sender/receiver 缓存协议。默认每个服务分配 **8 GiB**
共享内存缓存，单个对象上限 **2047 MiB**，将媒体张量写入共享内存，TP worker 读取
地址并还原张量。这样避免每条请求向每个 TP worker 重复 pickle/复制数 GiB 的视频
张量。缓存只包含媒体特征，prompt、占位范围、生成参数仍按请求计算；容量不足或对象
过大时沿用 vLLM 的非缓存传输。四实例总预算为 16 GiB loader 缓存加 32 GiB 共享缓存，
`/dev/shm` 需有足够空间，运行中张量和其他开销另计。

`scripts/test.sh` 和服务入口均支持调整 `--mm-processor-cache-gb`、
`--mm-processor-cache-type`（`shm` / `lru`）、`--mm-shm-cache-max-object-size-mb`。
例如 `--mm-processor-cache-gb 0` 关闭 IPC 缓存；`lru` 仅缓存在 API/engine 之间，
不消除 engine 到 TP worker 的大张量传输。

vLLM 0.26 的 SHM ring 使用有符号 32 位整数保存对象长度（包含 header）。原先
4096 MiB 的单对象上限会让约 2 GiB 以上的视频特征进入缓存分配，触发
`400: int too big to convert`。默认上限已降至 2047 MiB；AVProcessor 还会按
ring header 的实际宽度限制对象上限，因此显式传入 4096 也会自动保护。
超过上限的对象由 vLLM 原生非缓存路径传输，不减少帧数或修改特征。
修改后需重启服务；原评测命令加 `--resume` 可跳过成功样本并重试失败样本。

可选环境变量（原命令无需增加参数）：

- `QWENVL_MEDIA_CACHE_GB=4`：每个 API 服务的 CPU 缓存预算；`0` 关闭结果保留，
  仍复用处理器并合并正在进行的同视频预处理。
- `QWENVL_MEDIA_PROFILE=1`：在服务日志中输出 `AV media: cache=miss/hit/shared` 和
  加载器耗时，包含内容哈希与缓存等待，不包含上游文件读取和后续 GPU 推理。

使用本地模型的 tokenizer/processor 文件即可比较原 loader 与优化后 loader，无需加载
模型权重。脚本逐元素验证视觉/音频张量和 token ID 完全相同；可重复传 `--video`：

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 python scripts/benchmark_media.py \
  --model /path/to/model --video /path/to/video.mp4 \
  --interval 0.1 --max-frame 1024 --max-pixel 176400
python -m unittest discover -s tests -p 'test_media_cache.py' -v
python -m unittest discover -s tests -p 'test_av_processor_cache.py' -v
```

`original_seconds` 为原 loader 耗时，`first_seconds` 为优化后的首次处理耗时，
`cached_seconds` 为后续重复调用耗时；文件读取另计在 `read_seconds`。
这只衡量 CPU 预处理，不代表整个评测的加速倍数。

2026-09-30 真实模型验证：使用 `grpo_35ba3b_quickload/base + checkpoint-60`、
两张 H100、TP=2、并发 8、thinking，以及 1024 帧/176400 像素/0.1 秒采样参数，
从原评测集选取两个视频各四条问题。首次请求总耗时由原版 113.87 秒降为仅预处理
缓存版 90.77 秒，再降为加入共享内存后的 52.68 秒（2.16 倍）；平均 GPU 利用率
分别为 27.7%、35.4%、60.9%。三版完整结果对象逐条相同，包括全部 reasoning，
均为 636655 个输入 token 和 12995 个生成 token，8/8 成功。
这不包含下载、合并、服务启动；首次请求可能包含运行时 kernel 预热。
这是一组重复视频的小样本单实例结果，不代表全量数据或四实例吞吐。
另外测得的预热轮生成内容/长度不同，不能直接按耗时计算等输出加速倍数。
