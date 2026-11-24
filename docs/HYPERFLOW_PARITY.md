# 与 HyperFlow H3 Flash 开源实现的对齐情况

对照对象：`Video-Rebirth/hyperflow-h3-flash`（2026-11 版本，仓库根目录
`src/hyperflow_h3_flash/` 与 `integrations/ComfyUI-HyperFlow-Flash/`）。
本文只记录**能力级对齐**，不声称与 Python / ComfyUI 逐像素一致：
本 runtime 的引擎是 `native/engine/` 里的 C/CUDA 快照，权重来自
`ComfyUI_MiniMax-H3-V8` 的转换版，不包含官方 Python 代码。

## 已对齐（可直接互相对照）

| 项目 | 开源实现 | 本 runtime | 备注 |
|---|---|---|---|
| 采样网格 | `schedule.py: SIGMA_GRIDS`，4/6/8 步，默认 6 | `h3_host.c: h3_hyperflow_schedule_build()`，同一组 knot | 6 步时改为读取 checkpoint 内 `hyperflow_sigmas` 张量的实际值（转换版存的是 shift 后的 video 网格），与开源网格一致 |
| shift 公式 | `shift_sigmas`: `s·σ/(1+(s−1)·σ)`，video 12 / audio 3 | `h3_time_shift_sigma`，常量在 `h3_host.h` | audio 网格在读取 checkpoint 网格时由同一个 base 反推后 shift 3 |
| 两时间条件 | `build_row_time_pairs`：`r_i = 1−σ_{i+1}`；condition 行 `r=t`；参考音频 `t=r=1` | `h3_dit_schedule.c: prepare_rows / row_map` | 同语义；按 `(t,r)` 对去重，不是只按 `t` |
| 时间嵌入 | `temb = te(t) + gate·(ee(r)−te(t))`，`hyperflow_gate` 默认 0.25 | `h3_dit_schedule.c` 从权重 `hyperflow_gate` 读取 | 转换版 checkpoint 内为 0.25 |
| 剪块 | `dropped_blocks`：fl2va `[6,21,22]`、ref2va `[9]` | 转换版权重**已经物理删除**这些块并重编号（47/49）；`dit_layers < 50` 时才做额外启发式剪块 | 用 `balanced` / `quality` 等默认 `DitLayers=50`，不再叠加剪块 |
| INT8 ConvRot | 180 个线性层，Hadamard 块对角旋转，W8A8，`wscale` 按输出通道 | `h3_dit.c: load_projection` + `h3_st_read_f32` 的 I8 反量化路径，group 256 | 量化码与 scale 未重新计算，直接读转换版权重 |
| 低秩分解 | `LowRankLinear`，`lowrank_rank_multiple=8` | `h3_dit.c: load_lowrank` 按 checkpoint 形状读取 | 转换版已把因子 zero-pad 到 8 的倍数 |
| 序列与头 | 56 头 × 128、hidden 5376、50 层、text_dim 5120、rope_dim 16/θ=1e4 | `h3_dit.c` / `h3_text_encoder.c` 常量一致 | 含 packed `[text;cond;audio;video]` 布局 |
| 帧 / 画面 | 24 fps，`17n+5` 帧，32 倍数画布，短边 768，音频 32 kHz 立体声 | `h3_host.c` / `h3_audio_vae.c` / `h3_ffmpeg.c` 一致 | runtime 允许更短/更大的帧数与画布，按用户显式请求 |
| 参考图缩放 | `ref_image_size`: `match`（按输出像素面积） / `max`（短边上限） | ABI 字段 `reference_image_size`，`h3_reference_image_canvas` 同语义 | 只缩小不放大 |
| 文本编码器裁剪 | 保留第 51 层（读 `hidden_states[50]`），其余 64 层不加载 | 本 runtime 使用随 ComfyUI-V8 分发的 Q4_K_M GGUF，层已由转换流程处理 | 无 Python 侧 `trim_text_encoder`，因此不再额外裁剪 |

## 本轮修复的不一致

1. **步数被覆盖**：ABI `h3rt_fill_params()` 原先固定 `pdd8=true` 并把 `steps`
   强行改写成 8，导致 `steps: 4/6` 永远不生效。现在仅在调用方**未显式给
   steps**（或显式给 8）时保留旧的 8 步 PDD 默认值；显式 4/6 会原样透传。
2. **checkpoint sigma 只当开关**：原先只调用 `h3_st_dir_has_tensor()` 判断
   是否存在 `hyperflow_sigmas`，6 步网格在 C 里硬编码。现在新增
   `h3_st_dir_read_f32()` 读取该张量，`h3_hyperflow_schedule_from_stored()`
   用它构造 video/audio 网格；不合法或张量长度不匹配时回落到开源网格。
3. **WebUI 预设**：`turbo/balanced/quality/cinematic` 从「PDD 8 步 / 20 步
   均匀采样」改为 HyperFlow 原生 4/6/8/6 步；内置 demo 使用 `balanced`，
   即官方默认 6 步。高级版 `advSteps` 默认同步为 6。

## 已知差异（有意不移植 / 尚未移植）

- **多 GPU / Ulysses context parallel / FSDP2**：开源实现用 `torchrun` 切分
  packed sequence 并并行解码 VAE tile；本 runtime 是单 handle 串行引擎。
- **`offload=model/stream`**：开源实现把权重放进单个 pinned host arena，按层/
  按块流式搬到 GPU。本 runtime 依赖 mmap 权重 + `KeepDit` / `KeepResidentVae`
  两个驻留开关；24 GiB 卡连续生成时必须保持 `KeepResidentVae=false`。
- **`block_chunk=16384`**：开源实现把 transformer block 的逐 token 操作切片，
  让 15 秒混合 ref2va 能装进 24 GiB。本 runtime 用 VAE tile（`H3_VAE_TILE_PIXELS`）
  控制解码峰值，DiT 长序列没有等价切片。
- **FP8 文本编码器 / `fa3` / CUTLASS kernel 分派**：本 runtime 只服务随
  ComfyUI-V8 分发的 Q4_K_M GGUF 文本编码器与自编译 CUDA kernel。
- **参考音频 / 视频输入**：开源 `ref2va` 接受图像、视频（含音轨）和音频；
  WebUI 有对应字段，但是否完整生效取决于 ComfyUI-V8 引擎快照，未与 Python
  版逐例比对。
- **逐像素一致性**：即便参数一致，GGUF 反量化顺序、kernel 累加精度和
  RoPE/attention 实现不同，画面不会与 Python/ComfyUI 完全相同。

## 验证

### 双平台编译（Windows + WSL Debian 12）

- **Windows**：`scripts/build.cmd 89`（MSVC 14.51 + Windows SDK 10.0.26100 +
  CUDA sm_89）原生 DLL 与 .NET 解决方案均构建成功，0 警告 0 错误。
  `dist/win-x64/h3rt_smoke.exe`、`H3NetRuntime.Smoke` 全 PASS。
- **Linux / WSL Debian 12**：`scripts/build.sh 89`（gcc 12.2 + CUDA 13.0 +
  sm_89）原生 `libh3rt.so` 与 .NET 解决方案均构建成功，0 警告 0 错误。
  C smoke 与 .NET smoke（`.NET 10` 下用 `DOTNET_ROLL_FORWARD=LatestMajor`）
  全 PASS。`build_info` 报告 `platform=linux`、`runtime_version=1.2.0`。

### 采样对齐

- `dist/win-x64/h3rt_smoke.exe <model-pack>` 带真实模型包运行时，
  `h3rt_model_info()` 的 `sampling.hyperflow=true`、`checkpoint_sigmas`
  为 7 个 knot，校验通过。
- checkpoint 头部实测：`hyperflow_sigmas` shape `[7]`、值
  `(1.0, 0.9826559, 0.9146426, 0.6747557, 0.3852237, 0.1590553, 0.0)`，
  `hyperflow_gate = 0.25`；与开源 6 步网格经 shift 12 后的值一致。
- WebUI `/api/config` 返回四个预设的 `steps = 4/6/8/6`、`pdd8 = false`。

### 端到端生成（WebUI，HyperFlow 原生 6 步，RTX 4090）

| 模式 | 规格 | 结果 |
|---|---|---|
| 文生视频 `t2v` | 512×512 / 22 帧 / 6 步 | 完成，214 s，h264 22 帧 + aac 音频 |
| 首尾帧 `fl2v` | 512×512 / 22 帧 / 6 步，首帧 + 尾帧 | 完成，约 240 s，272 KB 成片 |
| 多图参考 `ref2v` | 512×512 / 22 帧 / 6 步，3 张参考图（49 层剪块） | 完成，约 330 s，221 KB 成片 |

三种模式都走 `hyperflow` 调度（日志：`HyperFlow two-time embedding: gate 0.2500`、
`transformer depth: 47 blocks (pruned)`），`denoise (0/6) … (6/6)` 单调推进。
注意：显存被其它进程占用 > 10 GiB 时，DiT 加载会掉进数小时/步的抖动慢路径；
测试前请保证 20.5 GiB 以上空闲显存。
