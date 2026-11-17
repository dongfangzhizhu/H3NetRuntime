# 模型权重：下载与安装

本仓库**不包含权重**。WebUI 里有一份同样内容的离线页面：
`wwwroot/guide.html`（启动后点「模型指引」）。

## 五个文件

面向 `ComfyUI_MiniMax-H3-V8` 发布的 HyperFlow Flash 权重，**保持原文件名即可**：

| 文件 | 目录 | 体积 | 用途 |
|---|---|---:|---|
| `minimax_h3_fl2va_hyperflow_flash_int8_convrot.safetensors` | `diffusion_models/` | ≈15.9 GB | FL2VA DiT，文生视频 / 首尾帧 |
| `minimax_h3_ref2va_hyperflow_flash_int8_convrot.safetensors` | `diffusion_models/` | ≈16.3 GB | Ref2VA DiT，仅多图参考需要 |
| `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` | `text_encoders/` | ≈14.6 GB | Qwen3-VL 文本编码器 |
| `minimax_h3_sol_vae_int8_convrot.safetensors` | `vae/` | ≈1.99 GB | 视频 VAE |
| `minimax_h3_audio_vae_fp32.safetensors` | `vae/` | ≈605 MB | 音频 VAE |

```text
ComfyUI_MiniMax-H3-V8/ComfyUI/models/
├── diffusion_models/   # 上面两个 DiT
├── text_encoders/      # GGUF 文本编码器
└── vae/                # 视频 + 音频 VAE
```

下载来源：ComfyUI_MiniMax-H3-V8 配套的 Hugging Face / ModelScope 仓库，
按文件名搜索；通常需要先接受 **MiniMax H3 Community License**。

**不需要** `config.json`、`pdd_heads.safetensors`、`pdd_lora.safetensors`
或 `tokenizer.json`：层数（47 / 49）、time embedder 宽度（rank 16）、
sigma 表与 gate（0.25）都写在 checkpoint 的 `__metadata__.hyperflow_int8` 里，
tokenizer 随本程序分发。

想用旧的 `FL2VA/`、`Ref2VA/` 标准包布局也可以，运行时仍按槽位探测；
但不要混用其它格式的 checkpoint（例如 bf16 / 非 convrot / GGUF DiT）。

## 告诉 WebUI 模型在哪

优先级：

```text
--model-dir / H3:ModelDir → H3_MODEL_DIR → h3.config.json 的 h3_dir
→ ~/h3-models/pdd8 → <程序目录>/models → ComfyUI_MiniMax-H3-V8/ComfyUI/models
```

启动后控制台会打印每个组件的**完整解析路径**，例如：

```text
模型来源  : .../ComfyUI/models  (ComfyUI models/ 目录)
  [OK  ] FL2VA DiT      transformer              .../diffusion_models/minimax_h3_fl2va_...safetensors
  [OK  ] 文本编码器     text_encoder              .../text_encoders/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf
```

页面顶部也会列出缺失组件；`POST /api/model/refresh` 可重新扫描磁盘。

## 权重散落在多个目录

不必改名或搬动，在 `appsettings.json` 里逐组件指定绝对路径：

```json
"H3": {
  "Models": {
    "Transformer": "E:/hf/minimax_h3_fl2va_hyperflow_flash_int8_convrot.safetensors",
    "Ref2vaTransformer": "E:/hf/minimax_h3_ref2va_hyperflow_flash_int8_convrot.safetensors",
    "TextEncoder": "E:/hf/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf",
    "VideoVae": "E:/hf/minimax_h3_sol_vae_int8_convrot.safetensors",
    "AudioVae": "E:/hf/minimax_h3_audio_vae_fp32.safetensors"
  },
  "PackCacheDir": "E:/hf/.h3rt-packs"
}
```

留空的项回落到自动探测结果。运行时用硬链接 / 目录联接（Windows）或
符号链接（Linux）拼出引擎认识的标准布局，**不复制权重**；
把 `PackCacheDir` 放在权重所在卷可保证链接不退化成复制。

## 常见问题

| 现象 | 处理 |
|---|---|
| 「多图参考」不可用 | 缺 Ref2VA DiT；其它四个组件与 FL2VA 共用 |
| `算力 89 ≠ 编译 120` | 用 `scripts/build.cmd <算力>` 重建原生库 |
| 去噪几十秒一步 | 显存被其它进程占用，`nvidia-smi` 检查 |
| ffmpeg 相关失败 | 把 `ffmpeg` / `ffprobe` 加入 PATH |

## 许可

权重遵循 **MiniMax H3 Community License Agreement**（欧盟、英国、韩国、美国需另行授权），
公开分享生成内容时请标注「由 AI 生成」。本项目代码 Apache-2.0，不含权重。
