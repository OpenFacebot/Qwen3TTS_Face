# Qwen3-TTS holdlistenstream

本实现基于 `Qwen/Qwen3-TTS-12Hz-1.7B-Base`。音频、面部 codec 和音色编码器使用冻结的已有权重，训练 Talker、HOLD/LISTEN 与历史标记、面部预测头。semantic-GAN 的 GAN 训练已经体现在 codec 权重中。

## 序列与数据

一个调度块为 0.4 秒。音频 codec 是 **12.5 Hz**，每块 5 帧，每帧有 16 个码本。面部 codec 输入为 25 Hz 的 61 维动作，hop=5，每块 2 组。

| codec | 主 LM 文本:音频:面部位置 | 每组面部码本 | 预测方式 |
|---|---|---|---|
| semantic / semantic_gan | 4:5:4 | 5 | 两个主码本 + 三个辅助码本 |
| vae_vq / vae_fsq | 4:5:2 | 1 | 串行 |
| semantic_first / semantic_first_gan | 4:5:10 | 5 | 串行 |

表中比例按主 LM 位置计算。若按全部离散码计数，semantic-GAN 每块包含 4 个文本槽、80 个音频码和 10 个面部码；其中 6 个面部辅助码由 MTP 预测。

四个文本槽表示文本 token 或 HOLD/LISTEN。转入 LISTEN 会清空尚未消耗的文本。训练按原 CosyVoice 规则在句首投递文本，语音结束后保持 0.8 秒；窗口默认 60 秒、每次前移 30 秒，重叠部分仅提供上下文。所有预测使用目标位置之前的隐藏状态。

推理轮次控制与 CosyVoice 一致：`speaking=None` 时，新文本进入说话状态，空文本保持上一次状态；显式 `speaking=False` 进入 LISTEN，不能同时提交新文本。0.4 秒是此调度协议的单位，音频 codec 本身的帧间隔是 0.08 秒。

原始动作沿用 V2 的 30→25 Hz 三次插值、异常值修复、眼球 roll 五点中值滤波，以及面部/头部/眼部 7/3/2 Hz 四阶零相位低通。归一化由已有 codec 内部的统计量完成。

## 环境和权重

当前远端 `.venv` 已准备好。需要重建环境时，在项目根目录运行以下命令；本机 Python 缺少 ensurepip，借用已有 pip 的 `--python` 选项安装到新环境中。现有 CosyVoice 环境独立保留。

```bash
cd /data/L202500493/Audio_Face/Qwen3TTS_Face
git submodule update --init --recursive
python3 -m venv --without-pip .venv
export PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL=""
../CosyVoice_Face/.venv/bin/python -m pip --python .venv/bin/python install pip
.venv/bin/pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -e . -e Facial_Codec numpy==1.26.4 scipy==1.15.3 pyarrow==18.1.0 pytest sentencepiece \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

`qwen-tts` 固定 Transformers 4.57.3、Accelerate 1.12.0。安装 Facial_Codec 基础包即可；其教师训练扩展使用另一套 Transformers 版本。

将完整的 Qwen Base 快照（包括 `speech_tokenizer/`）放在 `.runtime/models/Qwen3-TTS-12Hz-1.7B-Base`。模型和 tokenizer 内容均记录 SHA-256。面部权重默认从相邻 CosyVoice_Face 的 `face_tokenizer/runs/experiments/base/<variant>/checkpoints/epoch_005.pt` 读取，并核验 codec 类型、量化器、GAN 状态和文件摘要。

## 完整重建和恢复

首次重建两套数据，默认使用空闲 GPU 6、7：

```bash
DEVICES=6,7 bash finetuning/run_holdlistenstream_prepare.sh
```

脚本依次导出源清单、选取并编码固定参考音色、分配两张 GPU 转换全部人物流，最后校验所有产物。两套面部数据共享新生成的 Qwen 音频码和文本槽。本轮语料清单为 train 22,949、dev 322 条人物流。

保留旧清单的 train/dev 划分和已审定时间范围，只从旧 parquet 读取身份与时间范围；音频码、文本槽、面部码和参考音色均重新计算。本次训练/验证数据分别为 1,801.564 / 19.1122 小时，按 60/30 秒规则形成 203,999 / 2,126 个窗口。旧清单未覆盖的原始尾部最长为 50 个动作帧（2 秒）。

中断后续跑：

```bash
RESUME=1 DEVICES=6,7 bash finetuning/run_holdlistenstream_prepare.sh
```

完整重建要求使用新输出目录，以免覆盖已有数据。已经导出的源清单可独立用于再次重建，无需旧 token 缓存：

```bash
SOURCE_MANIFEST=data/qwen_holdlistenstream_v1/sources.json \
OUTPUT=data/qwen_holdlistenstream_v2 DEVICES=6,7 \
bash finetuning/run_holdlistenstream_prepare.sh
```

通用转换入口 `python -m qwen_tts.holdlistenstream.prepare` 支持 `--codecs` 选择六种 codec，`--stage index|references|encode|validate` 单独运行阶段，以及 `--worker/--workers` 分配编码任务。`--max-recordings` 只用于诊断；部分产物不能通过完整校验。

仅校验现有构建：

```bash
.venv/bin/python finetuning/prepare_holdlistenstream.py \
  --model .runtime/models/Qwen3-TTS-12Hz-1.7B-Base \
  --output data/qwen_holdlistenstream_v1 --validate-only
```

数据目录保存源清单、模型与 codec 指纹、每个产物的摘要和原始数据摘要、异常值修复记录、参考音频及音色向量。最终 `validation.json` 标记完整构建。训练加载时再次验证索引和使用到的文件摘要。

每个 split 内的每位参与者使用固定参考音色：选择其最长 VAD 片段，居中截取最多 6 秒；短于 3 秒的参考单独记录在验证报告中。

### 当前服务器上的存储

共享盘读取大量小文件较慢，本次运行使用本机缓存：

- `data/qwen_holdlistenstream_v1` 指向 `/tmp/qwen3tts_face/holdlistenstream_v1`。
- `.runtime/models/Qwen3-TTS-12Hz-1.7B-Base` 指向 `/tmp/qwen3tts_face/models/Qwen3-TTS-12Hz-1.7B-Base`。
- 完整数据归档保存在共享盘 `data/qwen_holdlistenstream_v1.tar`，旁边的 `.sha256` 文件用于校验；模型原件保存在 `.runtime/models/Qwen3-TTS-12Hz-1.7B-Base.shared`。

若本机缓存被清理，在项目根目录恢复后即可使用原有命令：

```bash
(cd data && sha256sum -c qwen_holdlistenstream_v1.tar.sha256)
mkdir -p /tmp/qwen3tts_face/models
test -d /tmp/qwen3tts_face/holdlistenstream_v1 || \
  tar -xf data/qwen_holdlistenstream_v1.tar -C /tmp/qwen3tts_face
test -d /tmp/qwen3tts_face/models/Qwen3-TTS-12Hz-1.7B-Base || \
  cp -a .runtime/models/Qwen3-TTS-12Hz-1.7B-Base.shared \
    /tmp/qwen3tts_face/models/Qwen3-TTS-12Hz-1.7B-Base
```

`data/qwen_holdlistenstream_v1.shared-partial` 只是切换缓存位置时保留的中间产物；完整数据以归档及通过校验的目录为准。训练权重直接保存在共享盘 `runs/` 下。

## 训练

semantic-GAN 的两轮短程验证：

```bash
CUDA_VISIBLE_DEVICES=6,7 .venv/bin/accelerate launch \
  --num_processes 2 --num_machines 1 --multi_gpu --mixed_precision bf16 --main_process_port 29671 \
  finetuning/train_holdlistenstream.py \
  --config finetuning/configs/holdlistenstream_semantic_gan.json \
  --epochs 2 --train-limit 512 --dev-limit 64 --output runs/smoke_semantic_gan
```

VAE-VQ 使用 `holdlistenstream_vae_vq.json` 和独立的输出目录。正式训练使用同一命令，删除三个短程参数 `--epochs 2 --train-limit 512 --dev-limit 64`，并选择正式输出目录。默认 20 epochs、每卡 batch=2、累积=4、BF16、梯度检查点、AdamW lr=1e-5、weight decay=0.01、梯度裁剪=1。

完整 semantic-GAN 训练命令：

```bash
CUDA_VISIBLE_DEVICES=6,7 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
  .venv/bin/accelerate launch \
  --num_processes 2 --num_machines 1 --multi_gpu --mixed_precision bf16 --main_process_port 29671 \
  finetuning/train_holdlistenstream.py \
  --config finetuning/configs/holdlistenstream_semantic_gan.json \
  --output runs/holdlistenstream_semantic_gan
```

每轮产生 `checkpoint-epoch-NNNN.pt`，包含模型、优化器、常数学习率调度器、各 rank 随机状态及数据指纹，完成后原子更新 `latest.pt`。恢复时保留原训练参数并添加：

```bash
--resume runs/smoke_semantic_gan/latest.pt
```

`metrics.jsonl` 记录音频首码本、音频辅助码本、面部损失及每轮验证结果。总损失为 `audio + 0.3 * audio_aux + face`，面部损失在所有码本上平均。

### semantic 30/6 实验

`finetuning/configs/holdlistenstream_semantic_30_6.json` 使用非 GAN 的 `semantic` codec，窗口 30 秒、步长 6 秒，保留 24 秒上下文；重叠部分不重复计算损失。训练为 10 epochs，每轮保存完整检查点，其他优化器设置沿用上面的配置。全量数据对应 996,910 个训练窗口和 10,298 个验证窗口。

本实验的数据为 `data/qwen_holdlistenstream_semantic_v1`，面部 token 从原始动作和冻结的 `semantic/epoch_005.pt` 重新生成。Qwen 音频码、文本槽和固定参考音色复用已完整校验的 `qwen_holdlistenstream_v1`。复用入口逐项比较源清单、Qwen 权重、预处理和时钟设置，并核对父构建索引固定的文件摘要；新构建保存独立副本及来源记录。

再次准备同类数据时选择新的输出目录，例如以下单卡命令；`--stage index` 可先完成索引和复用，再用 `--stage encode --worker/--workers` 分配多卡任务，最后 `--validate-only`：

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
  .venv/bin/python -m qwen_tts.holdlistenstream.prepare \
  --model .runtime/models/Qwen3-TTS-12Hz-1.7B-Base \
  --output data/qwen_holdlistenstream_semantic_v2 \
  --source-manifest data/qwen_holdlistenstream_v1/sources.json \
  --reuse-common-from data/qwen_holdlistenstream_v1 \
  --codecs semantic --stage all --rebuild all
```

使用已准备的 v1 数据启动训练：

```bash
CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
  .venv/bin/accelerate launch \
  --num_processes 2 --num_machines 1 --multi_gpu --mixed_precision bf16 --main_process_port 29672 \
  finetuning/train_holdlistenstream.py \
  --config finetuning/configs/holdlistenstream_semantic_30_6.json \
  --epochs 10 --train-limit 0 --dev-limit 0
```

输出目录为 `runs/holdlistenstream_semantic_30_6`。需要恢复时，保留相同参数并添加 `--resume runs/holdlistenstream_semantic_30_6/latest.pt`。使用另一次数据构建时，通过 `--data` 和 `--output` 指定新的数据与实验目录。

本机 v1 数据缓存指向 `/tmp/qwen3tts_face/holdlistenstream_semantic_v1`，完整归档保存为 `data/qwen_holdlistenstream_semantic_v1.tar` 及旁边的 `.sha256` 文件。恢复方式与上面的原数据缓存相同。

## 流式推理和验证

```python
import torch
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    ".runtime/models/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0", torch_dtype=torch.bfloat16, attn_implementation="sdpa",
)
stream = model.start_holdlistenstream(
    "runs/smoke_semantic_gan/latest.pt", ref_audio="reference.wav",
)
try:
    chunk = stream.tick("Hello, how are you?", speaking=True)
    # 每次返回 audio[9600]、motion[10,61]、audio_codes[5,16]、face_codes[2,K]。
    chunk = stream.tick("", speaking=True)   # 消耗剩余文本或 HOLD
    chunk = stream.tick("", speaking=False)  # LISTEN，清空待输出文本
finally:
    stream.close()
```

也可通过 `speaker_embedding=` 传入缓存的 2048 维音色向量，二者必须选一个。`close()` 释放会话状态，不额外生成块。默认贪心解码；`do_sample=True` 可启用采样。每次窗口切换重建有限长度的 LM KV cache，音频解码仅保留前 25 帧码，面部解码保持自己的有限历史。

同一个 Qwen 实例的会话共享 Talker 权重；不同 codec 或不同训练检查点应分别创建 Qwen 实例。

```bash
.venv/bin/python -m pytest -q tests/test_holdlistenstream.py

CUDA_VISIBLE_DEVICES=6 .venv/bin/python finetuning/verify_holdlistenstream.py \
  --checkpoint runs/smoke_semantic_gan/latest.pt \
  --reference data/qwen_holdlistenstream_v1/references/dev-P0799.npz \
  --output runs/verify_semantic_gan
```

参考文件请从构建的 `references.json` 中选择。验证脚本比较缓存和完整前缀的贪心 token，连续生成 450 块（180 秒），保存音频、动作、token 和时延/实时率报告。短程训练用于检查链路与恢复能力，模型质量需要正式训练后评估。

完整 codec 与旧预处理回归检查：

```bash
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
HOLDLISTEN_CODEC_ROOT=../CosyVoice_Face/face_tokenizer/runs/experiments/base \
HOLDLISTEN_SOURCE_MANIFEST=data/qwen_holdlistenstream_v1/sources.json \
HOLDLISTEN_TEST_DEVICE=cuda:0 \
HOLDLISTEN_LEGACY_TOOLS=/data/L202500493/tmp/semantic_motion/vendor/cosyvoice_base/tools \
.venv/bin/python -m pytest -q tests/test_holdlistenstream.py tests/test_holdlistenstream_assets.py
```

## 本次验证结果

详见 [验证报告](holdlistenstream_validation.json)。24 项单元检查和 7 项真实资产检查通过。每套模型使用 512 个训练窗口和 64 个验证窗口，先完成第 1 个 epoch，再恢复到第 2 个 epoch；每轮保存完整检查点，最终 global step 均为 64。两个 rank 的随机状态、优化器和调度器均已恢复，全部优化器参数状态均核对为 step 64。

| codec | 第 1 轮验证损失 | 第 2 轮验证损失 | 平均块耗时 | RTF |
|---|---:|---:|---:|---:|
| semantic-GAN | 11.639 | 10.670 | 0.504 秒 | 1.260 |
| VAE-VQ | 11.839 | 10.706 | 0.412 秒 | 1.031 |

两套模型均完成 180 秒连续生成，分别输出 4,320,000 个音频采样点和 4,500 帧动作，完成 4 次窗口滚动。缓存/完整前缀 token 一致性在 FP32 下验证，长流式测试使用 BF16。RTF 大于 1，当前实现尚未达到持续实时生成速度。完整 20 轮训练命令已准备好，本次仅执行上述短程验证。

原始动作中 14,540 条流含非有限值，共 73,353,813 个值，沿用旧 V2 规则修复。464 份参考音色中，`train-P0321` 的最长语音参考为 1.18 秒，已在数据报告中标记。共享盘归档的 70,277 个 NPZ 产物均重新核对了内容摘要。

检查点位于 `runs/smoke_semantic_gan/` 和 `runs/smoke_vae_vq/`；音频、动作、token 和流式报告位于 `runs/verify_semantic_gan/` 和 `runs/verify_vae_vq/`。
