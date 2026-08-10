# LiDAR-Enhanced MVSplat

本项目基于 [MVSplat](https://github.com/donydchen/mvsplat) 开发，在多视角 cost volume 深度预测中加入 nuScenes LiDAR 信息，用于研究稀疏 LiDAR 对前馈 3D Gaussian Splatting 的帮助。

当前主要实验分支：

- `temporal`：不使用 LiDAR 的 nuScenes 时序基线；
- `lidar_bias`：固定解析式 LiDAR Bias；
- `lidar_cross_attention`：LiDAR cross-attention 实验分支。

LiDAR Bias 由 Temperature scaling、Surface attraction prior 和 Free-space suppression prior 组成。

## 1. 环境安装

推荐使用 Linux、Python 3.10 和支持 CUDA 的 NVIDIA GPU。以下版本与 MVSplat 官方配置一致：

```bash
conda create -n lidar_mvsplat python=3.10 -y
conda activate lidar_mvsplat

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install nuscenes-devkit
pip install ./diff-gaussian-rasterization-modified-main
```

检查 CUDA rasterizer：

```bash
python -c "import diff_gaussian_rasterization; print('Rasterizer OK')"
```

如果编译失败，请检查 PyTorch CUDA、CUDA Toolkit、GCC/G++ 是否兼容，以及 `diff-gaussian-rasterization-modified-main/third_party` 是否完整。

## 2. 数据准备

将 nuScenes 放到：

```text
datasets/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-mini/
└── v1.0-trainval/
```

调试可使用 `v1.0-mini`，正式实验应统一数据版本、scene split、context/target 设置和随机种子。

当前默认时序输入：

```text
context: [t-2, t-1]
target:  [t]
camera:  CAM_FRONT
```

## 3. Checkpoint 准备

从 [MVSplat 官方仓库](https://github.com/donydchen/mvsplat)下载 RE10K 预训练模型，保存为：

```text
checkpoints/re10k.ckpt
```

所有 LiDAR 消融统一从以下 temporal checkpoint 开始：

```text
checkpoints/kll6000.ckpt
```

当前 `temporal.yaml` 和 `lidar_bias.yaml` 设置了 `unimatch_weights_path: null`，因此从 RE10K checkpoint 微调时不需要额外 GMDepth 权重；从头训练基础 MVSplat 时仍需按官方说明准备该权重。

## 4. 训练 temporal 基线

从官方 RE10K checkpoint 在 nuScenes 上训练 6000 steps：

```bash
python -m src.main \
  +experiment=temporal \
  mode=train \
  checkpointing.load=checkpoints/re10k.ckpt \
  checkpointing.resume=false \
  trainer.max_steps=6000
```

将训练完成的 checkpoint 保存为 `checkpoints/kll6000.ckpt`。所有后续消融必须从同一个 checkpoint 独立开始，不能继承其他消融实验的 checkpoint。

## 5. 训练 LiDAR Bias

正式的 Bias+Refine 训练：

```bash
python -m src.main \
  +experiment=lidar_bias \
  mode=train \
  checkpointing.load=checkpoints/kll6000.ckpt \
  checkpointing.resume=false \
  wandb.name=lidar_bias_refine
```

主要配置：

```yaml
model:
  encoder:
    use_lidar_bias: true
    use_lidar_cross_attention: false
    lidar_cross_attention_inference_mode: "off"
    use_lidar_coarse_loss: false
    use_lidar_refine_loss: true
    lidar_loss_weight: 0.0
    lidar_final_loss_weight: 0.05
    frozen_params: []
```

`frozen_params: []` 表示 backbone、cost volume、refinement U-Net 和 Gaussian head 均参与训练。所有主实验应使用一致的冻结策略。

## 6. 测试与评测

```bash
python -m src.main \
  +experiment=lidar_bias \
  mode=test \
  checkpointing.load=checkpoints/<YOUR_CHECKPOINT>.ckpt \
  dataset/view_sampler=evaluation \
  test.compute_scores=true \
  hydra.run.dir=outputs/lidar_bias/<EXPERIMENT_NAME>
```
评测结果和渲染图像会写入对应的 `outputs` 目录。
## 7. W&B


在 `config/main.yaml` 中设置项目与账号，或通过命令行覆盖。首次使用运行：

```bash
wandb login
```

请勿将个人 W&B API key 提交到 GitHub。

## 8. 推荐目录结构

```text
.
├── checkpoints/
│   ├── re10k.ckpt
│   └── kll6000.ckpt
├── datasets/
│   └── nuscenes/
├── diff-gaussian-rasterization-modified-main/
├── config/
├── src/
├── outputs/
├── requirements.txt
└── README.md
```

## 9. 常见问题

### `on` 被解析成布尔值

YAML 中的 `on/off` 可能被解析成布尔值，必须加引号：

```yaml
lidar_cross_attention_inference_mode: "off"
```

LiDAR Bias 实验应保持 Cross-Attention 关闭。

### 找不到数据或 checkpoint

所有相对路径均以项目根目录为基准。检查：

```bash
ls datasets/nuscenes
ls checkpoints/re10k.ckpt
ls checkpoints/kll6000.ckpt
```

## 10. 致谢与引用

本项目基于 [MVSplat](https://github.com/donydchen/mvsplat) 开发。原项目的安装、预训练模型和基础训练方式以官方说明为准。

```bibtex
@article{chen2024mvsplat,
  title   = {MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images},
  author  = {Chen, Yuedong and Xu, Haofei and Zheng, Chuanxia and Zhuang, Bohan and Pollefeys, Marc and Geiger, Andreas and Cham, Tat-Jen and Cai, Jianfei},
  journal = {arXiv preprint arXiv:2403.14627},
  year    = {2024}
}
```