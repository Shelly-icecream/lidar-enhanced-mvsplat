# LiDAR-Enhanced MVSplat

当前 `pruned_shared` 分支：删除 `corr_refine_net`、`regressor_residual`、`depth_head_lowres`，与 `pruned_split_alternating` 一致使用原始 cost-volume 相关性除以 `cost_volume_temperature`（默认 0.1）后做 softmax。`to_disparity` 和 `to_gaussians` 仍为两个 context 共享的头，无拆头和交替训练。旧 full/shared 权重加载时仅忽略被删除模块的参数，其余参数仍严格检查；跨架构初始化使用 `checkpointing.resume=false`，不恢复旧 optimizer 状态。

本项目基于 [MVSplat](https://github.com/donydchen/mvsplat) 开发，面向 nuScenes 时序多视角输入，将稀疏 LiDAR 引入 cost-volume 深度预测和 3D Gaussian 属性预测。当前实现包含 LiDAR 深度锚点、深度分支微调与 Gaussian 属性修正。

## 1. 当前进度

目前仓库包含以下实验配置（旧配置的冻结前缀迁移见下文）：

| 配置 | 作用 | 当前训练目标 |
| --- | --- | --- |
| `baseline.yaml` | 加载 RE10K checkpoint，关闭全部 LiDAR 分支 | 不训练额外模块，用于视觉模型直接推理 |
| `stage0.yaml` | 加载 RE10K checkpoint | LiDAR Gaussian adapter 关闭，用于推理与消融 |
| `stage2.yaml` | LiDAR Gaussian 属性修正 | 冻结基础网络，训练 Gaussian adapter |

注意：`baseline/re10k.yaml` 当前仍保留旧冻结项 `depth_predictor.to_disparity`。使用这些配置前，需要将该项替换为 `depth_predictor.to_disparity_disps` 和 `depth_predictor.to_disparity_opacity`，否则 encoder 初始化的严格前缀检查会报错。

已经完成的主要改动：

- 新增 nuScenes 时序数据接入，默认使用 `CAM_FRONT`、两个 context 帧和一个 target 帧；
- 新增运动物体 mask：将带动态属性的 nuScenes 3D annotation box 投影到图像，并扩张为保守的二维遮罩；动态区域的 LiDAR 点会在进入模型前被删除，训练损失也只在静态 target 像素上统计；
- 在视觉深度 refinement 之后注入合法 LiDAR disparity，LiDAR 点作为稀疏硬锚点，超出相机 near/far 范围的点会被丢弃；
- 新增 LiDAR Gaussian cross-attention adapter，可按配置修正 Gaussian 的 SH DC 与 opacity（当前 Stage 2 默认设置）；
- 新增Gaussian adapter 的专用训练损失、alpha 改善指标及动态诊断输出；
- Gaussian 中心深度统一由 camera z-depth 转换为沿射线距离，并使用预测的 sub-pixel offset 计算对应射线；
- 增加严格的 `frozen_params` 前缀检查，避免配置写错后意外训练基础网络。
- 将旧 `to_disparity` 拆为逆深度修正头 `to_disparity_disps` 和 density 头 `to_disparity_opacity`，各输出 `K = gaussians_per_pixel` 个通道，可分别冻结和训练；
- 新增 `ModelWrapper.load_split_checkpoint`：两个头复制旧第一层，最后一层分别取旧输出的前、后 K 个通道；新双头 checkpoint 直接加载。微调和测试入口均支持迁移，测试不再重复加载原始旧文件；
- target 动态 mask 合并 target 时刻与各 context 时刻动态物体 3D 框在 target 相机中的投影，进一步排除时序运动和遮挡区域的监督。

当前阶段仍以 `v1.0-mini`、`176 × 320` 分辨率和 2000 steps 配置进行开发验证；正式结论应在 trainval split、更长训练和统一 checkpoint 条件下复现。

## 2. 模型流程

```mermaid
flowchart TD
    A["两张 Context RGB"] --> B["backbone · 冻结"]
    B --> C["多深度候选 Cost Volume"]
    C --> D["corr_refine_net · 冻结"]
    C --> E["regressor_residual · 冻结"]
    D --> F["相加"]
    E --> F
    B --> I["upsampler · 冻结"]
    I --> J["proj_feature · 冻结"]

    subgraph DEPTH["深度分支 · Stage 3 可训练"]
        G["depth_head_lowres"]
        H["概率分布 → 粗逆深度上采样<br/>置信度上采样"]
        L["refine_unet"]
        M["to_disparity_disps"]
        V["基础逆深度 + 逆深度残差<br/>裁剪前视觉逆深度"]
        G --> H --> L --> M
        H -->|"基础逆深度"| V
        M -->|"残差"| V
    end

    subgraph ATTR["参考属性分支 · 冻结"]
        G0["depth_head_lowres_attributes"]
        H0["参考概率分布 → 粗逆深度上采样<br/>置信度上采样"]
        L0["refine_unet_attributes"]
        O0["to_disparity_opacity"]
        G0 --> H0 --> L0 --> O0
    end

    F --> G
    F --> G0
    J --> L
    J --> L0
    A --> L
    A --> L0

    L0 --> AC["拼接参考属性特征 + Context RGB + upsampler 特征"]
    A --> AC
    I --> AC
    AC --> N["to_gaussians · 冻结<br/>统一输出 offset + scale + rotation + color"]
    N --> P["高斯属性"]

    V --> CL["clamp 到 near/far 逆深度范围"]
    CL --> Z["取倒数 → 最终深度"]
    LD["Context LiDAR 逆深度<br/>及有效掩码"]
    O0 --> OP["sigmoid → density → opacity 映射"]
    Z --> Q["GaussianAdapter · 高斯参数构建<br/>非可训练 LiDAR 残差 adapter"]
    OP --> Q
    P --> Q
    Q --> R["3D Gaussians"]
    R --> RENDER["渲染目标帧 t"]
    RENDER --> RGBLOSS["L_RGB = L_MSE + 0.05 L_LPIPS + 0.05 L_grad"]
    TARGET["目标帧 RGB"] --> RGBLOSS
    MASK["目标帧 dynamic_mask"] -->|"MSE 排除动态像素；grad 要求两端有效<br/>当前 LPIPS 仍比较整图"| RGBLOSS
    V --> LL["L_LiDAR：有效像素逆深度 L1"]
    LD --> LL
    RGBLOSS --> TOTAL["L = L_RGB + 1.0 L_LiDAR"]
    LL --> TOTAL

    classDef trainable fill:#dcfce7,stroke:#16a34a;
    classDef frozen fill:#f1f5f9,stroke:#64748b;
    classDef loss fill:#fef3c7,stroke:#d97706;
    class G,L,M trainable;
    class B,D,E,I,J,G0,L0,O0,N frozen;
    class LL,RGBLOSS,TOTAL loss;
```



说明：`to_gaussians` 预测几何/颜色原始参数，`to_disparity_disps` 预测 full-resolution disparity residual，`to_disparity_opacity` 预测 density 原始值，再经 sigmoid 与 opacity 映射。中心位置还依赖像素偏移和相机参数。两个头共享 refinement 特征；单独训练一个头时，需冻结共享网络才能避免另一头的输入随训练改变。Stage 2 adapter 默认只修改 LiDAR 像素处的 SH DC 和 opacity，不修改 xy、scale、rotation 或其余 SH 系数。RGB 图像本身不会被 dynamic mask 擦除，因此 backbone 和 cost volume 仍能看到完整画面；mask 只约束 LiDAR 与损失的有效区域。

## 3. 环境安装

推荐 Linux、Python 3.10 和支持 CUDA 的 NVIDIA GPU：

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

如编译失败，请检查 PyTorch CUDA、CUDA Toolkit 和 GCC/G++ 的版本兼容性，以及 `diff-gaussian-rasterization-modified-main/third_party` 是否完整。

## 4. 数据与 checkpoint

将 nuScenes 放置为：

```text
datasets/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-mini/
└── v1.0-trainval/
```

默认时序设置为：

```text
context: [t-1, t+1]
target:  [t]
camera:  CAM_FRONT
```

### 4.1 运动物体 mask

数据集根据 nuScenes annotation 的动态 attribute 筛选运动物体，将其世界坐标 3D box 投影到对应相机。投影框会按比例扩张，并保证最小像素 padding，以覆盖标注框边缘和投影误差：

```yaml
dataset:
  dynamic_mask_expansion_ratio: 0.15
  dynamic_mask_min_padding_px: 8
  dynamic_mask_min_depth: 0.01
```

其中，`target.dynamic_mask` 是 target 时刻与各 context 时刻动态物体 3D annotation 在 target 相机中的投影并集。这里重新投影的是各时刻的 3D 框，并非直接 warp context 的二维 mask。

生成的 `context.dynamic_mask` 和 `target.dynamic_mask` 用于：

- 在数据加载阶段删除 context 动态区域内的 LiDAR depth/mask；target
  dynamic mask 不再生成对应的 LiDAR 张量；
- 从 MSE 和 Gaussian adapter loss 中排除动态 target 像素；
- 在启用 context render loss 时，只监督静态 LiDAR 像素。

从 [MVSplat 官方仓库](https://github.com/donydchen/mvsplat)下载 RE10K 预训练模型并保存为：

```text
checkpoints/re10k.ckpt
```

当前配置的 `unimatch_weights_path: null` 表示从上述完整 checkpoint 加载时无需额外准备 GMDepth/UniMatch 权重；从头训练 backbone 时需要自行提供相应初始化。

## 5. 训练

### 5.1 Stage 2：Gaussian 属性修正

Stage 2 训练 Gaussian adapter。仓库配置中的 checkpoint 文件名仅是本地实验默认值，建议显式覆盖：

```bash
python -m src.main \
  +experiment=stage2 \
  mode=train \
  checkpointing.load=checkpoints/re10k.ckpt \
  checkpointing.resume=false
```

Stage 2 冻结基础网络，只训练 `depth_predictor.lidar_gaussian_adapter`。当前配置允许它修正 `SH DC + opacity`，并在非 LiDAR 像素的最终写入边界保持原始 Gaussian 参数不变。

### 5.2 Adaptation：只微调逆深度头

```bash
python -m src.main \
  +experiment=adaptation \
  mode=train \
  checkpointing.load=checkpoints/re10k.ckpt \
  checkpointing.resume=false
```

当前配置使用学习率 `1e-6`、2000 steps、MSE 监督（LPIPS 权重为 0），冻结 backbone、refinement、opacity 头和 `to_gaussians`。LiDAR Gaussian adapter 关闭。旧单头权重会自动拆分；新双头权重再次微调时直接加载。`resume=true` 用于恢复结构与配置匹配的新双头训练 checkpoint，不用于迁移旧单头优化器状态。

## 6. 测试与评测

视觉模型 Baseline 关闭 LiDAR Gaussian adapter，直接加载
`re10k.ckpt` 推理：

```bash
python -m src.main \
  +experiment=baseline \
  mode=test \
  checkpointing.load=checkpoints/re10k.ckpt \
  checkpointing.resume=false \
  dataset/view_sampler=evaluation \
  test.compute_scores=true \
  hydra.run.dir=outputs/baseline/inference
```

根据 checkpoint 所属阶段选择相同实验配置：

```bash
python -m src.main \
  +experiment=stage2 \
  mode=test \
  checkpointing.load=checkpoints/<YOUR_CHECKPOINT>.ckpt \
  dataset/view_sampler=evaluation \
  test.compute_scores=true \
  hydra.run.dir=outputs/stage2/<EXPERIMENT_NAME>
```

评测分数和渲染图像写入对应的 `outputs` 目录。测试时可用 `save_lidar_alpha_diagnostics` 控制 LiDAR alpha 诊断；运动物体 mask 的生成与过滤逻辑见 4.1 节。

### 6.1 分别渲染 Context 高斯和完整高斯

脚本 `src/scripts/render_context_gaussians.py` 默认继承 `baseline.yaml`，
加载 `checkpoints/re10k.ckpt`，沿用 baseline 的数据集、相机、分辨率和测试采样配置。
一次编码后，将各 context view 的 Gaussian 子集分别渲染到 target 相机，
同时输出全部 context Gaussian 的完整渲染和 target 真值。需要 CUDA rasterizer：

```bash
python -m src.scripts.render_context_gaussians

# 仅渲染一个测试 batch，可覆盖 checkpoint 和输出目录
python -m src.scripts.render_context_gaussians \
  checkpointing.load=checkpoints/re10k.ckpt \
  render.max_batches=1 \
  render.output_dir=outputs/context_render
```

结果保存在
`outputs/context_render/<SCENE>/batch_<BATCH>/target_<INDEX>/`，其中
`context_0.png`、`context_1.png` 是各 context Gaussian 子集的渲染，
`full.png` 是全部 context Gaussian 的联合渲染，`ground_truth.png` 是 target 图像。
完整配置保存为输出目录下的 `config.yaml`。
nuScenes 当前返回的 `INDEX` 是样本内部视角编号，context/target 帧由数据集自动选取。

## 7. W&B

可在实验 YAML 中设置 W&B 项目、实体与运行名称，或通过命令行覆盖。首次使用运行：

```bash
wandb login
```

请勿将个人 W&B API key 提交到仓库。共享配置前也应将个人 `wandb.entity` 改为团队实体或通过命令行传入。

## 8. 推荐目录结构

```text
.
├── checkpoints/
│   ├── re10k.ckpt
│   └── <YOUR_CHECKPOINT>.ckpt
├── datasets/
│   └── nuscenes/
├── diff-gaussian-rasterization-modified-main/
├── config/
├── src/
├── outputs/
├── requirements.txt
└── README.md
```

所有相对路径均以项目根目录为基准。遇到数据或 checkpoint 找不到时，可先检查：

```bash
ls datasets/nuscenes
ls checkpoints/re10k.ckpt
```

### LiDAR bias：训练与测试独立开关

nuScenes 的 `stage0`、`stage1`、`stage2`、`test`、`test_` 配置均开启以下选项：

```yaml
dataset:
  load_lidar_depth: true
model:
  encoder:
    use_lidar_bias_train: true
    use_lidar_bias_test: true
```

训练使用 `use_lidar_bias_train`，验证和测试使用 `use_lidar_bias_test`。LiDAR 单帧点云按各 context 的相机时间和位姿投影到裁剪后的图像，在深度预测器完成 `to_disparity`、截断和逆深度转换后，硬替换有效像素的最终 Z-depth，再生成高斯。`context_depth` 与深度一致性损失也使用替换后的深度。

仅替换 near/far 范围内的有效 LiDAR 点；同像素多点取最近点，空缺处保留网络预测，不插值或补成稠密深度。替换处不向网络深度分支回传梯度，未替换处保留原梯度。此功能不新增网络参数，可加载已有 checkpoint。

例如，只在训练时使用 LiDAR：

```bash
python -m src.main +experiment=stage1 model.encoder.use_lidar_bias_train=true model.encoder.use_lidar_bias_test=false
```

非 LiDAR 数据集的 `acid`、`dtu` 配置默认关闭两个开关。深度诊断脚本默认检查未替换的网络预测，增加 `--lidar-bias` 可检查硬替换后的结果。

### Stage 3 使用六路相机

`config/experiment/stage3.yaml` 的 `dataset.camera_names` 已包含 nuScenes 全部六路相机。每路分别生成 `[t-1, t+1] → t` 时序样本，每个样本仍为 2 个 context 和 1 个 target；六路数据共同参与训练。`v1.0-mini` 下训练样本为 1842 个，验证/测试各 462 个。固定训练步数下，每路相机的覆盖次数会随样本总量增加而减少。

每帧使用该相机 `sample_data` 对应的内参、传感器外参和拍摄时刻 ego pose；裁剪后同步调整内参。LiDAR 使用对应 sample 的 `LIDAR_TOP`，由 nuScenes SDK 根据 LiDAR 与相机各自的标定和 ego pose 投影，再映射到裁剪后的图像。动态掩码也使用对应相机投影。多相机输出名称包含相机名，避免相互覆盖。

```bash
python -m src.main +experiment=stage3
# 恢复单路 CAM_FRONT
python -m src.main +experiment=stage3 dataset.camera_names=null dataset.camera_name=CAM_FRONT
```

其他 stage 默认保留单相机行为，也可以通过 `dataset.camera_names` 指定相机列表。

### LiDAR 监督截断前的逆深度

nuScenes 实验的 loss 列表已加入 `lidar_inverse_depth`：

```yaml
loss:
  lidar_inverse_depth:
    weight: 1.0
    beta: 0.01
```

该项在有效 context LiDAR 像素上比较 `粗逆深度 + to_disparity 修正值` 与 `1 / LiDAR Z-depth`，使用 Smooth L1，按有效像素和深度样本数归一化。预测取自 `clamp` 和 LiDAR 硬替换之前，不 detach，因而越界预测仍能获得梯度。无有效点时返回零；空缺、非有限或超出 near/far 的 LiDAR 点不参与。

这项监督独立于硬替换开关，只要求 `dataset.load_lidar_depth=true`。`weight=0` 可关闭它的贡献。stage0、stage2 当前冻结 `depth_predictor.to_disparity`，所以该项不会在这两个配置中更新深度分支；需要训练深度时，应从 `frozen_params` 中去掉该项。

## 9. 实验记录

PSNR/SSIM 越高越好，LPIPS 越低越好。

| 实验 | 增加loss | 本阶段训练模块 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 推理展示（scene-0103） |
| --- | :---: | --- | ---: | ---: | ---: | :---: |
| baseline | — | 无（直接推理） | 17.3907 | 0.4055 | 0.3885 | <img src="outputs/test/baseline/scene-0103_87e772078a494d42bd34cd16172808bc/color/000002.png" width="240"> |
| Stage 0 | — | 无（直接推理） | 17.4256 | 0.4068 | 0.3872 | <img src="outputs/test/baseline/scene-0103_87e772078a494d42bd34cd16172808bc/color/000002.png" width="240"> |
| Stage 1 | cross_visual_depth | `depth_predictor.to_disparity` | 20.4386 | 0.5668 | 0.3083 | <img src="outputs/test/stage1/scene-0103_87e772078a494d42bd34cd16172808bc/color/000002.png" width="240"> |
| Stage 2 | context_rgb | `depth_predictor.lidar_gaussians` | 20.8392 | 0.5738 | 0.2939 | <img src="outputs/test/stage2/scene-0103_87e772078a494d42bd34cd16172808bc/color/000002.png" width="240"> |
| Stage 3 | cross_visual_depth, lidar_inverse_depth, context_rgb | `depth_predictor.depth_head_lowres`、`depth_predictor.refine_unet`、`depth_predictor.to_disparity`、`depth_predictor.to_gaussians` | 21.4204 | 0.6007 | 0.2771 | <img src="outputs/test/stage3/scene-0103_87e772078a494d42bd34cd16172808bc/color/000002.png" width="240"> |
| Stage 4 | cross_visual_depth, lidar_inverse_depth, context_rgb | `all` | **21.7153** | **0.6168** | **0.2585** | <img src="outputs/test/stage4/scene-0103_CAM_FRONT_87e772078a494d42bd34cd16172808bc/color/000002.png" width="240"> |


| 实验 | 增加loss | 本阶段训练模块 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | :---: | --- | ---: | ---: | ---: |
| baseline | — | 无（直接推理） | 20.1596 | 0.5534 | 0.3652 |
| stage 5 | cross_visual_depth, lidar_inverse_depth, context_rgb | `all` | 19.5265 | 0.5118 | 0.3511 |


## 10. 致谢与引用

本项目基于 [MVSplat](https://github.com/donydchen/mvsplat) 开发。原项目的安装、预训练模型和基础训练方法以官方说明为准。

```bibtex
@article{chen2024mvsplat,
  title   = {MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images},
  author  = {Chen, Yuedong and Xu, Haofei and Zheng, Chuanxia and Zhuang, Bohan and Pollefeys, Marc and Geiger, Andreas and Cham, Tat-Jen and Cai, Jianfei},
  journal = {arXiv preprint arXiv:2403.14627},
  year    = {2024}
}
```

## 11. 版权声明

Copyright © 2026 张雪丽. All rights reserved.

未经著作权人书面许可，不得复制、修改、发布、再许可或用于商业及学术成果申报。
