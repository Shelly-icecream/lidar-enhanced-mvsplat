# full_shared 接力棒：数据扩展与消融实验

交接日期：2026-10-29。本文只建立实验计划，尚未执行合并、修改数据划分或启动训练。

## 1. 目标与代码版本

下一阶段以 full_shared 为主线：先扩大数据并建立同预算纯视觉微调对照，再解释 LiDAR 监督、硬替换和 context 重建的贡献。暂不继续扩展拆头和交替训练。

交接时的代码状态：

- full_shared：`01eedc6`，本地记录的 origin/full_shared 同提交。
- main：`96f7a71`；与 full_shared 已分叉，不能假设能 fast-forward。
- 当前工作分支：pruned_split_alternating；README.md 有未提交的 full_shared 指标补录。
- **full_shared 的实验入口是 `config/experiment/stage5.yaml`**，该版本没有 stage4.yaml。当前工作分支的 stage5 则是拆头交替模型，不能混用。
- full_shared/stage5 显式保留 costvolume U-Net 配置，完整共享、全参数训练。不要把 pruned 分支的实现或配置当作它的等价替代。
- 合并后逐项确认网络、配置与 full_shared 一致；不能只因为命令叫 stage5 就认为模型一致。

## 2. 如何把 full_shared 合并到 main

下面是供交接者执行的步骤，本文没有代为执行。先提交当前交接材料，避免切分支时丢失 README 的实验数据；只提交这两个文件，不使用 git add .。

```bash
git add README.md HANDOFF_ABLATIONS.md
git commit -m "docs: record full_shared results and ablation handoff"
git rev-parse HEAD
```

记下最后输出的提交号，以下以 `<HANDOFF_COMMIT>` 表示。

```bash
git fetch origin
git switch full_shared
git merge --ff-only origin/full_shared
git switch main
git merge --ff-only origin/main
git branch backup/main-before-full-shared-20261029
git merge --no-ff --no-commit full_shared
```

若 fast-forward 更新失败，先检查本地与远程差异，不要 reset 或 force push。若合并冲突，用 `git status` 查看，逐文件处理后 `git add <文件>`；README 整合两侧内容，模型相关冲突以保留 full_shared 实验实现为目标，同时检查 main 独有改动。不要整仓库无条件选 ours/theirs。需要放弃此次未完成合并可用 `git merge --abort`。

```bash
git diff --cached --stat
git diff --cached
# 完成配置核对与下文的小规模冒烟验证后：
git commit -m "Merge full_shared into main"
git cherry-pick <HANDOFF_COMMIT>
# cherry-pick 若遇 README 冲突，保留实验结果表和本交接文件，再执行：
# git add README.md HANDOFF_ABLATIONS.md
# git cherry-pick --continue
git status
git push origin main
```

若仓库要求 PR，改为在更新后的 main 上创建 `merge/full-shared` 分支，完成上述合并、检查和交接提交后推送该分支，通过 PR 合入 main。不要绕过保护规则。合并不等于 full_shared 的每个文件都会覆盖 main，必须审阅最终 diff。

## 3. 已有结果：参考，不是完整数据结论

来自交接者 README 的 CAM_6 记录。当前配置为 mini、176×320；历史每组实际 resolved config、checkpoint、评测目录仍需补档，不能凭当前 YAML 反推历史配置。

| 实验 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | steps |
| --- | ---: | ---: | ---: | ---: |
| baseline_depth_error，直接推理 | 20.0636 | 0.5540 | 0.3676 | 0 |
| baseline，直接推理 | 20.1596 | 0.5534 | 0.3652 | 0 |
| full_shared | 20.4492 | 0.5537 | 0.3213 | 40000 |
| pruned | 20.1844 | 0.5440 | 0.3340 | 40000 |
| pruned_split | 20.3112 | 0.5555 | 0.3351 | 40000 |
| pruned_split_alternating | 20.2357 | 0.5463 | 0.3362 | 40000 |
| pruned_split_alternating | 20.2359 | 0.5473 | 0.3348 | 60000 |

full_shared 相比 baseline：PSNR +0.2896 dB，LPIPS -0.0439，SSIM 基本持平。它是当前综合最优候选，不是所有指标最优。baseline 没有 nuScenes 微调，所以该差值不能全部归因于 LiDAR。两个 baseline 的实际差异需要查历史记录，不要自行合并。

CAM_6 表示六路相机样本共同训练：每个样本仍是同一相机 `[t-1,t+1] → t`，不是六相机联合融合。

## 4. 数据从 mini 扩大到 trainval

准备完整 `v1.0-trainval` 元数据及对应 samples 等数据，沿用 nuScenes 根目录结构与 SDK。训练覆盖：

```yaml
dataset:
  root: datasets/nuscenes
  version: v1.0-trainval
```

保持六路相机、176×320、near=1、far=80、frame_stride=1、keyframe 设置、动态 mask 和裁剪方式不变，先不要同时改分辨率。记录实际场景数、每路有效窗口数、跳过的样本数、LiDAR 有效点比例。

### 必须先修正评测划分

full_shared 的 `src/dataset/dataset_nuscenes.py::_build_index` 当前行为：mini 的 val/test 都选 mini_val；非 mini 的 train/val/test 分别选官方 train/val/test。**只把 version 改为 trainval 后执行 mode=test，会选择官方 test 场景，可能得到空数据；不能将空评测视为通过。** 官方 test 也不适合当前依赖动态 annotation 的本地评测流程。

下一位同学需要增加显式评测 split 配置，或明确让 trainval 的本地 test 使用有真值的 val 场景。使用官方 train 训练、官方 val 报告本地结果；若选择最佳 checkpoint，从 train 中另外按场景留出开发集，或预先固定步数，避免反复在最终报告集上调参。禁止随机按帧切分导致相邻时序窗口泄漏。

修复后先打印并核对场景列表、样本数和集合不相交，再用少量 batch 验证 train/离线 test。nuScenes 使用自己的时序采样逻辑，不要直接套用 RE10K 的 evaluation index JSON。

## 5. 第一轮必须完成的实验

所有训练组都使用 full_shared 同一架构、全参数训练、同一 RE10K 初始化和相同 RGB 基础 loss（target MSE + 0.05 LPIPS）。权重 D=LiDAR inverse-depth 1.0、C=context RGB 1.0；B=合法 LiDAR 像素的深度硬替换。

| ID | 组别 | D | B训练 | B测试 | C | 用途 |
| --- | --- | ---: | --- | --- | ---: | --- |
| R0 | RE10K 直接推理 | — | — | 关 | — | 域适配前参考，无需训练 |
| A0 | full_shared | 1 | 开 | 开 | 1 | 完整方案 |
| A1 | 纯视觉微调 | 0 | 关 | 关 | 0 | 最重要：排除普通微调收益 |
| A2 | 去掉深度监督 | 0 | 开 | 开 | 1 | 检查几何退化及监督贡献 |
| A3 | 去掉硬替换 | 1 | 关 | 关 | 1 | 只监督、不注入测量值 |
| A4 | 去掉 context loss | 1 | 开 | 开 | 0 | context 重建贡献 |
| T0 | A0 同权重无测试 LiDAR | 同A0 | 同A0 | 关 | 同A0 | 推理 LiDAR 依赖，无需重训 |

执行优先级：mini 上现有 A0 checkpoint 做 T0 → 完整数据 A1/A0 → A4 → A2 → A3。不同数据规模的结果分表，禁止把 mini T0 与完整数据 A0 直接计算差值。

关闭 loss 可通过 weight=0；保留 dataset.load_lidar_depth=true，尤其 A3 仍需要 LiDAR 监督。A1 若保留零权重 loss，仍会加载 LiDAR，但不能将其输入模型或用于非零训练项。weight=0 的 context loss 可能仍触发额外渲染；质量消融可用，效率对照需真正移除该 loss 并核对执行路径。

模板仅适用于合并后已核对的 full_shared 入口，完整数据离线测试须先修复第4节 split：

```bash
# A0；第一轮用固定 40000 步作为可比较的预算，并非保证完整数据充分收敛
python -m src.main +experiment=stage5 mode=train \
  dataset.version=v1.0-trainval \
  checkpointing.load=checkpoints/re10k.ckpt checkpointing.resume=false \
  trainer.max_steps=40000 seed=111123 \
  wandb.name=full_shared_A0_trainval_s111123 \
  hydra.run.dir=outputs/ablations/trainval/A0_s111123
```

其他组在同一命令中增加：

```text
A1: loss.lidar_inverse_depth.weight=0 loss.context_rgb.weight=0 model.encoder.use_lidar_bias_train=false model.encoder.use_lidar_bias_test=false
A2: loss.lidar_inverse_depth.weight=0
A3: model.encoder.use_lidar_bias_train=false model.encoder.use_lidar_bias_test=false
A4: loss.context_rgb.weight=0
```

离线测试模板，替换真实 checkpoint 路径与输出目录，带上该组相同的配置覆盖；T0 额外关闭 use_lidar_bias_test：

```bash
python -m src.main +experiment=stage5 mode=test \
  dataset.version=v1.0-trainval \
  checkpointing.load=/absolute/path/to/model.ckpt checkpointing.resume=false \
  test.compute_scores=true test.save_image=false \
  test.output_path=outputs/ablations/eval_A0 \
  hydra.run.dir=outputs/ablations/eval_A0/run
```

hydra.run.dir 不等于评测分数目录，评测还受 test.output_path 和运行名称控制。每组使用唯一输出目录，核对实际保存的 scores_all_avg.json，禁止覆盖旧结果。

## 6. 统一训练与评测口径

- 第一轮所有组固定相同步数（建议40000）、batch size=2、lr=1e-4、warmup=500、cosine、seed 和初始化；不要从已训练 A0 继续做删除组件的消融。
- 当前 YAML max_steps=60000。cosine 下“60000 调度中的第40000步”不等于“40000 总步数调度末尾”。历史曲线需补档；新对照必须统一完整调度。
- 完整数据更大，相同步数覆盖次数更少。第一轮后统一检查开发集曲线和累计样本覆盖，若需延长，统一增加关键对照预算并重新说明调度，不能只延长偏好的模型。
- 固定步数结果为主；若报告最佳 checkpoint，统一开发集指标和选择规则，最终报告集不用于选模。
- 优先单 seed 筛选，预算允许后 A0/A1 和关键消融至少补3个 seed，报告均值及波动。
- PSNR/SSIM/LPIPS 三项都报告；当前 test_step 计算整图指标，训练的动态 mask 不意味着测试指标也做了 mask。静态区域分数如新增，单独命名并明确算法；LPIPS 不能简单将动态区涂黑后当作标准指标。
- 增加逐相机和逐场景汇总，保存每样本身份和指标；先核查当前分数文件是否含稳定样本ID，缺少则补齐再做配对分析。
- 深度报告硬替换前网络预测的 AbsRel、RMSE 等；硬替换后 LiDAR 点误差可能被人为置零，不能代表网络学到了深度。尽量留出未参与输入/监督的点做独立几何验证。
- 固定场景输出 GT/A0/A1/消融及误差图，关注远处结构、深度边缘、遮挡、细杆和运动区域。保存相同相机和窗口，禁止只挑有利样本。
- 若报告效率，分开编码和渲染时间，写明设备、分辨率、batch、预热、是否包含 LiDAR 预处理；加载但不使用 LiDAR的 A1 不等于优化后的纯视觉部署耗时。

## 7. 第二轮：主结论成立后再做

1. LiDAR 稀疏度：100%/50%/25%/10% 点。先做同一 A0 权重的推理敏感性；重新训练的稀疏度实验另表。固定抽样种子、每帧点子集，所有方法共享；保留独立完整点集作几何评测，避免指标随抽样改变。
2. 深度权重：0.1/1/10，或 context 权重0.1/0.5/1。一次只改变一类权重；不是第一轮必做项。
3. 若发现硬替换使深度边缘或遮挡处变差，先量化失败，再考虑置信度融合或属性联合修正。不要为了创新堆模块。
4. 外部对照：调研 DrivingForward、ADGaussian 及近期公开方法的可复现性。明确相机数、LiDAR、已知位姿、动态标注和分辨率差异，不能直接抄不同协议的论文指标做排名。
5. 预算允许时在第二数据集或第二前馈骨干验证迁移；不作为启动第一轮的前置条件。

## 8. 交付清单与记录模板

每组交付：git commit、完整 resolved config（含 .hydra/config.yaml 和 overrides）、环境版本、启动命令、seed、split 场景列表、样本数、初始化和最终 checkpoint 路径、训练日志、原始分数JSON、汇总表与固定可视化。先确认历史 full_shared 40000 checkpoint 的实际路径，本文不虚构路径。

| ID/seed | commit | 数据/split | steps/调度 | checkpoint | PSNR | SSIM | LPIPS | 原始深度误差 | 输出目录 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 待填写 | | | | | | | | | |

结束第一轮时回答：完整数据上 A0 是否优于 A1？收益主要来自哪一项？去掉测试 LiDAR 后还保留多少？六路相机是否一致？几何改善与渲染改善是否一致？无法支持的结论如实保留。
