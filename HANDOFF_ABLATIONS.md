# full_shared 接力棒：数据扩展与消融实验
## 1. 分支说明
main分支是完整实验方案，baseline_depth_error分支是仅把mvsplat应用场景迁移后的代码，可以用于做消融实验，full_shared分支内容和main是一样的，其余分支是尝试过的方案暂时不管。
等你搭好环境和跑通评测时，我会把你加为这个项目的contributor，修改可以push到full_shared分支上，main分支用来保存可行版本，如果遇到错误可以回退。
看到markdown中没有说清楚的地方可以直接在群里问，不要猜想。

## 2. 消融实验步骤
1.检查代码逻辑是否有明显错误，可以边做实验边做检查
2.对比full_shared和baseline_depth_error除了新增加的loss、lidar bias是否还有别的不同，这些不同一定要向full_shared对齐
3.扩大数据集，现在还是mini数据集，你需要在nuscenes官方网站上找到完整数据集下载和使用，应该是要注册一个账号，数据集用于学习是可以的。
4.建立同预算纯视觉微调对照，再解释LiDAR监督、硬替换和context重建的贡献。
5.指标：psnr、lpips、ssim、推理时间、训练时间、lidar_loss
要留意深度是否截断


## 3. 已有结果：参考，不是完整数据结论
当前配置为 mini、176×320。
# CAM_6
| 实验 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | steps |
| --- | ---: | ---: | ---: | ---: |
| baseline_depth_error，直接推理 | 20.0636 | 0.5540 | 0.3676 | 0 |
| baseline，直接推理 | 20.1596 | 0.5534 | 0.3652 | 0 |
| full_shared | 20.4492 | 0.5537 | 0.3213 | 40000(训练60000步，取的40000步) |
CAM_6 表示六路相机样本共同训练：每个样本仍是同一相机 `[t-1,t+1] → t`，不是六相机联合融合。

## 4. 完整消融流程
数据从 mini 扩大到 trainval

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

## 5. 第一轮必须完成：8组训练、16组评测

正式性能对照固定 full_shared 架构、全参数训练、相同 RE10K 初始化，各组独立训练 **60000 optimizer steps**。确认某项功能是否生效不要求训练60000步，先按下列验证层级执行。三个训练因素做完整的二元组合：

- D：`lidar_inverse_depth` 有/无；有时 weight=1.0、beta=0.01。
- C：`context_rgb` 有/无；有时 weight=1.0。
- B_train：`model.encoder.use_lidar_bias_train` 为 true/false。

每个训练 checkpoint 分别以 `use_lidar_bias_test=false` 和 `true` 评测，共8次训练、16次评测/seed。测试开关不会改变已学权重，不需要为它再训练一遍。若开启在线验证，其结果会受 test 开关影响，因此本轮保持在线验证关闭、统一离线评测。

### 功能验证与正式实验分开执行

- **配置检查，不训练**：用 `--cfg job` 确认 loss 列表、训练/测试硬替换开关、数据版本及输出路径。
- **前向与数据检查，1～数个 batch**：确认 split 非空、数据形状正确；有效 LiDAR 像素在 bias 开启时被替换，关闭时保留视觉预测；无效点不替换。同一权重切换测试开关直接评测即可，不重训。
- **反向与开关检查，1～数个 optimizer step**：确认启用的 loss 参与总损失和预期参数梯度，删除的 loss 不再执行其专用路径。仅观察 loss 数值不足以证明梯度路径正确。
- **短程稳定性，按需几十至几百步**：检查 NaN、显存、数据循环及运行错误；必要时延长。这类实验标记为功能验证，不据此宣称最终质量提升或收敛结论。
- **正式消融，统一60000步**：功能验证通过后再启动，用于 PSNR/SSIM/LPIPS 等最终性能对照。已有匹配条件的 checkpoint 可复用，不重复训练。

短程训练覆盖 `trainer.max_steps` 会改变 OneCycleLR 调度；其结果不能当作正式60000步训练的等价片段。计时也可用独立短程运行完成，但必须满足预热和测量窗口要求；完整60000步训练总时长仅从正式训练累计。

### 5.1 训练矩阵

基础 loss 统一保留 target MSE + 0.05 LPIPS（当前 full_shared 中的原始 RGB 目标）。执行前核对 baseline_depth_error 的历史配置；若“原 loss”与此不同，先记录差异并固定同一基础目标，不允许 baseline 与其他组偷偷使用不同 LPIPS 权重。不要把其他历史阶段的 cross_visual_depth 等目标混入。

| ID | D | C | B_train | B_test评测 | 定义 |
| --- | --- | --- | --- | --- | --- |
| E000 | 无 | 无 | false | false / true | 原 RGB loss 全参数训练 baseline；test=false 是纯视觉基线 |
| E100 | 有 | 无 | false | false / true | 仅增加 LiDAR 深度监督 |
| E010 | 无 | 有 | false | false / true | 仅增加 context RGB |
| E110 | 有 | 有 | false | false / true | 两个新增 loss，无训练硬替换 |
| E001 | 无 | 无 | true | false / true | 仅增加训练硬替换 |
| E101 | 有 | 无 | true | false / true | 深度监督 + 训练硬替换 |
| E011 | 无 | 有 | true | false / true | context RGB + 训练硬替换 |
| E111 | 有 | 有 | true | false / true | 完整训练方案；test=true 对应 full_shared |

ID 三位依次表示 D、C、B_train。报告结果时增加后缀 `_test0` / `_test1`。例如 E000_test1 是纯视觉训练后仅推理注入 LiDAR，不能命名为纯视觉测试。

另保留 RE10K 未微调直接推理 R0 作为参考，不计入8组训练，不能替代 E000 的60000步 baseline。

建议先完成 E000/E111，再补齐其他6组；全部均为必做，不用逐项移除的少数组合替代完整矩阵。现有 mini 60000总步数训练中取出的40000步结果只作历史参考；新表统一使用60000步 checkpoint，mini 与 trainval 分表。

### 5.2 训练命令及各组覆盖

入口为 `config/experiment/full_shared.yaml`。完整数据离线评测必须先完成第4节 split 修复。

```bash
# E111：完整方案
python -m src.main +experiment=full_shared mode=train \
  dataset.version=v1.0-trainval dataset.load_lidar_depth=true \
  checkpointing.load=checkpoints/re10k.ckpt checkpointing.resume=false \
  trainer.max_steps=60000 seed=111123 train.run_validation=false \
  model.encoder.use_lidar_bias_train=true \
  model.encoder.use_lidar_bias_test=true \
  wandb.name=E111_s111123 \
  hydra.run.dir=outputs/ablations/trainval/E111_s111123
```

其余组替换运行名称和目录，并增加以下覆盖；删除 loss 用 Hydra 的 `~` 语法，避免仅置零却仍执行 context 渲染：

```text
E000: '~loss.lidar_inverse_depth' '~loss.context_rgb' model.encoder.use_lidar_bias_train=false
E100: '~loss.context_rgb' model.encoder.use_lidar_bias_train=false
E010: '~loss.lidar_inverse_depth' model.encoder.use_lidar_bias_train=false
E110: model.encoder.use_lidar_bias_train=false
E001: '~loss.lidar_inverse_depth' '~loss.context_rgb'
E101: '~loss.context_rgb'
E011: '~loss.lidar_inverse_depth'
E111: 无额外覆盖
```

所有组可统一保留 `dataset.load_lidar_depth=true`，便于同一权重的 test=true 评测和数据检查；E000/E010 训练时不使用 LiDAR 数值。若另测纯视觉部署效率，应关闭其 LiDAR 加载，单独注明数据预处理条件。先用 `--cfg job` 核对每组最终 loss 列表和开关，再做短程冒烟验证。

### 5.3 每个 checkpoint 评测两次

对每组使用对应训练配置（包括相同的 loss 删除覆盖）及同一个60000步 checkpoint。下面以 E111_test0 为例；第二次改 test 开关为 true，并将所有输出名称改为 test1：

```bash
python -m src.main +experiment=full_shared mode=test \
  dataset.version=v1.0-trainval dataset.load_lidar_depth=true \
  checkpointing.load=/absolute/path/to/E111_step60000.ckpt \
  model.encoder.use_lidar_bias_train=true \
  model.encoder.use_lidar_bias_test=false \
  wandb.name=E111_test0_s111123 \
  test.compute_scores=true test.save_image=false \
  test.output_path=outputs/ablations/evaluation \
  hydra.run.dir=outputs/ablations/evaluation_runs/E111_test0_s111123
```

训练开关在 eval 模式不生效，测试由 use_lidar_bias_test 决定。所有16组使用同一评测 split、窗口和指标。输出目录为 `test.output_path / wandb.name`；禁止覆盖旧结果。

### 5.4 需要回答的问题

- 基础微调收益：R0 对比 E000_test0。
- D贡献：固定 C、B_train、B_test，对比 D有/无的配对结果。
- C贡献：固定 D、B_train、B_test，对比 C有/无的配对结果。
- 训练硬替换贡献：固定 D、C、B_test，对比 B_train真/假。
- 测试硬替换贡献：同一个 checkpoint 的 test1 与 test0 配对。
- 训练/测试失配：比较 B_train/B_test 的 false/false、false/true、true/false、true/true，检查是否依赖训练时的锚点。
- 交互作用：检查 D或C的收益是否随训练硬替换改变，不能只用 E111 对比 E000 就归因于某一组件。

## 6. 统一训练与评测口径

- 所有训练组固定60000步、batch size=2、lr=1e-4、相同 seed、初始化及调度；不要从 E111 继续训练其他消融。当前 cosine_lr=true 使用 OneCycleLR（total_steps=max_steps+10、pct_start=0.01），此路径不使用 warm_up_steps=500。
- 当前 YAML max_steps=60000。cosine 下“60000 调度中的第40000步”不等于“40000 总步数调度末尾”。历史曲线需补档；新对照必须统一完整调度。
- 完整数据更大，相同步数覆盖次数更少。第一轮后统一检查开发集曲线和累计样本覆盖，若需延长，统一增加关键对照预算并重新说明调度，不能只延长偏好的模型。
- 固定步数结果为主；若报告最佳 checkpoint，统一开发集指标和选择规则，最终报告集不用于选模。
- 优先单 seed 筛选，预算允许后 E111/E000 和关键消融至少补3个 seed，报告均值及波动。
- PSNR/SSIM/LPIPS 三项都报告；当前 test_step 计算整图指标，训练的动态 mask 不意味着测试指标也做了 mask。静态区域分数如新增，单独命名并明确算法；LPIPS 不能简单将动态区涂黑后当作标准指标。
- 增加逐相机和逐场景汇总，保存每样本身份和指标；先核查当前分数文件是否含稳定样本ID，缺少则补齐再做配对分析。
- 深度报告硬替换前网络预测的 AbsRel、RMSE 等；硬替换后 LiDAR 点误差可能被人为置零，不能代表网络学到了深度。尽量留出未参与输入/监督的点做独立几何验证。
- 固定场景输出 GT/E111/E000/消融及误差图，关注远处结构、深度边缘、遮挡、细杆和运动区域。保存相同相机和窗口，禁止只挑有利样本。

### 6.1 训练与推理时长：必测项

正式8组训练均记录训练耗时，16组测试均记录推理耗时，与质量指标一起交付。功能验证无需跑满60000步；稳定阶段速度可独立短程测量，完整训练总时长随正式训练记录。

**训练：**

- 记录完整60000步训练的实际墙钟时间（小时），包括数据读取、前向、loss、反向、优化器更新和按统一频率保存 checkpoint。启动/数据索引/加载权重时间另外记录；恢复训练需累加各段实际运行时间，不含停机等待。
- 另测稳定阶段平均 ms/optimizer step、samples/s，以及峰值显存。建议前100步预热，随后至少计时500步；所有组使用相同区间、batch size、梯度累积和 checkpoint 频率。计时窗口是否包含保存 checkpoint 要注明。
- context loss 增加的 context 渲染属于训练代价，应纳入。关闭项从 loss 列表删除，不能仅设 weight=0 后将无效计算算作该组必要开销。

**推理：**

- 模型时间：分别测 encoder（包含 Gaussian 构建）、decoder（渲染一个 target）及编码到渲染完成的总延迟，单位 ms/sample；batch size=1，一个 sample 为同相机两张 context 输入、一个 target 输出。
- 完整流程：另测包含图像读取、LiDAR 读取与投影、数据预处理、CPU→GPU传输和模型执行的耗时/吞吐，写清边界。单独列出一次性模型加载和数据索引构建时间。无需 LiDAR 的 test0 计时运行应关闭 LiDAR 加载；保留加载的统一数据质量评测结果不能当作纯视觉部署开销。
- 计时单独运行，关闭 PSNR/SSIM/LPIPS、保存图片和视频，不把评测器计算或磁盘输出算作模型推理。建议先预热20个样本，随后固定相同的至少200个窗口，重复3次，报告均值、中位数、P95和吞吐。
- 单目标场景的推理 FPS 可由稳定总推理时间换算，不能只用 decoder 时间宣称整套模型 FPS；也不能将六路独立样本吞吐解释成六相机同步重建 FPS。

**计时实现与环境：**

当前 `src/misc/benchmarker.py` 使用 CPU wall clock 包围 GPU调用，未显式同步 CUDA；现有 `benchmark.json` 仅作参考，不能直接认定为准确 GPU 时长。下一位同学需实现专用计时：模型阶段使用 CUDA Events 并在读取结果前同步；完整流程/训练窗口用 perf_counter，边界同步 CUDA。批量计时窗口避免每步强制同步扰动吞吐。阶段拆分计时与总吞吐可以分开测量。

固定 GPU型号/数量、CPU、精度、PyTorch/CUDA版本、分辨率、batch size、worker数和后台负载，计时不要与其他训练并行。数据缓存、是否含动态 mask 生成及 LiDAR 预处理必须注明。短程效率测量和60000步实际训练总耗时分别报告，不用短程外推值冒充完整训练实测值。

| 训练组/seed | 训练总时长 h | 启动时间 s | 稳态 ms/step | samples/s | 峰值显存 GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 待填写 | | | | | |

| 组别/test开关 | encoder ms | decoder ms | 模型总延迟 ms（均值/中位/P95） | 完整流程 ms/sample | 流程吞吐 samples/s | 峰值显存 GiB |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 待填写 | | | | | | |

## 7. 第二轮：主结论成立后再做

1. LiDAR 稀疏度：100%/50%/25%/10% 点。先做同一 E111 权重的推理敏感性；重新训练的稀疏度实验另表。固定抽样种子、每帧点子集，所有方法共享；保留独立完整点集作几何评测，避免指标随抽样改变。
2. 深度权重：0.1/1/10，或 context 权重0.1/0.5/1。一次只改变一类权重；不是第一轮必做项。
3. 若发现硬替换使深度边缘或遮挡处变差，先量化失败，再考虑置信度融合或属性联合修正。不要为了创新堆模块。
4. 外部对照：调研 DrivingForward、ADGaussian 及近期公开方法的可复现性。明确相机数、LiDAR、已知位姿、动态标注和分辨率差异，不能直接抄不同协议的论文指标做排名。
5. 预算允许时在第二数据集或第二前馈骨干验证迁移；不作为启动第一轮的前置条件。

## 8. 交付清单与记录模板

每组交付：git commit、完整 resolved config（含 .hydra/config.yaml 和 overrides）、环境版本、启动命令、seed、split 场景列表、样本数、初始化和最终 checkpoint 路径、训练日志、原始分数JSON、训练/推理原始计时记录与统计脚本、汇总表与固定可视化。先确认历史 full_shared 40000 checkpoint 的实际路径，本文不虚构路径。

| ID/test开关/seed | commit | 数据/split | steps/调度 | checkpoint | PSNR | SSIM | LPIPS | 原始深度误差 | 输出目录 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 待填写 | | | | | | | | | |

结束第一轮时回答：完整数据上 E111 是否优于 E000？收益主要来自哪一项？去掉测试 LiDAR 后还保留多少？六路相机是否一致？几何改善与渲染改善是否一致？无法支持的结论如实保留。
