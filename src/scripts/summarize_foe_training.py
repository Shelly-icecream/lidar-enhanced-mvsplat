"""Compare saved pilot renders on identical masks across ALL checkpoints."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, distance_transform_edt, binary_erosion

ROOT=Path(__file__).resolve().parents[2]/"outputs/foe_training"
OLD_ROOT=ROOT
dynamic="--dynamic-training" in sys.argv
if dynamic: ROOT=ROOT.parent/"foe_training_dynamic"
steps=[0,10,50,100]
rows=[]

def read(path): return np.asarray(Image.open(path)).astype(np.float32)/255
def grad(im):
    x=im.mean(-1)
    return np.hypot(sobel(x,axis=0)/8,sobel(x,axis=1)/8)

for group in ["A","B"]:
    for sample in range(2):
        folders=[ROOT/group/f"step_{s:04d}"/f"sample_{sample}" for s in steps]
        mask=np.logical_and.reduce([read(f/"static_overlap.png")[...,0]>.5 for f in folders])
        # Keep edge convolution support away from masked boundaries.
        mask=binary_erosion(mask,iterations=2)
        target=read(folders[0]/"target.png"); gt=grad(target)
        fig,axes=plt.subplots(2,5,figsize=(16,7))
        axes[0,0].imshow(target); axes[0,0].set_title("Target")
        axes[1,0].imshow(mask,cmap="gray"); axes[1,0].set_title("Fixed static overlap")
        for idx,(step,folder) in enumerate(zip(steps,folders)):
            rgb=read(folder/"combined.png"); g=grad(rgb)
            split=[read(folder/f"context{s}_render.png") for s in range(2)]
            grads=[grad(x) for x in split]
            edges=[(x>np.quantile(x[mask],.8))&mask for x in grads]
            edge_dist=np.mean([distance_transform_edt(~edges[1])[edges[0]].mean(),distance_transform_edt(~edges[0])[edges[1]].mean()])
            row=dict(group=group,sample=sample,step=step,pixels=int(mask.sum()),gradient_ratio=float(g[mask].mean()/gt[mask].mean()),gradient_error=float(np.abs(g-gt)[mask].mean()),mse=float(((rgb-target)**2)[mask].mean()),split_rgb_difference=float(np.abs(split[0]-split[1])[mask].mean()),split_edge_distance_px=float(edge_dist))
            rows.append(row)
            axes[0,idx+1].imshow(rgb); axes[0,idx+1].set_title(f"Step {step}, gradient {row['gradient_ratio']:.2f}")
            diff=np.where(mask,np.abs(split[0]-split[1]).mean(-1),np.nan)
            axes[1,idx+1].imshow(diff,cmap="magma",vmin=0,vmax=.3); axes[1,idx+1].set_title("Split difference (fixed mask)")
        for ax in axes.flat: ax.axis("off")
        fig.suptitle(f"Group {group}, validation sample {sample} | fixed mask across checkpoints")
        fig.tight_layout(); fig.savefig(ROOT/f"{group}_sample{sample}_comparison.png",dpi=140); plt.close(fig)
        # Depth panels share scales across time; geometry metrics are not ground-truth errors.
        base=np.load(folders[0]/"diagnostics.npz")
        fig,axes=plt.subplots(3,4,figsize=(13,9))
        for idx,(step,folder) in enumerate(zip(steps,folders)):
            arrays=np.load(folder/"diagnostics.npz")
            for ax,arr,title,lo,hi,cmap in [
                (axes[0,idx],arrays["depth"][0],f"Step {step}: context0 depth",1,100,"viridis"),
                (axes[1,idx],arrays["depth"][0]-base["depth"][0],"Depth change",-20,20,"coolwarm"),
                (axes[2,idx],arrays["disparity_residual"][0],"Inverse-depth residual",-.03,.03,"coolwarm")]:
                img=ax.imshow(arr,vmin=lo,vmax=hi,cmap=cmap); ax.set_title(title); ax.axis("off"); fig.colorbar(img,ax=ax,shrink=.65)
        fig.tight_layout(); fig.savefig(ROOT/f"{group}_sample{sample}_depth.png",dpi=120); plt.close(fig)

(ROOT/"fixed_mask_metrics.json").write_text(json.dumps(rows,indent=2))
for group in ["A","B"]:
    for step in [0,100]:
        selected=[r for r in rows if r["group"]==group and r["step"]==step]
        print(group,step,{k:round(float(np.mean([r[k] for r in selected])),5) for k in ["gradient_ratio","mse","split_edge_distance_px","split_rgb_difference"]})

raw=json.loads((ROOT/"metrics.json").read_text())
def avg(source,group,step,key):
    return float(np.mean([r[key] for r in source if r["group"]==group and r["step"]==step]))
lines=["# 无 LiDAR 的训练前后诊断与 A/B 小规模对照", "",
"## 结论", "",
"两组都完成 100 步训练。当前小样本不能证明弱视差是训练变糊的原因：较强横向平移的 B 组同样出现细节梯度下降。A 组深度漂移及分开渲染的不一致增加，但并未形成‘只有 A 退化、B 稳定’的结果。", "",
"## 实际运行设置", "",
"- 使用仓库历史提交 `660f49c` 的原版 src/config，隔离在 original/，不读取 adaptation 配置，也不使用当前修改后的 encoder。完整 encoder 权重严格加载 re10k.ckpt。",
"- 不读取 LiDAR 点云，不生成/使用 LiDAR 深度、bias、adapter 或深度损失。nuScenes 3D 物体框仅用于诊断遮罩，不用于训练。",
"- 256×256，深度范围 1–100，128 个逆深度候选；全 encoder 参数可训练；MSE + 0.05 VGG LPIPS；Adam，原版 300001 步级别 OneCycle 调度的前 100 步，实际学习率约 8e-6；梯度裁剪 0.5。",
"- 两组同一初始化、随机种子、步数和 batch size=1。不是原版 batch=14 的完整训练复现。",
"- 从 95 个非停车候选中选出 6 对无重复 token 的三元组；每组 4 个训练、2 个固定评估。保存 0/10/50/100 步诊断与最终模型。",
"- A: 横向平移/总平移 <0.10、中心 30m/50m 投影差 <0.5 特征像素；B: 比率 >0.20、投影差 >0.5。匹配总位移、图像纹理强度与静态遮罩占比。",
"- **没有按真实深度匹配两组**；B 组评估样本都来自 scene-0916，存在场景内容混杂。该实验是探索性对照，不能当作严格受控因果实验。", "",
"## 训练前后结果", "",
"下表为每组两个评估三元组的等权均值。RGB 指标用全部四个检查点共同满足静态且两份 alpha>0.5 的区域，并腐蚀 2 像素后计算，使用保存的 8-bit 图像。", "",
"| 指标（0 → 100 步） | A 弱视差 | B 较强横向平移 |", "|---|---:|---:|"]
for title,source,key in [("细节梯度/目标梯度",rows,"gradient_ratio"),("静态共同区域 MSE",rows,"mse"),("两份渲染边缘距离代理（像素）",rows,"split_edge_distance_px"),("两份 RGB 平均绝对差",rows,"split_rgb_difference"),("投影 3σ 半径中位数的均值（像素）",raw,"projected_radius_median"),("中心深度相对初始值的平均变化（深度单位）",raw,"depth_change_center"),("外围深度相对初始值的平均变化（深度单位）",raw,"depth_change_outer"),("全体 Gaussian 平均 opacity",raw,"opacity_mean"),("静态目标平均合并 alpha",raw,"alpha")]:
    vals=[f"{avg(source,g,0,key):.4f} → {avg(source,g,100,key):.4f}" for g in ["A","B"]]
    lines.append(f"| {title} | {vals[0]} | {vals[1]} |")
lines += ["", "## 如何解释", "",
"- 细节梯度下降意味着这个指标下变平滑，不等价于主观画质一定全面变差；MSE 同时下降，说明像素误差改善可以伴随细节变软。梯度也会受对比度和伪影影响。",
"- A 组中心深度变化大于外围，但这是相对初始预测的漂移，不是真实深度误差。天空未排除，也可能影响中心均值。",
"- 边缘距离是阈值边缘的最近距离代理，没有建立真实对应关系，可能受纹理/对比度变化影响，不能直接当成几何重投影误差。",
"- 投影半径由 JΣJᵀ 的最大特征值估算，筛选目标画面内、静态像素处、1–100 深度且 opacity>0.05 的 Gaussian；不是 CUDA rasterizer 的精确 footprint，不保证可见且不做逐 Gaussian 因果归因。",
"- 固定双 alpha 遮罩很保守，有些建筑和路面被排除，指标只代表留下的区域。两组深度与场景内容没有充分配平。",
"- 单种子、每组仅两个评估样本、100 步，且训练评估可能同场景；不能据此否定弱视差的影响，也不能确认它是主因。", "",
"## 查看文件", "",
"- `A_sample0_comparison.png` 等 4 张图：目标图、0/10/50/100 步合并图；下排为固定静态共同遮罩及其中的两份渲染差。白色差图背景为未参与比较的区域。",
"- `A_sample0_depth.png` 等 4 张图：相同色标下的 context0 深度、相对初始深度变化和逆深度残差。",
"- `A/step_0100/sample_0/` 等目录：两个 context 分开渲染、合并 RGB、分开/合并 alpha、静态重叠遮罩、完整深度/残差/投影尺寸/opacity 数组。",
"- `selection.json`：每组训练前四条和评估后两条，完整 token 和匹配特征；`settings.json`、`encoder.yaml` 为实验配置。",
"- `fixed_mask_metrics.json` 是跨所有检查点固定遮罩的最终图像指标；`metrics.json` 为推理现场指标（其中 RGB 指标遮罩随检查点改变，不用于主表 RGB 比较）。",
"- `A_final.pt`、`B_final.pt` 为最终 encoder；`A_training.json`、`B_training.json` 为训练曲线；`run.log` 为日志。", "",
"## 复现", "", "从仓库根目录运行（需要 CUDA 和可用的原版快照）：", "", "```bash", "MKL_THREADING_LAYER=GNU MPLCONFIGDIR=/tmp/mvsplat-mpl /home/xixiangtang/miniconda3/envs/mvsplat/bin/python src/scripts/diagnose_foe_training.py --steps 100", "MKL_THREADING_LAYER=GNU MPLCONFIGDIR=/tmp/mvsplat-mpl /home/xixiangtang/miniconda3/envs/mvsplat/bin/python src/scripts/summarize_foe_training.py", "```", ""]
(ROOT/"REPORT.md").write_text("\n".join(lines))
if dynamic:
    comparisons=[]
    for group in ["A","B"]:
        for sample in range(2):
            paths=[root/group/f"step_{step:04d}"/f"sample_{sample}" for root in [OLD_ROOT,ROOT] for step in steps]
            common=binary_erosion(np.logical_and.reduce([read(p/"static_overlap.png")[...,0]>.5 for p in paths]),iterations=2)
            for label,root in [("unmasked",OLD_ROOT),("dynamic_masked",ROOT)]:
                for step in [0,100]:
                    path=root/group/f"step_{step:04d}"/f"sample_{sample}"
                    im=read(path/"combined.png"); target=read(path/"target.png")
                    comparisons.append(dict(group=group,sample=sample,run=label,step=step,pixels=int(common.sum()),gradient_ratio=float(grad(im)[common].mean()/grad(target)[common].mean()),mse=float(((im-target)**2)[common].mean())))
    (ROOT/"mask_vs_unmasked_common_metrics.json").write_text(json.dumps(comparisons,indent=2))
    lines=["# 动态 mask 训练重跑（无 LiDAR）", "",
        "使用与上次完全相同的 A/B 三元组和训练/评估划分、初始化、100 步、batch=1、原版全参数训练。结果独立保存，未覆盖原实验。",
        "", "## Mask 与损失", "",
        "- 沿用项目动态属性：vehicle.moving、pedestrian.moving、cycle.with_rider；投影 3D 标注框，扩大 15%，至少 8 像素。仅用标注与相机位姿，不读取点云或 LiDAR 深度。",
        "- MSE 只在静态 target 像素统计并归一化。LPIPS 两侧的动态区域都填同一个 0.5 常数，动态 RGB 不参与差异。对每组四个训练样本检查损失对动态 target 预测像素的梯度，必须严格为零。",
        "- 原始 context RGB 不擦除，动态物体仍可能影响特征或投影到静态区域；LPIPS 有空间感受野。因此排除的是直接动态区域监督，不能宣称消除了所有动态干扰。",
        "- 每个 context/target 的静态 mask 和红色动态 overlay 见 masks/；覆盖比例见 mask_audit.json。",
        "", "## 新实验训练前后", "", "指标基于新实验所有检查点共同的静态双 alpha 有效区域；与旧实验直接比较需看下表。", "",
        "| 指标 0→100 | A | B |", "|---|---:|---:|"]
    for title,source,key in [("细节梯度/目标梯度",rows,"gradient_ratio"),("MSE",rows,"mse"),("边缘距离代理（px）",rows,"split_edge_distance_px"),("投影半径估计（px）",raw,"projected_radius_median")]:
        values=[f"{avg(source,g,0,key):.4f} → {avg(source,g,100,key):.4f}" for g in ["A","B"]]
        lines.append(f"| {title} | {values[0]} | {values[1]} |")
    lines += ["", "## 有无训练 mask 的公平图像比较", "", "下面使用两次实验所有检查点共同有效的同一静态区域，取每组两个样本等权均值。梯度下降是变平滑的代理，不是完整画质结论。", "", "| 组 | 训练方式 | 梯度比 0→100 | MSE 0→100 |", "|---|---|---:|---:|"]
    for group in ["A","B"]:
        for label in ["unmasked","dynamic_masked"]:
            def value(step,key): return np.mean([r[key] for r in comparisons if r["group"]==group and r["run"]==label and r["step"]==step])
            lines.append(f"| {group} | {label} | {value(0,'gradient_ratio'):.4f} → {value(100,'gradient_ratio'):.4f} | {value(0,'mse'):.5f} → {value(100,'mse'):.5f} |")
    lines += ["", "## 限制与文件", "", "两组真实深度和场景仍未完全配平，B 评估来自同一场景；单种子、少样本、100 步，CUDA 渲染存在非确定性。不能仅凭此次差异证明因果。遮罩精度受标注覆盖和矩形投影限制，未排除天空。", "", "comparison.png 为训练前后 RGB 和固定区域分开渲染差异；depth.png 为深度/残差。各 step 目录含分开和合并 RGB/alpha、npz。settings.json、selection.json、mask_audit.json、run.log 保存设置、选帧、遮罩和梯度核查。", "", "复现：原训练命令增加 --dynamic-training；汇总脚本同样增加 --dynamic-training。"]
    (ROOT/"REPORT.md").write_text("\n".join(lines))
