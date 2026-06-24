## BibTeX

```bibtex
@article{chen2024mvsplat,
    title   = {MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images},
    author  = {Chen, Yuedong and Xu, Haofei and Zheng, Chuanxia and Zhuang, Bohan and Pollefeys, Marc and Geiger, Andreas and Cham, Tat-Jen and Cai, Jianfei},
    journal = {arXiv preprint arXiv:2403.14627},
    year    = {2024},
}
```

## Acknowledgements

The project is largely based on [pixelSplat](https://github.com/dcharatan/pixelsplat) and has incorporated numerous code snippets from [UniMatch](https://github.com/autonomousvision/unimatch). Many thanks to these two projects for their excellent contributions!




# 当前状态（Current Status）

- 已完成：
  - ✅ MVSplat 在 re10k 跑通
  - ✅ nuScenes 数据加载正常
  - ✅ 时序输入（t-3,t-2,t-1→t）可用
  - ✅ 时序适配训练（无 LiDAR）
  - ✅ lidar bias
  - ✅ lidar loss
  - 正在进行 跑实验结果
  - 预计还要做Unet改造

