import torch
from omegaconf import OmegaConf
import sys
import os

# 确保能找到 src
sys.path.append(os.getcwd())

from src.dataset import get_dataset
# 如果路径不对，尝试 from src.dataset.data_module import get_data_shim
try:
    from src.dataset.data_module import get_data_shim
except:
    def get_data_shim(encoder): return lambda x: x

def debug():
    print("1. 开始加载配置...")
    cfg = OmegaConf.load("config/main.yaml")
    # 模拟你的命令行覆盖
    cfg.dataset = OmegaConf.load("config/dataset/nuscenes.yaml")
    cfg.dataset.view_sampler = OmegaConf.load("config/dataset/view_sampler/bounded.yaml")
    
    print(f"2. 初始化 nuScenes 测试集 (Split: test)...")
    try:
        dataset = get_dataset(cfg.dataset, "train", None)
        print(f"   数据集长度: {len(dataset)}")
    except Exception as e:
        print(f"❌ 数据集初始化失败: {e}")
        return

    print("3. 尝试读取第一条数据 (这将触发 __getitem__)...")
    try:
        # 这里最容易卡死
        for i, item in enumerate(dataset):
            print(f"✅ 成功读取第 {i} 条数据!")
            print(f"   场景名称: {item['scene']}")
            print(f"   Context 帧索引: {item['context']['index']}")
            if i >= 0: break 
    except Exception as e:
        print(f"❌ 读取数据时崩溃: {e}")

if __name__ == "__main__":
    debug()