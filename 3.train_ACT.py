import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps

# --- 1. 配置参数 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_steps = 3000
log_freq = 100 #日志打印频率
dataset_repo = "fr3_pnp"
root_dir = './demo_data'
checkpoint_path = './ckpt/ACT_ee_pose'

# --- 2. 数据增强定义 ---
class AddGaussianNoise(object):
    """向 Tensor 添加高斯噪声"""
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

transform = transforms.Compose([
    AddGaussianNoise(mean=0., std=0.02),
    transforms.Lambda(lambda x: x.clamp(0, 1))
])

# --- 3. 采样器定义 (用于推理测试) ---
class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        # 根据情节索引获取数据帧的起止范围
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)

def main():
    # --- 4. 初始化策略和配置 ---
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo, root=root_dir)
    features = dataset_to_policy_features(dataset_metadata.features)
    
    # 区分输入和输出特征，放进不同的字典
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    # 根据 notebook 逻辑移除不需要的观测
    if "observation.wrist_image" in input_features:
        input_features.pop("observation.wrist_image")

    # 配置 ACT 策略
    cfg = ACTConfig(
        input_features=input_features, 
        output_features=output_features, 
        chunk_size=10, #未来10步的动作
        n_action_steps=10 #主臂的动作(label)
    )
    
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)#时间戳偏移量，计算获取哪些数据
    policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # --- 5. 加载数据集 ---
    dataset = LeRobotDataset(
        dataset_repo, 
        delta_timestamps=delta_timestamps, 
        root=root_dir, 
        image_transforms=transform
    )

    dataloader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=(device.type != "cpu"),
        drop_last=True,
    )

    # --- 6. 训练循环 ---
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    print(f"开始训练，预计步数: {training_steps}")
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            
            loss, _ = policy.forward(inp_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            
            step += 1
            if step >= training_steps:
                done = True
                break

    # 保存模型
    policy.save_pretrained(checkpoint_path)
    print(f"模型已保存至 {checkpoint_path}")

    # --- 7. 推理测试 (可选可视化) ---
    print("开始推理测试...")
    policy.eval()
    actions_list = []
    gt_actions_list = []
    
    episode_index = 0
    episode_sampler = EpisodeSampler(dataset, episode_index)
    test_dataloader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=1,
        shuffle=False,
        pin_memory=(device.type != "cpu"),
        sampler=episode_sampler,
    )
    
    policy.reset()
    with torch.no_grad():
        for batch in test_dataloader:
            inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            action = policy.select_action(inp_batch)
            actions_list.append(action)
            gt_actions_list.append(inp_batch["action"][:,0,:])

    actions = torch.cat(actions_list, dim=0)
    gt_actions = torch.cat(gt_actions_list, dim=0)
    
    error = torch.mean(torch.abs(actions - gt_actions)).item()
    print(f"平均动作误差 (Mean action error): {error:.3f}")

    # 可视化前 7 个自由度的动作
    action_dim = 7
    fig, axs = plt.subplots(action_dim, 1, figsize=(10, 12))
    for i in range(action_dim):
        axs[i].plot(actions[:, i].cpu().numpy(), label="pred")
        axs[i].plot(gt_actions[:, i].cpu().numpy(), label="gt")
        axs[i].legend()
        axs[i].set_ylabel(f"Dim {i}")
    plt.tight_layout()
    plt.savefig(f"{checkpoint_path}/inference_results.png", dpi=150, bbox_inches='tight')
    print(f"✓ 推理结果图已保存至 {checkpoint_path}/inference_results.png")
    plt.show()

if __name__ == "__main__":
    main()