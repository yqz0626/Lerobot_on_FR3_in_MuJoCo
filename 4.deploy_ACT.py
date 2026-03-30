from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.common.datasets.utils import write_json, serialize_dict
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features
import torch
from PIL import Image
import torchvision
from mujoco_env.y_env import SimpleEnv

DEBUG_PRINT_PRED = True
DEBUG_PRINT_EVERY = 20

#导入环境
xml_path = './asset/example_scene_y.xml'
PnPEnv = SimpleEnv(xml_path, action_type='joint_angle')

#导入训练好的策略
device = 'cuda'
dataset_metadata = LeRobotDatasetMetadata("fr3_pnp", root='./demo_data')
features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}
input_features.pop("observation.wrist_image")
# 策略是通过一个配置类进行初始化的，在这个例子中是 `ACTConfig`。
# 对于这个例子，我们将只使用默认值，因此除了输入/输出特征之外，不需要传递任何其他参数。
# 时间集成（Temporal ensemble）用于实现更平滑的轨迹预测
cfg = ACTConfig(
    input_features=input_features, 
    output_features=output_features, 
    chunk_size= 10, 
    n_action_steps=1, 
    temporal_ensemble_coeff = 0.9
)
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)#时间戳偏移量，对时间对齐，用于取数据
# 现在我们可以使用这个配置和数据集的统计信息来实例化我们的策略了。
policy = ACTPolicy.from_pretrained('./ckpt/ACT_ee_pose', config = cfg, dataset_stats=dataset_metadata.stats)
policy.to(device)

#跑一遍策略
step = 0
PnPEnv.reset(seed=None)
policy.reset()
policy.eval()
save_image = True
img_transform = torchvision.transforms.ToTensor()
while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # Check if the task is completed
        success = PnPEnv.check_success()
        if success:
            print('Success')
            # Reset the environment and action queue
            policy.reset()
            PnPEnv.reset(seed=0)
            step = 0
            save_image = False
        # Get the current state of the environment
        state = PnPEnv.get_ee_pose()
        # Get the current image from the environment
        image, wirst_image = PnPEnv.grab_image()
        image = Image.fromarray(image)
        image = image.resize((256, 256))
        image = img_transform(image)
        wrist_image = Image.fromarray(wirst_image)
        wrist_image = wrist_image.resize((256, 256))
        wrist_image = img_transform(wrist_image)
        data = {
            'observation.state': torch.tensor([state]).to(device),
            'observation.image': image.unsqueeze(0).to(device),
            'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
            'task': ['Put mug cup on the plate'],
            'timestamp': torch.tensor([step/20]).to(device)
        }
        # Select an action
        action = policy.select_action(data)
        action = action[0].cpu().detach().numpy()

        # 打印模型预测输出（限频，避免刷屏）
        if DEBUG_PRINT_PRED and step % DEBUG_PRINT_EVERY == 0:
            print(
                f"pred step={step} shape={action.shape} "
                f"min={action.min():.4f} max={action.max():.4f} "
                f"mean={action.mean():.4f}"
            )
            print(f"pred action = {np.array2string(action, precision=4, suppress_small=True)}")

        # Take a step in the environment
        _ = PnPEnv.step(action)
        PnPEnv.render()
        step += 1
        success = PnPEnv.check_success()
        if success:
            print('Success')
            break