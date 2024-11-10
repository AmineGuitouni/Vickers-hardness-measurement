from dataclasses import dataclass
import torch

@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 2
    learning_rate: float = 0.0001
    gradient_clipping: float = 1.0
    dice_weight: float = 1.0
    image_size: int = 256
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir: str = './checkpoints'
    data_file: str = './bounding_boxes_points_all.csv'
    model_name: str = 'test'