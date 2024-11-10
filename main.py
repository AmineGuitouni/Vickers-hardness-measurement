from config import TrainingConfig
from data.dataset import ImagesDataSetUNet
from evaluate_unet import UNetEvaluator
from train_unet import UNetTrainer
from unet.unet_model import UNet
from torch.utils.data import DataLoader
import torch

def main():
    # Load configuration
    config = TrainingConfig()
    
    # Initialize dataset and dataloader
    dataset = ImagesDataSetUNet(
        config.data_file,
        image_size=config.image_size
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Initialize model
    try:
        model = torch.load(f'{config.checkpoint_dir}/{config.model_name}-1.pth')
        print("Loaded existing model checkpoint")
    except FileNotFoundError:
        print("Creating new model")
        model = UNet(n_channels=1, n_classes=1)
    
    # Training
    trainer = UNetTrainer(model, config)
    for epoch in range(1, config.epochs + 1):
        epoch_loss = trainer.train_epoch(train_loader, epoch)
        print(f'Epoch {epoch} - Average loss: {epoch_loss:.4f}')
        trainer.save_checkpoint(epoch)
    
    # Evaluation
    evaluator = UNetEvaluator(model, config.device)
    for data in dataset:
        if evaluator.evaluate_sample(data):
            break

if __name__ == '__main__':
    main()