import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from config import TrainingConfig
from utils.dice_score import dice_loss

class UNetTrainer:
    def __init__(self, model, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            'max',
            patience=5
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.grad_scaler = torch.amp.GradScaler(enabled=False)
        
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_loss = 0
        
        with tqdm(total=len(train_loader.dataset),
                 desc=f'Epoch {epoch}/{self.config.epochs}',
                 unit='img') as pbar:
            
            for batch in train_loader:
                loss = self._train_step(batch)
                epoch_loss += loss.item()
                
                pbar.update(batch['image'].shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
        return epoch_loss / len(train_loader)
    
    def _train_step(self, batch):
        images = batch['image'].unsqueeze(1).to(
            device=self.device,
            dtype=torch.float32,
            memory_format=torch.channels_last
        )
        true_masks = batch['mask'].to(
            device=self.device,
            dtype=torch.long
        )
        
        # Forward pass
        masks_pred = self.model(images)
        loss = self.criterion(masks_pred.squeeze(1), true_masks.float())
        
        # Calculate dice loss
        pred_mask = F.sigmoid(masks_pred.squeeze(1))
        dice = dice_loss(pred_mask, true_masks.float()) * self.config.dice_weight
        loss += dice
        
        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clipping
        )
        
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        
        return loss
    
    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'{self.config.model_name}-{epoch}.pth'
        )
        torch.save(self.model, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')