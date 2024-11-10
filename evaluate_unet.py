import cv2
import torch
import numpy as np

class UNetEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.thresholds = [0.5, 0.7, 0.9, 0.95, 0.99]
    
    def evaluate_sample(self, data):
        self.model.eval()
        with torch.no_grad():
            image = data['image'].float().unsqueeze(0).unsqueeze(0).to(self.device)
            prediction = self.model(image)
            prediction = torch.sigmoid(prediction)
            
            self._display_results(data, prediction)
    
    def _display_results(self, data, prediction):
        # Original image and ground truth
        cv2.imshow("image", data['image'].numpy())
        cv2.imshow("ground_truth", data['mask'].numpy())
        
        # Show predictions at different thresholds
        for threshold in self.thresholds:
            mask_name = f"pred_threshold_{threshold}"
            pred_mask = (prediction > threshold).float()
            pred_mask = pred_mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
            cv2.imshow(mask_name, pred_mask)
        
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        return key == ord('q')