from ultralytics import YOLO
import torch

class FinetuneYolo(YOLO):
    def load_backbone(self, ckptPath):
        """
        Transfers backbone parameters with matching names and shapes from 'weights' to model.
        """
        backboneWeight = torch.load(ckptPath)
        self.model.load_state_dict(backboneWeight, strict=False)
        return self
    
    def freeze_backbone(self, freeze):
        # Freeze backbone params
        freeze = [f'model.{x}.' for x in range(freeze)] # layers to freeze
        for k, v in self.model.named_parameters():
            v.requires_grad = True # train all layers
            # v.register_hook(lambda x: torch.nan_to_num(x)) # NaN to 0(commented for erratic training results)
            if any(x in k for x in freeze):
                v.requires_grad = False
        return self
    
    def unfreeze_backbone(self):
        # unfreeze backbone params
        for k, v in self.model.named_parameters():
            v.required_grad = True # train all layers
            # v.register_hook(lambda x: torch.nan_to_num(x)) # NaN to 0(commented for erratic training results)
        return self

model = FinetuneYolo(r'runs/detect/train/weights/best.pt')
model.freeze_backbone(10)

# Use the model
model.train(data="hand.yaml", epochs=200, imgsz=640, batch=28)  # train the model