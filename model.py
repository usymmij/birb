import timm
import torch

class 
    def __init__():
        model = timm.create_model('vit_base_patch16_384.augreg_in21k_ft_in1k', 
                         pretrained=True, num_classes=206)
        model
        
        data_config = timm.data.resolve_model_data_config(model)
        
        transforms = timm.data.create_transform(**data_config, is_training=False)
        
        def train(self):
            self.model.train()
        def eval(self):
            self.model.eval()


