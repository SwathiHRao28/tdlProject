import torch
import torch.nn as nn
import torchvision.models as models

class VisionEncoder(nn.Module):
    def __init__(self, model_type="vit", encoded_image_size=14):
        super(VisionEncoder, self).__init__()
        self.model_type = model_type.lower()
        self.encoded_image_size = encoded_image_size
        
        if self.model_type == "vit":
            # ViT-B/16 extracts 14x14 grid of 768-dim patches for 224x224 image
            self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            self.out_dim = 768
            # We don't need the classification head
            self.model.heads = nn.Identity()
            
        elif self.model_type == "resnet":
            resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            modules = list(resnet.children())[:-2] # Remove adaptive pool & fc
            self.model = nn.Sequential(*modules)
            self.out_dim = 2048
            
        else:
            raise ValueError(f"Unknown encoder type: {self.model_type}")

        # Fine-tuning config: freeze early layers if needed, or all for debug
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, images):
        """
        Forward pass.
        Args:
            images: Tensor of shape (batch_size, 3, image_size, image_size)
        Returns:
            features: Tensor of shape (batch_size, num_pixels, feature_dim)
        """
        if self.model_type == "vit":
            # Process through ViT
            x = self.model._process_input(images)
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.model.encoder(x)
            
            # Remove class token (index 0), keep the 196 spatial patches
            features = x[:, 1:, :] # shape: (batch_size, 196, 768)
            
        elif self.model_type == "resnet":
            features = self.model(images) # (batch_size, 2048, 7, 7) or (14, 14) depending on input
            features = features.permute(0, 2, 3, 1) # (batch_size, 14, 14, 2048)
            features = features.view(features.size(0), -1, features.size(-1)) # (batch, num_pixels, 2048)
            
        return features
