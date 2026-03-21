import torch
import torch.nn as nn
from models.encoder import VisionEncoder
from models.decoder import CaptionDecoder

class CaptionModel(nn.Module):
    def __init__(self, encoder_type, vocab_size, embed_size, hidden_size, num_layers, nhead, max_length):
        super(CaptionModel, self).__init__()
        
        self.encoder = VisionEncoder(model_type=encoder_type)
        self.decoder = CaptionDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nhead=nhead,
            max_length=max_length
        )
        
        # Linear layer to map visual features to text hidden dimension
        self.vis_project = nn.Linear(self.encoder.out_dim, hidden_size)

    def forward(self, images, captions, pad_idx=None):
        """
        images: (B, 3, H, W)
        captions: (B, seq_len)
        Returns: outputs (B, seq_len, vocab_size), attn_weights (B, seq_len, num_pixels)
        """
        features = self.encoder(images)  # (B, num_pixels, encoder_dim)
        features = self.vis_project(features) # (B, num_pixels, hidden_size)
        
        # Teacher forcing
        # The input to decoder doesn't include the last token of captions to predict the next
        captions_input = captions[:, :-1] 
        outputs, attn_weights = self.decoder(features, captions_input, pad_idx)
        return outputs, attn_weights

    @torch.no_grad()
    def generate(self, image, vocab, max_length=20):
        # image layout: (1, 3, H, W)
        self.eval()
        features = self.encoder(image)
        features = self.vis_project(features)
        
        caption = [vocab.stoi[vocab.start_token]]
        
        for _ in range(max_length):
            captions_tensor = torch.tensor(caption).unsqueeze(0).to(image.device)
            outputs, attn_weights = self.decoder(features, captions_tensor, pad_idx=None)
            
            # Get the prediction for the last time step
            predicted_word_idx = outputs[0, -1, :].argmax(dim=-1).item()
            caption.append(predicted_word_idx)
            
            if predicted_word_idx == vocab.stoi[vocab.end_token]:
                break
                
        # Return tokens and attention weights for the generated sequence
        captions_tensor = torch.tensor(caption).unsqueeze(0).to(image.device)
        outputs, attn_weights = self.decoder(features, captions_tensor, pad_idx=None)
        
        # Convert indices to words
        words = [vocab.itos[idx] for idx in caption]
        return words, attn_weights
