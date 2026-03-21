import torch
from captum.attr import LayerGradientXActivation

class CaptionModelWrapper(torch.nn.Module):
    """
    Wrapper for Captum to compute attribution of a specific predicted word
    with respect to the visual features.
    """
    def __init__(self, model, pad_idx):
        super().__init__()
        self.model = model
        self.pad_idx = pad_idx

    def forward(self, features, captions, target_step_idx):
        """
        features: (B, num_pixels, encoder_dim)
        captions: (B, seq_len)
        target_step_idx: the index of the word we want attribution for.
        """
        # We project features here if needed, but normally attribution
        # is computed against a specific layer's output.
        # This wrapper expects features to be the output of vis_project
        
        # Teacher forcing: predict next words
        outputs, _ = self.model.decoder(features, captions, self.pad_idx)
        
        # Return the logits for the target step
        # shape: (B, vocab_size)
        return outputs[:, target_step_idx, :]

def compute_attribution(model, features, captions, pad_idx, target_step_idx, target_class):
    """
    Compute Layer Gradient X Activation (similar to Grad-CAM) for a specific word.
    
    Args:
        model: The CaptionModel
        features: The visual features (B, num_pixels, hidden_size)
        captions: The input caption sequence (B, seq_len)
        target_step_idx: The step in the sequence to explain
        target_class: The actual word index predicted/ground truth at that step
    """
    wrapper = CaptionModelWrapper(model, pad_idx)
    
    # We want attribution with respect to the visual features
    # LayerGradientXActivation is efficient for training loops
    lgxa = LayerGradientXActivation(wrapper, layer=wrapper)
    
    # Captum wrapper trick: since we pass `features` as the first argument, 
    # we can use the wrapper itself as the layer if we modify layer logic, 
    # OR we can just use backprop manually which is often easier for custom architectures.
    
    # Let's do a simple manual Input X Gradient to avoid Captum layer tracking issues on raw tensors
    
    features.requires_grad_(True)
    outputs = wrapper(features, captions, target_step_idx)
    
    # Gather the logit for the target class
    score = outputs.gather(1, target_class.unsqueeze(1)).squeeze(1)
    
    # Compute gradients
    model.zero_grad()
    score.sum().backward(retain_graph=True)
    
    gradients = features.grad.detach()  # (B, num_pixels, hidden_size)
    
    # Input X Gradient
    attribution = features.detach() * gradients
    
    # Sum over the hidden dimension to get spatial heatmap
    # Shape: (B, num_pixels)
    attribution_map = attribution.sum(dim=-1)
    
    # Apply ReLU to keep only positive contributions
    attribution_map = torch.relu(attribution_map)
    
    # Clean up gradients
    features.requires_grad_(False)
    
    return attribution_map

def compute_batch_attribution(model, features, captions, pad_idx, fast_mode=False):
    """
    Compute attribution maps for all non-padded words in the sequence.
    This aggregates the attribution across the sequence.
    
    Args:
        model: The CaptionModel
        features: (B, num_pixels, hidden_size)  
        captions: (B, seq_len)
        pad_idx: Padding index
        fast_mode: If True, use approximation for faster computation (useful for debugging/CPU)
    """
    B, seq_len = captions.shape
    device = features.device
    num_pixels = features.shape[1]
    
    # Output tensor to hold attribution maps for each word step
    # Shape: (B, seq_len, num_pixels)
    batch_attr_maps = torch.zeros((B, seq_len, num_pixels), device=device)
    
    if fast_mode:
        # Fast approximation: use L2 norm of features as simple attribution
        # This avoids expensive backpropagation - useful for debugging
        feature_magnitude = torch.norm(features, p=2, dim=-1)  # (B, num_pixels)
        for t in range(seq_len):
            batch_attr_maps[:, t, :] = feature_magnitude
    else:
        # Full attribution computation via gradients (expensive)
        features.requires_grad_(True)
        outputs, _ = model.decoder(features, captions, pad_idx)
        
        for t in range(seq_len):
            # We explain the prediction of the true next word (teacher forcing)
            target_class = captions[:, t].clone()
            score = outputs[:, t, :].gather(1, target_class.unsqueeze(1)).squeeze(1)
            
            model.zero_grad()
            if features.grad is not None:
                features.grad.zero_()
                
            score.sum().backward(retain_graph=True)
            gradients = features.grad.detach()
            
            # Detach features to prevent building a graph that expects features.grad to remain unchanged
            attribution = features.detach() * gradients
            attr_map = torch.relu(attribution.sum(dim=-1))
            batch_attr_maps[:, t, :] = attr_map
        
        features.requires_grad_(False)
    
    # Normalize maps (min-max normalization per map)
    # Shape: (B, seq_len, num_pixels)
    min_vals = batch_attr_maps.min(dim=-1, keepdim=True)[0]
    max_vals = batch_attr_maps.max(dim=-1, keepdim=True)[0]
    
    # Avoid division by zero
    val_range = max_vals - min_vals
    val_range[val_range == 0] = 1e-8
    
    normalized_attr_maps = (batch_attr_maps - min_vals) / val_range
    
    return normalized_attr_maps
