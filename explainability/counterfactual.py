import torch
import torch.nn as nn

class CounterfactualLoss(nn.Module):
    def __init__(self, mask_ratio=0.2):
        """
        mask_ratio: fraction of the top attributed pixels to mask out
        """
        super(CounterfactualLoss, self).__init__()
        self.mask_ratio = mask_ratio

    def forward(self, model, features, captions, pad_idx, attribution_maps, padding_mask=None):
        """
        Calculates counterfactual loss by masking the top attributed features and
        maximizing the probability drop of the predicted words.
        
        Args:
            model: The compiled CaptionModel
            features: (B, num_pixels, hidden_size)
            captions: (B, seq_len)
            attribution_maps: (B, seq_len, num_pixels)
            padding_mask: (B, seq_len) Boolean mask True for padded
            
        Returns:
            Scalar loss
        """
        B, seq_len, num_pixels = attribution_maps.shape
        hidden_size = features.shape[-1]
        
        # We need to compute the original probabilities
        with torch.no_grad():
            orig_outputs, _ = model.decoder(features, captions, pad_idx)
            # Softmax to get probabilities (B, seq_len, vocab_size)
            orig_probs = torch.softmax(orig_outputs, dim=-1)
            
            # Extract probability of the target words (next words)
            # We predict captions[:, t+1] at step t, but our target is exactly the caption
            target_words = captions.unsqueeze(-1) # (B, seq_len, 1)
            orig_target_probs = orig_probs.gather(2, target_words).squeeze(-1) # (B, seq_len)
            
        total_cf_loss = 0.0
        valid_tokens = 0
        
        # It's computationally heavy to re-run the decoder for EVERY masked target word 
        # individually in a batch. To approximate and optimize for training, we can 
        # compute an aggregated mask over the sequence, or re-run for a few sampled words.
        # Alternatively, we can do batch-wise masking: for each word in the sequence, 
        # we create a masked feature map. This expands features to (B * seq_len, num_pixels, hidden_size).
        
        # To avoid OOM, let's process the sequence step-by-step or randomly sample 1 word per batch element.
        # In this implementation, to follow the paper faithfully, we process step-by-step
        
        for t in range(seq_len):
            if padding_mask is not None and padding_mask[:, t].all():
                continue # Skip if entirely padded
                
            attr_map_t = attribution_maps[:, t, :].detach() # (B, num_pixels)
            
            # Find the threshold for the top `mask_ratio` pixels
            k = max(1, int(num_pixels * self.mask_ratio))
            # Get the k-th largest value for each batch element
            topk_vals, _ = torch.topk(attr_map_t, k, dim=-1)
            thresholds = topk_vals[:, -1].unsqueeze(-1)
            
            # Create mask: 1 for regions to KEEP, 0 for regions to MASK
            keep_mask = (attr_map_t < thresholds).unsqueeze(-1).float() # (B, num_pixels, 1)
            
            # Apply mask
            masked_features = features * keep_mask
            
            # Re-run decoder with masked features
            masked_outputs, _ = model.decoder(masked_features, captions, pad_idx)
            masked_probs = torch.softmax(masked_outputs, dim=-1)
            
            # We only care about the probability drop for the word at step t
            target_word_t = target_words[:, t, :] # (B, 1)
            masked_target_prob_t = masked_probs[:, t, :].gather(1, target_word_t).squeeze(-1) # (B,)
            
            # Original prob at step t
            orig_target_prob_t = orig_target_probs[:, t] # (B,)
            
            # Loss: we want the drop (orig - masked) to be large and positive.
            # So we minimize - (orig - masked), which means minimizing masked_target_prob_t
            # Or directly penalize if original is not much larger than masked.
            # Let's use a margin ranking loss or simply - (orig_target_prob_t - masked_target_prob_t)
            drop = orig_target_prob_t - masked_target_prob_t
            
            if padding_mask is not None:
                step_valid = ~padding_mask[:, t]
                drop = drop * step_valid
                valid_tokens += step_valid.sum()
                total_cf_loss = total_cf_loss - drop.sum()
            else:
                valid_tokens += B
                total_cf_loss = total_cf_loss - drop.sum()
                
        # Average
        cf_loss = total_cf_loss / (valid_tokens + 1e-8)
        
        # The prompt says loss encourages probability drop for masked word:
        # L_cf = sum(P_orig(w) - P_masked(w))
        # Since we want to MINIMIZE the overall loss during training, 
        # and we want P_orig - P_masked to be LARGE,
        # we should RETURN -sum() or simply minimize P_masked directly.
        # Returning -drop means minimizing the negative drop, which maximizes the drop.
        
        # However, to avoid negative losses pushing ad infinitum:
        # A common approach is RELU(margin - (P_orig - P_masked))
        # Let's just use -drop.mean() for now as mathematically requested.
        
        return cf_loss
