import os
import torch
import torch.nn as nn
from explainability.attribution import compute_batch_attribution
from explainability.alignment_loss import AlignmentLoss
from explainability.counterfactual import CounterfactualLoss
from torch.utils.tensorboard import SummaryWriter

def train_model(model, train_loader, val_loader, vocab, config, device):
    # Setup
    pad_idx = vocab.stoi[vocab.pad_token]
    criterion_caption = nn.CrossEntropyLoss(ignore_index=pad_idx)
    criterion_align = AlignmentLoss()
    criterion_cf = CounterfactualLoss(mask_ratio=0.2)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=config["weight_decay"]
    )
    
    # We will only train decoder and vis_project components
    model.encoder.eval()
    
    writer = SummaryWriter(log_dir=os.path.join("outputs", "logs"))
    
    epochs = config["debug_epochs"] if config["debug"] else config["epochs"]
    global_step = 0
    
    lambda_align = config["alignment_weight"]
    lambda_cf = config["counterfactual_weight"]
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_caption_loss = 0
        total_align_loss = 0
        total_cf_loss = 0
        
        for batch_idx, (images, captions, raw_captions) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)
            
            # 1. Forward pass for captioning
            # For teacher forcing: input is captions[:, :-1], target is captions[:, 1:]
            targets = captions[:, 1:]
            padding_mask = (targets == pad_idx)
            
            outputs, attn_weights = model(images, captions, pad_idx)
            
            # outputs shape: (B, seq_len - 1, vocab_size)
            # targets shape: (B, seq_len - 1)
            caption_loss = criterion_caption(
                outputs.reshape(-1, outputs.shape[-1]), 
                targets.reshape(-1)
            )
            
            # 2. Extract features manually to compute attribution without running encoder again
            with torch.no_grad():
                features = model.encoder(images)
                features = model.vis_project(features)
                
            loss = caption_loss
            align_loss = torch.tensor(0.0).to(device)
            cf_loss = torch.tensor(0.0).to(device)
            
            if config["use_alignment_loss"] or config["use_counterfactual_loss"]:
                # 3. Compute Attribution Maps
                # We need input features to track gradients
                features = features.detach().clone()
                features.requires_grad_(True)
                
                # The actual inputs to decoder
                decoder_input = captions[:, :-1]
                attribution_maps = compute_batch_attribution(model, features, decoder_input, pad_idx, fast_mode=config["debug"])
                
                if config["use_alignment_loss"]:
                    # Ensure attn_weights shape matches attribution maps (B, seq_len, num_pixels)
                    if attn_weights is not None:
                        align_loss = criterion_align(attn_weights, attribution_maps, padding_mask)
                        loss = loss + lambda_align * align_loss
                        
                if config["use_counterfactual_loss"]:
                    # Counterfactual logic
                    cf_loss = criterion_cf(model, features, decoder_input, pad_idx, attribution_maps, padding_mask)
                    loss = loss + lambda_cf * cf_loss
                    
            # 4. Backward & Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_caption_loss += caption_loss.item()
            total_align_loss += align_loss.item()
            total_cf_loss += cf_loss.item()
            
            writer.add_scalar("Loss/Train_Total", loss.item(), global_step)
            writer.add_scalar("Loss/Train_Caption", caption_loss.item(), global_step)
            writer.add_scalar("Loss/Train_Align", align_loss.item(), global_step)
            writer.add_scalar("Loss/Train_CF", cf_loss.item(), global_step)
            
            global_step += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{batch_idx}], "
                      f"Loss: {loss.item():.4f} (Cap: {caption_loss.item():.4f}, "
                      f"Align: {align_loss.item():.4f}, CF: {cf_loss.item():.4f})")
                      
            # Limit the number of steps per epoch (defaulting to 100 if not specified)
            max_steps = config.get("max_steps_per_epoch", 100)
            if batch_idx >= max_steps - 1:
                print(f"Reached max steps per epoch ({max_steps}). Finishing epoch...")
                break
                      
        # End of epoch logging
        print(f"--- Epoch {epoch} Summary ---")
        print(f"Avg Loss: {total_loss/len(train_loader):.4f}")
        
        # Save checkpiont
        if epoch % config["save_every"] == 0:
            os.makedirs(config["checkpoint_dir"], exist_ok=True)
            checkpoint_path = os.path.join(config["checkpoint_dir"], f"epoch_{epoch:02d}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
    writer.close()
    return model
