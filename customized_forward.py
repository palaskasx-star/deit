# 2022.10.14-Changed for building manifold kd
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#
# Modified from Fackbook, Deit
# {haozhiwei1, jianyuan.guo}@huawei.com
#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

from types import MethodType

import torch

def register_forward(model, model_name):
    # Check for DINOv3 in the name
    if 'dinov3' in model_name.lower():
        model.forward_features = MethodType(dinov3_forward_features, model)
        model.forward = MethodType(dinov3_forward, model)
    elif model_name.split('_')[0] == 'deit' or  model_name.split('_')[0] == 'vit':
        model.forward_features = MethodType(vit_forward_features, model)
        model.forward = MethodType(vit_forward, model)
    elif model_name.split('_')[0] == 'cait':
        model.forward_features = MethodType(cait_forward_features, model)
        model.forward = MethodType(cait_forward, model)
    else:
        raise RuntimeError(f'Not defined customized method forward for model {model_name}')

# --- New DINOv3 specific methods ---
def register_forward(model, model_name):
    # Check for DINOv3 in the name
    if 'dinov3' in model_name.lower():
        model.forward_features = MethodType(dinov3_forward_features, model)
        model.forward = MethodType(dinov3_forward, model)
    elif model_name.split('_')[0] == 'deit' or  model_name.split('_')[0] == 'vit':
        model.forward_features = MethodType(vit_forward_features, model)
        model.forward = MethodType(vit_forward, model)
    elif model_name.split('_')[0] == 'cait':
        model.forward_features = MethodType(cait_forward_features, model)
        model.forward = MethodType(cait_forward, model)
    else:
        raise RuntimeError(f'Not defined customized method forward for model {model_name}')

# --- New DINOv3 specific methods ---
def dinov3_forward_features(self, x, require_feat: bool = False):
    
    # Initialize lists
    patch_tokens_list = []
    cls_tokens_list = []
    reg_tokens_list = []
    num_reg = self.reg_token.shape[1]

    # --------------------------------------------------------
    # 1. Capture Raw/Initial Representations (Matching ViT Logic)
    # --------------------------------------------------------
    if require_feat:
        # Get patch size (assuming tuple or int)
        if hasattr(self.patch_embed, 'patch_size'):
            p = self.patch_embed.patch_size
            if isinstance(p, tuple): p = p[0]
        else:
            # Fallback for some implementations
            p = 14 

        # A. Raw Patches
        if len(x.shape) == 4:
            # unfold logic: (B, C, H, W) -> (B, N, C*p*p)
            raw_patches = x.unfold(2, p, p).unfold(3, p, p)
            B_new, C, H_p, W_p, p1, p2 = raw_patches.shape
            raw_flattened = raw_patches.permute(0, 2, 3, 1, 4, 5).reshape(B_new, -1, C * p1 * p2)
            
            patch_tokens_list.append(raw_flattened)
            
            # B. Raw "CLS" (Full image flattened, following your ViT reference)
            cls_tokens_list.append(x.view(x.size(0), -1).unsqueeze(1))

            reg_tokens_list.append(x.view(x.size(0), -1).unsqueeze(1).expand(-1, num_reg, -1))
        else:
            pass 

    # --------------------------------------------------------
    # 2. Embedding & Token Prep
    # --------------------------------------------------------
    x = self.patch_embed(x)
    x = x.flatten(1, 2) 

    # Expand special tokens
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    reg_tokens = self.reg_token.expand(x.shape[0], -1, -1)
    
    
    # Concatenate: [CLS, REG..., PATCH...]
    x = torch.cat((cls_token, reg_tokens, x), dim=1)
    x = self.pos_drop(x)

    # --------------------------------------------------------
    # 3. Capture Post-Embedding Representations
    # --------------------------------------------------------
    if require_feat:
        cls_tokens_list.append(x[:, 0].unsqueeze(1).clone())
        reg_tokens_list.append(x[:, 1:1+num_reg].clone())
        patch_tokens_list.append(x[:, 1+num_reg:].clone())

    # --------------------------------------------------------
    # 4. Blocks
    # --------------------------------------------------------
    for blk in self.blocks:
        x = blk(x)
        
        if require_feat:
            cls_tokens_list.append(x[:, 0].unsqueeze(1).clone())
            reg_tokens_list.append(x[:, 1:1+num_reg].clone())
            patch_tokens_list.append(x[:, 1+num_reg:].clone())

    # --------------------------------------------------------
    # 5. Final Output
    # --------------------------------------------------------
    x = self.norm(x)

    if require_feat:
        return x[:, 0], patch_tokens_list, reg_tokens_list, cls_tokens_list
    else:
        return x[:, 0]

def dinov3_forward(self, x, require_feat: bool = True):
    if require_feat:
        # Get all lists
        cls_feat, patch_tokens_list, reg_tokens_list, cls_tokens_list = self.forward_features(x, require_feat=True)
        
        # Compute final logits
        x_cls = self.head(cls_feat)
        
        # Return: (Patches, Registers, CLS, Logits)
        return patch_tokens_list, cls_tokens_list, x_cls
    else:
        cls_feat = self.forward_features(x, require_feat=False)
        x_cls = self.head(cls_feat)
        return x_cls

# deit & vit
def vit_forward_features(self, x, require_feat: bool = False):
    # --------------------------------------------------------
    # CASE A: Vanilla Vision Transformer (No Distillation Token)
    # Ref: VanillaVisionTransformer.forward_features
    # --------------------------------------------------------
    B = x.shape[0]
    
    cls_tokens_representations = []
    representations = []

    # 1. Capture Raw/Initial Representations (Specific to Manifold KD Vanilla logic)
    if require_feat:
        # Replicating the 'unfold' logic for raw patches from reference
        p = self.patch_embed.patch_size[0]
        # Ensure input is suitable for unfolding (B, C, H, W)
        if len(x.shape) == 4:
            raw_patches = x.unfold(2, p, p).unfold(3, p, p)
            B_new, C, H_p, W_p, p1, p2 = raw_patches.shape
            raw_flattened = raw_patches.permute(0, 2, 3, 1, 4, 5).reshape(B_new, -1, C * p1 * p2)
            representations.append(raw_flattened)
            
            # Capture initial CLS token state (before embedding/blocks)
            # Note: Reference uses x.view... here, but x is an image. 
            # We stick to the reference logic:
            cls_tokens_representations.append(x.view(x.size(0), -1).unsqueeze(1))
        else:
            # Fallback if x is already flattened/embedded (edge case)
            pass

    # 2. Embedding
    x = self.patch_embed(x)
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # 3. Capture Post-Embedding Representations
    if require_feat:
        cls_tokens_representations.append(x[:, 0].unsqueeze(1).clone())
        representations.append(x[:, 1:, :].clone())

    x = x + self.pos_embed
    x = self.pos_drop(x)

    # 4. Blocks
    for blk in self.blocks:
        x = blk(x)
        if require_feat:
            cls_tokens_representations.append(x[:, 0].unsqueeze(1).clone())
            representations.append(x[:, 1:, :].clone())

    # 5. Final Norm
    x = self.norm(x)
    final_cls = x[:, 0]

    if require_feat:
        return final_cls, representations, cls_tokens_representations
    else:
        return self.pre_logits(final_cls)


def vit_forward(self, x, require_feat: bool = True):
    # --------------------------------------------------------
    # CASE A: Vanilla Vision Transformer
    # --------------------------------------------------------
    if require_feat:
        x_cls, representations, cls_tokens = self.forward_features(x, require_feat=True)
        x_cls = self.head(x_cls)
        # Returns: (representations list, cls_tokens list, final_logits)
        return representations, cls_tokens, x_cls

    # --------------------------------------------------------
    # CASE B: Distilled Vision Transformer
    # --------------------------------------------------------
    else:
        if require_feat:
            x_cls, x_dist, representations, dist_tokens = self.forward_features(x, require_feat=True)
        else:
            x_cls, x_dist = self.forward_features(x, require_feat=False)

        x_cls = self.head(x_cls)
        x_dist = self.head_dist(x_dist)

        if self.training:
            if require_feat:
                # Returns: (rep list, dist list, cls_logits, dist_logits)
                return representations, dist_tokens, x_cls, x_dist
            return x_cls, x_dist
        else:
            # Inference average
            out = (x_cls + x_dist) / 2
            if require_feat:
                # Returns: (rep list, dist list, averaged_logits)
                return representations, dist_tokens, out
            return out

def cait_forward_features(self, x, require_feat: bool = False):
    B = x.shape[0]
    
    cls_tokens_representations = []
    representations = []

    # --------------------------------------------------------
    # 1. Capture Raw/Initial Representations
    # (Exact copy of Manifold KD / ViT logic provided)
    # --------------------------------------------------------
    if require_feat:
        p = self.patch_embed.patch_size[0]
        if len(x.shape) == 4:
            # Extract raw patches
            raw_patches = x.unfold(2, p, p).unfold(3, p, p)
            B_new, C, H_p, W_p, p1, p2 = raw_patches.shape
            raw_flattened = raw_patches.permute(0, 2, 3, 1, 4, 5).reshape(B_new, -1, C * p1 * p2)
            representations.append(raw_flattened)
            
            # Capture raw image as initial CLS state (following reference logic)
            cls_tokens_representations.append(x.view(x.size(0), -1).unsqueeze(1))
        else:
            pass

    # --------------------------------------------------------
    # 2. Embedding
    # --------------------------------------------------------
    x = self.patch_embed(x)
    x = x + self.pos_embed
    x = self.pos_drop(x)
    
    # Initialize CLS token (Expanded, but not yet processed)
    cls_tokens = self.cls_token.expand(B, -1, -1)

    # 3. Capture Post-Embedding Representations
    if require_feat:
        cls_tokens_representations.append(cls_tokens.clone())
        representations.append(x.clone())

    # --------------------------------------------------------
    # 4. Stage 1: Patch-Only Blocks (Self-Attention)
    # --------------------------------------------------------
    # In this stage, only 'x' (patches) evolves. 'cls_tokens' is static.
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        if require_feat:
            representations.append(x.clone())
            # CLS token hasn't interacted yet, so we append the current static state
            # to keep list lengths consistent with depth.
            cls_tokens_representations.append(cls_tokens.clone())

    # --------------------------------------------------------
    # 5. Stage 2: Class-Attention Blocks (Cross-Attention)
    # --------------------------------------------------------
    # In this stage, 'x' (patches) is frozen/used as keys/values. 'cls_tokens' evolves.
    for i, blk in enumerate(self.blocks_token_only):
        cls_tokens = blk(x, cls_tokens)
        if require_feat:
            # Patches don't change here, so we append the frozen patch state
            representations.append(x.clone())
            cls_tokens_representations.append(cls_tokens.clone())

    # --------------------------------------------------------
    # 6. Final Norm & Output
    # --------------------------------------------------------
    # Concatenate to apply global norm (standard CaiT procedure)
    x_concat = torch.cat((cls_tokens, x), dim=1)
    x_norm = self.norm(x_concat)
    
    # Extract the final normalized CLS token
    final_cls = x_norm[:, 0]

    if require_feat:
        return final_cls, representations, cls_tokens_representations
    else:
        return final_cls

def cait_forward(self, x, require_feat: bool = True):
    if require_feat:
        # Unpack the 3 return values from forward_features
        x_cls, representations, cls_tokens = self.forward_features(x, require_feat=True)
        
        # Pass final CLS through the head
        x_cls = self.head(x_cls)
        
        # Returns: (representations list, cls_tokens list, final_logits)
        return representations, cls_tokens, x_cls
    else:
        # Standard Inference
        x_cls = self.forward_features(x, require_feat=False)
        x_cls = self.head(x_cls)
        return x_cls

# --------------------
# RegNetY (from timm)
# --------------------
def regnet_forward_features(self, x, require_feat: bool = False):
    """
    Custom forward_features for timm RegNetY-160.
    Captures intermediate outputs after each stage.
    """
    block_outs = []

    # Stem
    x = self.stem(x)
    block_outs.append(torch.nn.Unfold(kernel_size=8, stride=8)(x).permute(0, 2, 1))

    # Stages (typical timm RegNet has 4 stages: s1-s4)
    x = self.s1(x); block_outs.append(torch.nn.Unfold(kernel_size=4, stride=4)(x).permute(0, 2, 1))
    x = self.s2(x); block_outs.append(torch.nn.Unfold(kernel_size=2, stride=2)(x).permute(0, 2, 1))
    x = self.s3(x); block_outs.append(torch.nn.Unfold(kernel_size=1, stride=1)(x).permute(0, 2, 1))
    x = self.s4(x); block_outs.append(torch.nn.Unfold(kernel_size=1, stride=1)(torch.nn.AdaptiveAvgPool2d(14)(x)).permute(0, 2, 1))


    # Head
    x = self.head.global_pool(x)
    x = self.head.flatten(x)
    x = self.head.fc(x)

    if require_feat:
        return x, block_outs
    else:
        return x


def regnet_forward(self, x, require_feat: bool = True):
    if require_feat:
        logits, feats = self.forward_features(x, require_feat=True)
        return logits, feats
    else:
        return self.forward_features(x)


def get_layer_names(model, input_size=(1, 3, 224, 224)):
    """
    Runs a dummy forward pass to count the actual number of feature layers 
    returned by the model and generates names x^(0) to x^(N).
    """
    device = next(model.parameters()).device
    dummy_input = torch.zeros(input_size, device=device)
    
    # We set training to False to avoid updating batch stats or dealing with dropout
    was_training = model.training
    model.eval()

    with torch.no_grad():
        try:
            # Run forward pass requesting features
            # Note: We must handle the different return signatures of your models
            output = model(dummy_input, require_feat=True)
            
            # --------------------------------------------------------
            # SCENARIO 1: Output is a tuple (Standard for your models)
            # --------------------------------------------------------
            if isinstance(output, tuple):
                # We need to find which element of the tuple is the list of features.
                # Based on your code:
                # DINOv3 -> (cls, block_outs) -> Index 1 is list
                # ViT Vanilla -> (reps, cls_tokens, logits) -> Index 0 is list
                # ViT Distilled -> (reps, dists, logits_cls, logits_dist) -> Index 0 is list
                # CaiT -> (logits, block_outs) -> Index 1 is list
                # RegNet -> (logits, block_outs) -> Index 1 is list
                
                feature_list = None
                print('Here')
                # Check the first element
                if isinstance(output[0], list):
                    feature_list = output[0]
                    print('Here 1')
                # Check the second element
                elif len(output) > 1 and isinstance(output[1], list):
                    feature_list = output[1]
                    print('Here 2')
                
                if feature_list is not None:
                    count = len(feature_list)
                    print('Here 3')
                    return [f'x^({i})' for i in range(count)]
            
            # --------------------------------------------------------
            # SCENARIO 2: Output is just the list (Unlikely but possible)
            # --------------------------------------------------------
            elif isinstance(output, list):
                count = len(output)
                return [f'x^({i})' for i in range(count)]

            raise RuntimeError("Could not find a list in the model output.")

        except Exception as e:
            print(f"Error during dummy forward pass: {e}")
            return []
            
        finally:
            # Restore original training state
            model.train(was_training)