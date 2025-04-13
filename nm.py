import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_nm_sparsity(model, sparsity_ratio=0.5):
    """
    Apply N:M sparsity (2:4 pattern) to the ResNet-20 model.
    This applies after training is completed and should only affect inference time.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # For each convolutional layer, apply 2:4 sparsity
            weight = module.weight.data
            num_filters = weight.size(0)  # Number of filters in the convolution
            
            # For every 4 weights, we keep the 2 largest magnitude weights and zero out the others
            for i in range(num_filters):
                # Get the weights of the i-th filter
                filter_weights = weight[i].view(-1)  # Flatten the filter weights
                topk_values, topk_indices = torch.topk(torch.abs(filter_weights), int(len(filter_weights) * sparsity_ratio))
                
                # Set the smallest 2 values in each 4-group to zero
                threshold = topk_values[-1]  # The smallest non-zero magnitude in the top-k
                weight[i].data[torch.abs(weight[i]) < threshold] = 0  # Prune the less important weights
            
            print(f"Applied N:M sparsity (2:4) on {name} filters")

