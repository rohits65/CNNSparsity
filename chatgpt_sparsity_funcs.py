import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_feather_sparsity(model, sparsity_ratio=0.5):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=sparsity_ratio)  # L1 pruning
            print(f"Applied Feather sparsity with {sparsity_ratio*100}% pruning on {name}")

def apply_entropy_aware_sparsity(model, train_loader, sparsity_ratio=0.5):
    with torch.no_grad():
        # Compute entropy of filter activations
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                entropy = []
                for inputs, _ in train_loader:
                    inputs = inputs.to(device)
                    outputs = module(inputs)
                    activations = outputs.view(outputs.size(0), -1)  # Flatten activations
                    entropy.append(-torch.sum(F.softmax(activations, dim=1) * F.log_softmax(activations, dim=1), dim=1))  # Shannon entropy

                avg_entropy = torch.mean(torch.stack(entropy), dim=0)
                _, indices = torch.topk(avg_entropy, int(avg_entropy.size(0) * (1 - sparsity_ratio)), largest=False)  # Prune least informative filters
                mask = torch.ones(module.weight.size(0)).bool().to(device)
                mask[indices] = False
                module.weight.data = module.weight.data[mask]
                print(f"Applied entropy-aware pruning with {sparsity_ratio*100}% pruning on {name}")



def apply_spartan_sparsity(model, sparsity_ratio=0.5):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Spartan filter pruning (channel pruning)
            num_filters = module.weight.size(0)
            num_pruned = int(num_filters * sparsity_ratio)
            normed_filters = module.weight.norm(p=2, dim=(1, 2, 3))  # L2 norm for each filter
            _, indices = torch.topk(normed_filters, num_filters - num_pruned, largest=False)  # Find least important filters
            mask = torch.ones(num_filters).bool().to(device)
            mask[indices] = False
            module.weight.data = module.weight.data[mask]
            print(f"Applied Spartan sparsity with {sparsity_ratio*100}% filter pruning on {name}")
