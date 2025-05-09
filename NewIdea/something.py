"""
Here's what I want to do:
1. Use any small text generation model from huggingface
2. Capture activations
3. Change activation values
4. See new results
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.utils import get_act_name
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional

# Ensure output directory exists
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Initialize model with TransformerLens
model = HookedTransformer.from_pretrained("Qwen/Qwen2-1.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")

def plot_neuron_activations(
    activations,
    title,
    save_path,
    cmap = "RdYlBu_r"
):
    plt.figure(figsize=(15, 8))
    
    # Convert to numpy and take mean across batch dimension
    act_np = activations.detach().cpu().numpy()
    if len(act_np.shape) == 3:
        act_np = np.mean(act_np, axis=0)
    
    # Create heatmap
    sns.heatmap(
        act_np,
        cmap=cmap,
        center=0,
        cbar_kws={"label": "Activation Strength"}
    )
    
    plt.title(title, pad=20, fontsize=14)
    plt.xlabel("Neuron Index", fontsize=12, labelpad=10)
    plt.ylabel("Sequence Position", fontsize=12, labelpad=10)
    
    # Add grid lines
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, save_path), dpi=300, bbox_inches="tight")
    plt.close()

def plot_feature_importance(
    importance,
    title,
    save_path,
    top_k = 20
):
    plt.figure(figsize=(15, 8))
    
    # Convert to numpy and get top k features
    imp_np = importance.detach().cpu().numpy()
    top_indices = np.argsort(imp_np)[-top_k:]
    top_importance = imp_np[top_indices]
    
    # Create bar plot
    bars = plt.bar(range(top_k), top_importance, color="skyblue")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom"
        )
    
    plt.title(title, pad=20, fontsize=14)
    plt.xlabel("Feature Index", fontsize=12, labelpad=10)
    plt.ylabel("Average Activation Magnitude", fontsize=12, labelpad=10)
    
    # Add grid
    plt.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, save_path), dpi=300, bbox_inches="tight")
    plt.close()

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_weight = 0.1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_weight = sparsity_weight
        
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        
        # Initialize biases to zero
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode
        hidden = self.encoder(x)
        
        # Apply L1 regularization for sparsity
        sparsity_loss = torch.mean(torch.abs(hidden))
        
        # Decode
        reconstructed = self.decoder(hidden)
        return reconstructed, hidden, sparsity_loss

def train_sae(
    activations,
    input_dim,
    hidden_dim,
    num_epochs = 100,
    batch_size = 32,
    learning_rate = 1e-3,
    device = "cuda" if torch.cuda.is_available() else "cpu"
):
    # Prepare data
    all_activations = torch.cat([
        act.reshape(-1, act.shape[-1]) for act in activations
    ], dim=0)
    
    # Create data loader
    dataset = TensorDataset(all_activations)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Initialize model and optimizer
    sae = SparseAutoencoder(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        total_reconstruction_loss = 0
        total_sparsity_loss = 0
        
        for batch in dataloader:
            batch = batch[0].to(device)
            
            # Forward pass
            reconstructed, hidden, sparsity_loss = sae(batch)
            
            # Compute reconstruction loss
            reconstruction_loss = F.mse_loss(reconstructed, batch)
            
            # Total loss
            loss = reconstruction_loss + sae.sparsity_weight * sparsity_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_sparsity_loss += sparsity_loss.item()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            avg_reconstruction = total_reconstruction_loss / len(dataloader)
            avg_sparsity = total_sparsity_loss / len(dataloader)
            print(
                f"Epoch {epoch + 1}/{num_epochs}\n"
                f"Total Loss: {avg_loss:.4f}\n"
                f"Reconstruction Loss: {avg_reconstruction:.4f}\n"
                f"Sparsity Loss: {avg_sparsity:.4f}\n"
            )
    
    return sae

def visualize_activations(cache, input_text, layer_idx = 0):
    # Get activations for the specified layer
    act_name = get_act_name("mlp_out", layer_idx)
    activations = cache[act_name]
    
    # Plot neuron activations
    plot_neuron_activations(
        activations,
        title=f'Activation Heatmap - Layer {layer_idx}\nInput: "{input_text}"',
        save_path=f"activation_heatmap_layer_{layer_idx}.png"
    )

def analyze_activation_patterns(cache, input_text):
    # Get activation magnitudes for each layer
    layer_means = []
    for layer in range(model.cfg.n_layers):
        act_name = get_act_name("mlp_out", layer)
        layer_means.append(torch.mean(torch.abs(cache[act_name])).item())
    
    # Plot activation magnitudes
    plot_feature_importance(
        torch.tensor(layer_means),
        title=f'Average Activation Magnitude Across Layers\nInput: "{input_text}"',
        save_path="activation_magnitude_across_layers.png"
    )

def analyze_sae_features(
    sae,
    cache,
    input_text,
    layer_idx = 0
) -> torch.Tensor:
    # Get activations for the specific layer
    act_name = get_act_name("mlp_out", layer_idx)
    activations = cache[act_name]
    
    # Get SAE features
    with torch.no_grad():
        _, features, _ = sae(activations.reshape(-1, activations.shape[-1]))
    
    # Reshape back to sequence
    features = features.reshape(activations.shape[0], activations.shape[1], -1)
    
    # Plot feature activations
    plot_neuron_activations(
        features,
        title=f'SAE Feature Activations - Layer {layer_idx}\nInput: "{input_text}"',
        save_path=f"sae_features_layer_{layer_idx}.png"
    )
    
    return features

def visualize_feature_importance(features, layer_idx, input_text):
    # Calculate feature importance
    feature_importance = torch.mean(torch.abs(features), dim=(0, 1))
    
    # Plot feature importance
    plot_feature_importance(
        feature_importance,
        title=f'Top Features - Layer {layer_idx}\nInput: "{input_text}"',
        save_path=f"feature_importance_layer_{layer_idx}.png"
    )

def main():
    # Test with different inputs
    input_texts = [
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "I love machine learning and artificial intelligence.",
    ]

    for input_text in input_texts:
        print(f"\nAnalyzing input: {input_text}")
        
        # Get activations using TransformerLens
        cache = model.run_with_cache(
            input_text,
            return_type=None,
            names_filter=lambda name: "mlp_out" in name
        )[1]
        
        # # Generate outputs
        # output = model.generate(
        #     inputs["input_ids"],
        #     attention_mask=inputs["attention_mask"],
        #     max_length=50,
        #     num_return_sequences=1,
        #     temperature=0.7,
        #     do_sample=True,
        #     pad_token_id=tokenizer.eos_token_id,
        # )   
        # print(f"\nGenerated text: {output}")
        
        # Train SAE on the activations
        input_dim = model.cfg.d_model  # hidden_dim of the model
        hidden_dim = 512  # number of features we want to learn
        sae = train_sae(
            [cache[get_act_name("mlp_out", i)] for i in range(model.cfg.n_layers)],
            input_dim,
            hidden_dim
        )
        
        # Analyze all layers
        for layer_idx in range(model.cfg.n_layers):
            features = analyze_sae_features(sae, cache, input_text, layer_idx)
            visualize_feature_importance(features, layer_idx, input_text)
            
            # Find most active features
            feature_means = torch.mean(torch.abs(features), dim=(0, 1))
            top_features = torch.argsort(feature_means, descending=True)[:5]
            
            print(f"\nTop 5 most active SAE features in layer {layer_idx}:")
            for feature in top_features:
                print(f"Feature {feature}: {feature_means[feature]:.4f}")

if __name__ == "__main__":
    main()
