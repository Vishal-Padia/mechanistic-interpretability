"""
Here's what I want to do:
1. Use any small text generation model from huggingface
2. Capture activations
3. Change activation values
4. See new results
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from torch import nn
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B")

# capture activations
activations = []

def hook_fn(module, input, output):
    if isinstance(output, tuple):
        hidden_states = output[0] # (batch_size, sequence_length, hidden_dim)
        activations.append(hidden_states.detach().cpu().numpy())
    else:
        activations.append(output.detach().cpu().numpy()) # (batch_size, sequence_length, hidden_dim)

# hook the layers
for layer in model.model.layers:
    layer.register_forward_hook(hook_fn)

def visualize_activations(activations, input_text, layer_idx=0):
    # get tokens for x-axis labels
    tokens = tokenizer.tokenize(input_text)
    print(f"Number of tokens: {len(tokens)}")  # sequence_length

    # get activation data for specified layer
    layer_activations = activations[layer_idx]  # (batch_size, sequence_length, hidden_dim)

    # compute mean activation across the hidden dimension
    mean_activations = np.mean(layer_activations, axis=1) # (batch_size, hidden_dim)

    # Create heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(mean_activations, 
                xticklabels=tokens,
                yticklabels=False,
                cmap='viridis')
    plt.title(f'Activation Heatmap - Layer {layer_idx}')
    plt.xlabel('Input Tokens')
    plt.ylabel('Sequence Position')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'activation_heatmap_layer_{layer_idx}.png')
    plt.close()

def analyze_activation_patterns(activations, input_text):
    layer_means = [np.mean(np.abs(act)) for act in activations]
    print(f"Number of layers: {len(layer_means)}")  # number of transformer layers
    print(f"Layer means shape: {len(layer_means)}")  # list of length num_layers

    plt.figure(figsize=(10, 6))
    plt.plot(layer_means, marker='o')
    plt.title('Average Activation Magnitude Across Layers')
    plt.xlabel('Layer')
    plt.ylabel('Mean Activation Magnitude')
    plt.grid(True)
    plt.savefig(f'activation_magnitude_across_layers.png')
    plt.close()

def find_most_active_neurons(activations, layer_idx=0, top_k=10):
    layer_activations = activations[layer_idx] # (batch_size, sequence_length, hidden_dim)

    # compute mean activation for each neuron
    neuron_means = np.mean(np.abs(layer_activations), axis=(0, 1)) # (hidden_dim,)

    # get the top k neurons
    top_k_neurons = np.argsort(neuron_means)[-top_k:]

    # Get the corresponding activation values
    top_k_activations = neuron_means[top_k_neurons]

    return top_k_neurons, top_k_activations

# Test with different inputs
input_texts = [
    "Hello, how are you?",
    "The quick brown fox jumps over the lazy dog.",
    "I love machine learning and artificial intelligence."
]

for input_text in input_texts:
    print(f"\nAnalyzing input: {input_text}")
    
    # Clear previous activations
    activations.clear()
    
    # Get new activations
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    
    output = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    
    # Visualize activations for first layer
    visualize_activations(activations, input_text, layer_idx=0)
    
    # Analyze activation patterns
    analyze_activation_patterns(activations, input_text)
    
    # Find most active neurons
    top_neurons, neuron_activations = find_most_active_neurons(activations, layer_idx=0)
    print(f"Top {len(top_neurons)} most active neurons in layer 0:")
    for neuron, activation in zip(top_neurons, neuron_activations):
        print(f"Neuron {neuron}: {activation:.4f}")


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_weight=0.1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        # encode
        hidden = self.encoder(x)

        # apply L1 regularization for sparsity
        sparsity_loss = torch.mean(torch.abs(hidden))

        # decode
        reconstructed = self.decoder(hidden)
        return reconstructed, hidden, sparsity_loss

def train_sae(activations, input_dim, hidden_dim, num_epochs=100, batch_size=32):
    # prepare data
    all_activations = np.concatenate([act.reshape(-1, act.shape[-1]) for act in activations], axis=0) # (num_samples, hidden_dim)

    # intialize model and optimizer
    device = "cpu"
    sae = SparseAutoencoder(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(sae.parameters())

    # training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(all_activations), batch_size):
            batch = torch.FloatTensor(all_activations[i:i+batch_size]).to(device)

            # forward pass
            reconstructed, hidden, sparsity_loss = sae(batch)

            # compute reconstruction loss
            reconstruction_loss = F.mse_loss(reconstructed, batch)

            # total loss
            loss = reconstruction_loss + sae.sparsity_weight * sparsity_loss
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update total loss
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / (i + batch_size):.4f}")

    return sae

def visualize_activations(activations, input_text, layer_idx=0):
    """
    Enhanced visualization of activations with better formatting and more information
    """
    # get tokens for x-axis labels
    tokens = tokenizer.tokenize(input_text)
    print(f"Number of tokens: {len(tokens)}")  # sequence_length

    # get activation data for specified layer
    layer_activations = activations[layer_idx]  # (batch_size, sequence_length, hidden_dim)
    
    # compute mean activation across the hidden dimension
    mean_activations = np.mean(layer_activations, axis=1)  # (batch_size, hidden_dim)
    
    # Create figure with better styling
    plt.figure(figsize=(15, 8))
    
    # Create heatmap with better color scheme and formatting
    sns.heatmap(mean_activations, 
                xticklabels=tokens,
                yticklabels=False,
                cmap='RdYlBu_r',  # More intuitive color scheme
                center=0,  # Center the colormap at 0
                cbar_kws={'label': 'Activation Strength'})
    
    plt.title(f'Activation Heatmap - Layer {layer_idx}\nInput: "{input_text}"', 
             pad=20, fontsize=14)
    plt.xlabel('Input Tokens', fontsize=12, labelpad=10)
    plt.ylabel('Sequence Position', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    
    # Add grid lines for better readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'activation_heatmap_layer_{layer_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_activation_patterns(activations, input_text):
    """
    Enhanced visualization of activation patterns across layers
    """
    layer_means = [np.mean(np.abs(act)) for act in activations]
    print(f"Number of layers: {len(layer_means)}")  # number of transformer layers
    
    # Create figure with better styling
    plt.figure(figsize=(12, 6))
    
    # Plot with enhanced styling
    plt.plot(layer_means, marker='o', linestyle='-', linewidth=2, markersize=8)
    
    # Add gradient fill
    plt.fill_between(range(len(layer_means)), layer_means, alpha=0.3)
    
    # Customize appearance
    plt.title(f'Average Activation Magnitude Across Layers\nInput: "{input_text}"', 
             pad=20, fontsize=14)
    plt.xlabel('Layer', fontsize=12, labelpad=10)
    plt.ylabel('Mean Activation Magnitude', fontsize=12, labelpad=10)
    
    # Add grid and customize
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, len(layer_means), 2))  # Show every other layer number
    
    # Add annotations for min and max points
    max_idx = np.argmax(layer_means)
    min_idx = np.argmin(layer_means)
    plt.annotate(f'Max: {layer_means[max_idx]:.3f}', 
                xy=(max_idx, layer_means[max_idx]),
                xytext=(max_idx+1, layer_means[max_idx]),
                arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate(f'Min: {layer_means[min_idx]:.3f}', 
                xy=(min_idx, layer_means[min_idx]),
                xytext=(min_idx+1, layer_means[min_idx]),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.tight_layout()
    plt.savefig(f'activation_magnitude_across_layers.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_sae_features(sae, activations, input_text, layer_idx=0):
    """
    Enhanced visualization of SAE features
    """
    # Get activations for the specific layer
    layer_activations = activations[layer_idx]  # (batch_size, sequence_length, hidden_dim)
    
    # Reshape for SAE
    flattened_activations = layer_activations.reshape(-1, layer_activations.shape[-1])
    
    # Get SAE features
    with torch.no_grad():
        _, features, _ = sae(torch.FloatTensor(flattened_activations))
    
    # Reshape back to sequence
    features = features.reshape(layer_activations.shape[0], layer_activations.shape[1], -1)
    
    # Create figure with better styling
    plt.figure(figsize=(20, 10))
    
    # Create heatmap with better color scheme
    sns.heatmap(features[0].numpy(), 
                cmap='RdYlBu_r',
                center=0,
                cbar_kws={'label': 'Feature Activation Strength'})
    
    # Customize appearance
    plt.title(f'SAE Feature Activations - Layer {layer_idx}\nInput: "{input_text}"', 
             pad=20, fontsize=14)
    plt.xlabel('SAE Features', fontsize=12, labelpad=10)
    plt.ylabel('Sequence Position', fontsize=12, labelpad=10)
    
    # Add grid lines
    plt.grid(True, alpha=0.3)
    
    # Add feature importance annotations
    feature_means = np.mean(np.abs(features[0].numpy()), axis=0)
    top_features = np.argsort(feature_means)[-5:]
    
    for idx, feature in enumerate(top_features):
        plt.annotate(f'Top {idx+1}', 
                    xy=(feature, 0),
                    xytext=(feature, -2),
                    ha='center',
                    va='top',
                    color='red',
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'sae_features_layer_{layer_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return features

def visualize_feature_importance(features, layer_idx, input_text):
    """
    New function to visualize feature importance
    """
    # Calculate feature importance
    feature_importance = np.mean(np.abs(features.numpy()), axis=(0, 1))
    
    # Get top 20 features
    top_k = 20
    top_indices = np.argsort(feature_importance)[-top_k:]
    top_importance = feature_importance[top_indices]
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Create bar plot
    bars = plt.bar(range(top_k), top_importance, color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Customize appearance
    plt.title(f'Top {top_k} Most Important SAE Features - Layer {layer_idx}\nInput: "{input_text}"',
             pad=20, fontsize=14)
    plt.xlabel('Feature Index', fontsize=12, labelpad=10)
    plt.ylabel('Average Activation Magnitude', fontsize=12, labelpad=10)
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'feature_importance_layer_{layer_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Modify the main loop to include the new visualization
for input_text in input_texts:
    print(f"\nAnalyzing input: {input_text}")
    
    # Clear previous activations
    activations.clear()
    
    # Get new activations
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    output = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    
    # Train SAE on the activations
    input_dim = activations[0].shape[-1]  # hidden_dim of the model
    hidden_dim = 512  # number of features we want to learn
    sae = train_sae(activations, input_dim, hidden_dim)
    
    # Analyze SAE features
    for layer_idx in range(3):  # Analyze first 3 layers
        features = analyze_sae_features(sae, activations, input_text, layer_idx)
        
        # Add new feature importance visualization
        visualize_feature_importance(features, layer_idx, input_text)
        
        # Find most active features
        feature_means = np.mean(np.abs(features.numpy()), axis=(0, 1))
        top_features = np.argsort(feature_means)[-5:]  # top 5 features
        
        print(f"\nTop 5 most active SAE features in layer {layer_idx}:")
        for feature in top_features:
            print(f"Feature {feature}: {feature_means[feature]:.4f}")