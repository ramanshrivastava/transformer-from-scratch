"""
Attention Mechanism Visualization
==================================
Create beautiful visualizations of the attention mechanism
Perfect for sharing on LinkedIn and educational purposes
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from attention_mechanism import (
    MultiHeadAttention,
    TransformerBlock,
    PositionalEncoding,
    ScaledDotProductAttention
)

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def visualize_attention_weights(attention_weights, tokens=None, save_path='attention_heatmap.png'):
    """
    Visualize attention weights as a heatmap
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Multi-Head Attention Visualization (8 Heads)', fontsize=16, fontweight='bold')

    for head_idx in range(8):
        ax = axes[head_idx // 4, head_idx % 4]

        # Take first batch
        weights = attention_weights[0, head_idx].detach().numpy()

        # Create heatmap
        im = ax.imshow(weights, cmap='Blues', aspect='auto')
        ax.set_title(f'Head {head_idx + 1}', fontsize=12)

        if tokens:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_yticklabels(tokens)
        else:
            ax.set_xlabel('Keys')
            ax.set_ylabel('Queries')

        # Add colorbar for each subplot
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Attention heatmap saved as '{save_path}'")


def visualize_positional_encoding(d_model=512, max_len=100, save_path='positional_encoding.png'):
    """
    Visualize the sinusoidal positional encoding pattern
    """
    pos_encoder = PositionalEncoding(d_model, max_len)
    pos_encoding = pos_encoder.pe[0, :max_len].numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Positional Encoding Visualization', fontsize=16, fontweight='bold')

    # Show first 64 dimensions as heatmap
    im1 = ax1.imshow(pos_encoding[:, :64].T, cmap='RdBu', aspect='auto')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Dimension')
    ax1.set_title('Positional Encoding Pattern (First 64 Dimensions)')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Show specific dimension patterns
    dimensions_to_plot = [0, 1, 4, 5, 10, 11]
    for dim in dimensions_to_plot:
        ax2.plot(pos_encoding[:, dim], label=f'Dim {dim}', linewidth=2)

    ax2.set_xlabel('Position')
    ax2.set_ylabel('Encoding Value')
    ax2.set_title('Sinusoidal Patterns for Selected Dimensions')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Positional encoding visualization saved as '{save_path}'")


def demonstrate_attention_on_sentence():
    """
    Demonstrate attention on a real sentence
    """
    print("\n" + "="*70)
    print("ATTENTION MECHANISM ON SAMPLE SENTENCE")
    print("="*70)

    # Sample sentence tokens
    tokens = ["The", "transformer", "architecture", "revolutionized", "NLP", "and", "AI", "."]
    seq_len = len(tokens)

    print(f"\nInput Sentence: {' '.join(tokens)}")
    print(f"Number of tokens: {seq_len}")

    # Create random embeddings for demonstration
    torch.manual_seed(42)
    d_model = 128  # Smaller for demonstration
    n_heads = 8

    # Simulate word embeddings
    embeddings = torch.randn(1, seq_len, d_model)

    # Add positional encoding
    pos_encoder = PositionalEncoding(d_model, seq_len)
    embeddings_with_pos = pos_encoder(embeddings)

    # Apply multi-head attention
    attention = MultiHeadAttention(d_model, n_heads)
    output, attention_weights = attention(
        embeddings_with_pos,
        embeddings_with_pos,
        embeddings_with_pos
    )

    print(f"\nAttention Output Shape: {output.shape}")
    print(f"Attention Weights Shape: {attention_weights.shape}")

    # Visualize attention for this sentence
    visualize_attention_weights(attention_weights, tokens, 'sentence_attention.png')

    # Analyze which tokens pay most attention to which
    avg_attention = attention_weights[0].mean(dim=0).detach().numpy()

    print("\n" + "="*70)
    print("ATTENTION ANALYSIS")
    print("="*70)

    for i, token_from in enumerate(tokens):
        top_3_indices = np.argsort(avg_attention[i])[-3:][::-1]
        top_3_tokens = [tokens[idx] for idx in top_3_indices]
        top_3_scores = [avg_attention[i][idx] for idx in top_3_indices]

        print(f"\n'{token_from}' pays most attention to:")
        for tok, score in zip(top_3_tokens, top_3_scores):
            print(f"  • '{tok}' (score: {score:.3f})")


def create_architecture_diagram():
    """
    Create a simple architecture diagram
    """
    fig, ax = plt.subplots(figsize=(10, 12))
    fig.suptitle('Transformer Architecture Overview', fontsize=16, fontweight='bold')

    # Define components and their positions
    components = [
        ("Input Embeddings", 0.5, 0.1, 'lightblue'),
        ("Positional Encoding", 0.5, 0.2, 'lightgreen'),
        ("Multi-Head Attention", 0.5, 0.4, 'salmon'),
        ("Add & Norm", 0.5, 0.5, 'lightyellow'),
        ("Feed Forward", 0.5, 0.6, 'lightcoral'),
        ("Add & Norm", 0.5, 0.7, 'lightyellow'),
        ("Output", 0.5, 0.9, 'lightgray'),
    ]

    # Draw components
    for name, x, y, color in components:
        rect = plt.Rectangle((x-0.15, y-0.03), 0.3, 0.06,
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontweight='bold', fontsize=10)

    # Draw connections
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    for i in range(len(components)-1):
        ax.annotate('', xy=(0.5, components[i+1][2]-0.03),
                   xytext=(0.5, components[i][2]+0.03),
                   arrowprops=arrow_props)

    # Draw residual connections
    residual_props = dict(arrowstyle='->', lw=1.5, color='blue', alpha=0.6)
    # Residual around attention
    ax.annotate('', xy=(0.7, 0.5), xytext=(0.7, 0.3),
               arrowprops=residual_props)
    # Residual around FFN
    ax.annotate('', xy=(0.7, 0.7), xytext=(0.7, 0.5),
               arrowprops=residual_props)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Add description
    description = """
    Key Components:
    • Self-Attention: Allows each position to attend to all positions
    • Multi-Head: Multiple attention mechanisms in parallel
    • Positional Encoding: Adds position information
    • Feed Forward: Two linear transformations with ReLU
    • Residual Connections: Help with gradient flow
    """
    ax.text(0.05, 0.02, description, fontsize=9, verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig('transformer_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Architecture diagram saved as 'transformer_architecture.png'")


if __name__ == "__main__":
    print("\n" + "🚀 "*20)
    print("ATTENTION IS ALL YOU NEED - VISUALIZATION")
    print("🚀 "*20)

    # 1. Visualize positional encoding
    print("\n1. Creating Positional Encoding Visualization...")
    visualize_positional_encoding()

    # 2. Demonstrate attention on a sentence
    print("\n2. Demonstrating Attention on Sample Sentence...")
    demonstrate_attention_on_sentence()

    # 3. Create architecture diagram
    print("\n3. Creating Architecture Diagram...")
    create_architecture_diagram()

    print("\n" + "="*70)
    print("READY FOR LINKEDIN!")
    print("="*70)
    print("""
    You now have:
    ✓ attention_mechanism.py - Clean implementation
    ✓ visualize_attention.py - Visualization code
    ✓ attention_heatmap.png - Multi-head attention visualization
    ✓ positional_encoding.png - Positional encoding patterns
    ✓ sentence_attention.png - Attention on real sentence
    ✓ transformer_architecture.png - Architecture overview

    Perfect for sharing your understanding of transformers!

    Suggested LinkedIn post:
    ------------------------
    🧠 Implemented the Attention Mechanism from "Attention Is All You Need"

    Just created a clean Python implementation of the transformer architecture
    that revolutionized NLP and gave us ChatGPT, BERT, and more!

    Key insights:
    • Self-attention captures long-range dependencies
    • Multi-head attention learns different relationships
    • Positional encoding preserves sequence order
    • Highly parallelizable architecture

    Code available on GitHub: [your-repo]

    #MachineLearning #NLP #Transformers #AI #DeepLearning
    """)