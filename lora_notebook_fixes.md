# Critical Fixes for LoRA Notebook

## 1. Missing Import
Add after the other imports in cell 4:
```python
import copy  # Add this import
```

## 2. Missing Full Fine-tuning Implementation
Add before the LoRA training section (after cell 16):

```python
### Full Fine-tuning for Comparison

# First, let's implement full fine-tuning to have a baseline for comparison
print("=== FULL FINE-TUNING ===")

# Tokenize datasets for efficient training
def tokenize_dataset(dataset, tokenizer, max_length=128):
    """Pre-tokenize dataset for faster training"""
    tokenized = []
    for item in tqdm(dataset, desc="Tokenizing"):
        inputs = tokenizer(
            item['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=max_length,
            return_tensors='pt'
        )
        tokenized.append({
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': item['label']
        })
    return tokenized

# Tokenize train and test sets
print("Tokenizing datasets...")
tokenized_train = tokenize_dataset(train_dataset, tokenizer)
tokenized_test = tokenize_dataset(test_dataset, tokenizer)

# Full fine-tuning baseline
full_ft_model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

# Train with full fine-tuning
optimizer_full = torch.optim.AdamW(full_ft_model.parameters(), lr=2e-5)

def train_full_finetuning(model, train_data, test_data, epochs=2):
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        for i in tqdm(range(0, len(train_data), 16), desc=f"Epoch {epoch+1}"):
            batch_indices = list(range(i, min(i+16, len(train_data))))
            
            input_ids = torch.stack([torch.tensor(train_data[j]['input_ids']) for j in batch_indices])
            attention_mask = torch.stack([torch.tensor(train_data[j]['attention_mask']) for j in batch_indices])
            labels = torch.tensor([train_data[j]['label'] for j in batch_indices])
            
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            
            optimizer_full.zero_grad()
            loss.backward()
            optimizer_full.step()
    
    training_time = time.time() - start_time
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, min(500, len(test_data)), 16):
            batch_indices = list(range(i, min(i+16, len(test_data))))
            
            input_ids = torch.stack([torch.tensor(test_data[j]['input_ids']) for j in batch_indices])
            attention_mask = torch.stack([torch.tensor(test_data[j]['attention_mask']) for j in batch_indices])
            labels = torch.tensor([test_data[j]['label'] for j in batch_indices])
            
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(-1)
            
            correct += (predictions == labels).sum().item()
            total += len(labels)
    
    accuracy = correct / total
    return accuracy, training_time

print("Training full fine-tuning model...")
full_finetuning_accuracy, full_finetuning_time = train_full_finetuning(
    full_ft_model, tokenized_train, tokenized_test, epochs=2
)

print(f"Full fine-tuning accuracy: {full_finetuning_accuracy:.3f}")
print(f"Full fine-tuning time: {full_finetuning_time:.1f} seconds")
```

## 3. Fix the Parameter Count Display
In cell 18, the LoRA parameter calculation is incorrect. Fix it:

```python
# Replace this line:
print(f"{'LoRA':<20} {final_accuracy:<10.3f} {lora_time:<10.1f} {len(lora_parameters)*2:,}{'':>5}")

# With:
lora_param_count = sum(p.numel() for p in lora_parameters)
print(f"{'LoRA':<20} {final_accuracy:<10.3f} {lora_time:<10.1f} {lora_param_count:,}{'':>5}")
```

## 4. Add Visual Diagram Cell
Add after cell 0 (the introduction):

```python
### Visual Overview of LoRA

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Traditional Fine-tuning
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Traditional Fine-tuning', fontsize=16, fontweight='bold')

# Draw model
model_box = FancyBboxPatch((2, 3), 6, 4, 
                          boxstyle="round,pad=0.1",
                          facecolor='lightcoral',
                          edgecolor='darkred',
                          linewidth=2)
ax1.add_patch(model_box)
ax1.text(5, 5, 'Full Model\n(66M params)', ha='center', va='center', fontsize=12, fontweight='bold')
ax1.text(5, 1.5, '❌ All parameters updated\n❌ High memory usage\n❌ Slow training', 
         ha='center', va='center', fontsize=10)

# LoRA Approach
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('LoRA Fine-tuning', fontsize=16, fontweight='bold')

# Frozen model
frozen_box = FancyBboxPatch((1, 3), 4, 4,
                           boxstyle="round,pad=0.1",
                           facecolor='lightblue',
                           edgecolor='darkblue',
                           linewidth=2)
ax2.add_patch(frozen_box)
ax2.text(3, 5, 'Frozen Model\n(66M params)', ha='center', va='center', fontsize=11)

# LoRA adapter
lora_box = FancyBboxPatch((6, 3.5), 3, 3,
                         boxstyle="round,pad=0.1",
                         facecolor='lightgreen',
                         edgecolor='darkgreen',
                         linewidth=2)
ax2.add_patch(lora_box)
ax2.text(7.5, 5, 'LoRA\n(0.6M params)', ha='center', va='center', fontsize=11, fontweight='bold')

# Connection
arrow = ConnectionPatch((5, 5), (6, 5), "data", "data",
                       arrowstyle="->", shrinkA=5, shrinkB=5,
                       mutation_scale=20, fc="black", lw=2)
ax2.add_artist(arrow)

ax2.text(5, 1.5, '✅ Only LoRA updated\n✅ Low memory usage\n✅ Fast training', 
         ha='center', va='center', fontsize=10, color='darkgreen')

plt.tight_layout()
plt.show()
```

## 5. Add Interactive Rank Visualization
Add after the LoRA math section (after cell 12):

```python
### Interactive: How Rank Affects Approximation Quality

def interactive_rank_demo():
    # Create a sample "weight update" pattern
    np.random.seed(42)
    
    # Create a structured weight update (not random)
    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x, y)
    
    # Create a pattern with clear structure
    Z = np.sin(5 * X) * np.cos(5 * Y) + 0.5 * np.sin(10 * X * Y)
    
    # Normalize
    Z = (Z - Z.mean()) / Z.std()
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(Z)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    ranks = [1, 2, 4, 8, 16, 32, 50]
    
    # Original
    im = axes[0].imshow(Z, cmap='coolwarm', vmin=-2, vmax=2)
    axes[0].set_title('Original ΔW', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Different rank approximations
    for i, rank in enumerate(ranks[:-1], 1):
        Z_approx = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
        axes[i].imshow(Z_approx, cmap='coolwarm', vmin=-2, vmax=2)
        axes[i].set_title(f'Rank-{rank} LoRA', fontsize=12)
        axes[i].axis('off')
        
        # Calculate approximation error
        error = np.linalg.norm(Z - Z_approx, 'fro') / np.linalg.norm(Z, 'fro')
        axes[i].text(0.5, -0.1, f'Error: {error:.2%}', 
                    transform=axes[i].transAxes, 
                    ha='center', fontsize=10)
    
    # Error vs rank plot
    errors = []
    all_ranks = range(1, 51)
    for r in all_ranks:
        Z_approx = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
        error = np.linalg.norm(Z - Z_approx, 'fro') / np.linalg.norm(Z, 'fro')
        errors.append(error * 100)
    
    axes[7].plot(all_ranks, errors, 'b-', linewidth=2)
    axes[7].axvline(x=16, color='r', linestyle='--', alpha=0.7, label='LoRA default (r=16)')
    axes[7].set_xlabel('Rank')
    axes[7].set_ylabel('Relative Error (%)')
    axes[7].set_title('Approximation Error vs Rank')
    axes[7].grid(True, alpha=0.3)
    axes[7].legend()
    axes[7].set_ylim(0, max(errors) * 1.1)
    
    plt.colorbar(im, ax=axes[:7], fraction=0.046, pad=0.04)
    plt.suptitle('LoRA Approximation Quality with Different Ranks', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("💡 Key Insight: Even with rank=16 (0.64% of full rank), we can capture most of the weight update!")

interactive_rank_demo()
```

## 6. Add Summary Statistics
Add at the very end of the notebook:

```python
### 📊 Final Summary

# Create a comprehensive summary table
import pandas as pd

summary_data = {
    'Method': ['Baseline (Random)', 'Full Fine-tuning', 'LoRA (r=16)'],
    'Accuracy': [baseline_acc, full_finetuning_accuracy, final_accuracy],
    'Parameters': [0, 66_955_010, lora_param_count],
    'Training Time (s)': ['N/A', full_finetuning_time, lora_time],
    'Memory Usage': ['Inference only', 'High', 'Low'],
    'Inference Speed': ['Fast', 'Fast', 'Fast']
}

df = pd.DataFrame(summary_data)
print("📈 Performance Comparison")
print("=" * 70)
print(df.to_string(index=False))
print("=" * 70)

# Key metrics
param_reduction = (1 - lora_param_count / 66_955_010) * 100
accuracy_retained = (final_accuracy / full_finetuning_accuracy) * 100
speedup = full_finetuning_time / lora_time

print(f"\n🎯 LoRA Achievements:")
print(f"  • Parameter Reduction: {param_reduction:.1f}%")
print(f"  • Accuracy Retained: {accuracy_retained:.1f}%")
print(f"  • Training Speedup: {speedup:.1f}x")
print(f"  • Can run on: Consumer GPUs with limited memory")
```

## 7. Add "Try It Yourself" Section
Add as a new cell at the end:

```python
### 🚀 Try It Yourself!

# Experiment 1: Different Ranks
print("Experiment 1: Try different rank values")
print("Uncomment and run:")
print("""
# for rank in [1, 4, 8, 32, 64]:
#     model = create_lora_model_with_rank(baseline_model, rank=rank)
#     # Train and evaluate...
""")

# Experiment 2: Different Learning Rates
print("\nExperiment 2: Try different learning rates")
print("LoRA often works better with higher learning rates!")

# Experiment 3: Apply to Different Layers
print("\nExperiment 3: Apply LoRA to different layers")
print("Try applying LoRA only to specific layers or including FFN layers")

# Challenge: Multi-task LoRA
print("\n🏆 Challenge: Implement Multi-task LoRA")
print("Train multiple LoRA adapters for different tasks on the same base model!")
```