"""
Fresh collection of activating examples using AutoInterp methodology.
This is vendored and stripped-down from the sae_bench implementation.
Only includes the text collection functionality, not the LLM scoring.
"""

import os
import random
import torch
import pandas as pd
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
from torch import Tensor
from transformer_lens import HookedTransformer


# ============================================================================
# Example and Examples classes from sae_bench autointerp/main.py
# ============================================================================

class Example:
    """
    Data for a single example sequence.
    Vendored from sae_bench autointerp/main.py
    """

    def __init__(
        self,
        toks: list[int],
        acts: list[float],
        act_threshold: float,
        model: HookedTransformer,
    ):
        self.toks = toks
        self.str_toks = model.to_str_tokens(torch.tensor(self.toks))
        self.acts = acts
        self.act_threshold = act_threshold
        self.toks_are_active = [act > act_threshold for act in self.acts]
        self.is_active = any(self.toks_are_active)

    def to_str(self, mark_toks: bool = True) -> str:
        """Convert to string with optional token marking.
        Modified to use {{}} notation for compatibility with existing pipeline."""
        if mark_toks:
            # Use {{}} notation instead of <<>> for compatibility
            return (
                "".join(
                    f"{{{{{tok}}}}}" if is_active else tok
                    for tok, is_active in zip(self.str_toks, self.toks_are_active)
                )
                .replace("�", "")
                .replace("\n", "↵")
            )
        else:
            return (
                "".join(self.str_toks)
                .replace("�", "")
                .replace("\n", "↵")
            )


class Examples:
    """
    Data for multiple example sequences.
    Vendored from sae_bench autointerp/main.py
    """

    def __init__(self, examples: list[Example], shuffle: bool = False) -> None:
        self.examples = examples
        if shuffle:
            random.shuffle(self.examples)
        else:
            # Sort by max activation (highest first)
            self.examples = sorted(
                self.examples, key=lambda x: max(x.acts), reverse=True
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]


# ============================================================================
# Dataset loading and caching
# ============================================================================

def load_and_cache_dataset(
    model: HookedTransformer,
    dataset_name: str,
    total_tokens: int,
    context_size: int,
    cache_dir: str = "downloaded_datasets"
) -> torch.Tensor:
    """
    Load and tokenize dataset with caching.
    Based on run_eval_single_sae from autointerp/main.py
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache filename (following autointerp naming convention)
    model_name = model.cfg.model_name.replace("/", "_")
    dataset_name_safe = dataset_name.replace("/", "_")
    cache_file = os.path.join(
        cache_dir,
        f"{model_name}_{dataset_name_safe}_{total_tokens}_tokens_{context_size}_ctx.pt"
    )
    
    if os.path.exists(cache_file):
        print(f"Loading cached tokenized dataset from {cache_file}")
        return torch.load(cache_file)
    
    print(f"Loading and tokenizing {dataset_name} dataset...")
    
    # Use SAEBench's dataset loading - if you're using fresh collection, you need SAEBench anyway
    try:
        from sae_bench.sae_bench_utils import dataset_utils
    except ImportError:
        raise ImportError("sae_bench required for fresh example collection. Install with: pip install sae-bench")
    
    # Load and tokenize dataset using SAEBench utilities
    tokenized_dataset = dataset_utils.load_and_tokenize_dataset(
        dataset_name,
        context_size,
        total_tokens,
        model.tokenizer,
    )
    
    # Save to cache
    torch.save(tokenized_dataset, cache_file)
    print(f"Saved tokenized dataset to {cache_file}")
    
    return tokenized_dataset


# ============================================================================
# Feature collection logic from AutoInterp.gather_data()
# ============================================================================

def gather_activating_examples(
    tokenized_dataset: torch.Tensor,
    model: HookedTransformer,
    sae,
    feature_indices: List[int],
    n_top_ex: int = 20,
    n_iw_sampled_ex: int = 10,
    n_top_ex_for_generation: int = 10,
    n_iw_sampled_ex_for_generation: int = 5,
    buffer: int = 10,
    no_overlap: bool = True,
    act_threshold_frac: float = 0.01,
    batch_size: int = 32,
) -> Dict[int, Examples]:
    """
    Gather activating examples for specified features.
    This is the core logic from AutoInterp.gather_data(), 
    but only collecting generation examples (not scoring examples).
    """
    
    # Import utilities from sae_bench
    try:
        from sae_bench.sae_bench_utils import activation_collection
        from sae_bench.sae_bench_utils.indexing_utils import (
            get_k_largest_indices,
            get_iw_sample_indices,
            index_with_buffer
        )
    except ImportError:
        raise ImportError("sae_bench required. Install with: pip install sae-bench")
    
    device = next(model.parameters()).device
    tokenized_dataset = tokenized_dataset.to(device)
    dataset_size, seq_len = tokenized_dataset.shape
    
    print(f"\n" + "="*70)
    print(f"PASS 2: Collecting activating examples for {len(feature_indices)} selected features")
    print("="*70)
    print(f"For each feature, collecting:")
    print(f"  - Top {n_top_ex_for_generation} highest activating examples")
    print(f"  - {n_iw_sampled_ex_for_generation} importance-weighted samples")
    print(f"\nCollecting activations (this may take a few minutes)...\n")
    
    # Collect activations for selected features only
    # Note: collect_sae_activations internally uses tqdm for progress
    acts = activation_collection.collect_sae_activations(
        tokenized_dataset,
        model,
        sae,
        batch_size,
        sae.cfg.hook_layer,
        sae.cfg.hook_name,
        mask_bos_pad_eos_tokens=True,
        selected_latents=feature_indices,
        activation_dtype=torch.bfloat16,  # reduce memory usage
    )
    
    generation_examples = {}
    
    print(f"\nExtracting top examples for each feature...\n")
    for i, latent in enumerate(tqdm(feature_indices, desc="Processing features")):
        # Get top-scoring examples
        top_indices = get_k_largest_indices(
            acts[..., i],
            k=n_top_ex,
            buffer=buffer,
            no_overlap=no_overlap,
        )
        top_toks = index_with_buffer(
            tokenized_dataset, top_indices, buffer=buffer
        )
        top_values = index_with_buffer(
            acts[..., i], top_indices, buffer=buffer
        )
        
        # Skip if no significant activations found
        if top_values.max() < 1e-6:
            continue
            
        act_threshold = act_threshold_frac * top_values.max().item()
        
        # Get importance-weighted examples
        threshold = top_values[:, buffer].min().item()
        acts_thresholded = torch.where(acts[..., i] >= threshold, 0.0, acts[..., i])
        
        # Check if we have valid IW examples
        if acts_thresholded[:, buffer:-buffer].max() < 1e-6:
            # No valid IW examples, just use top examples
            # Random split indices for generation
            rand_top_ex_split_indices = torch.randperm(len(top_toks))
            top_gen_indices = rand_top_ex_split_indices[:n_top_ex_for_generation]
            
            examples = []
            for j in top_gen_indices:
                examples.append(Example(
                    toks=top_toks[j].tolist(),
                    acts=top_values[j].tolist(),
                    act_threshold=act_threshold,
                    model=model,
                ))
        else:
            # Get IW examples
            iw_indices = get_iw_sample_indices(
                acts_thresholded, k=n_iw_sampled_ex, buffer=buffer
            )
            iw_toks = index_with_buffer(
                tokenized_dataset, iw_indices, buffer=buffer
            )
            iw_values = index_with_buffer(
                acts[..., i], iw_indices, buffer=buffer
            )
            
            # Random split indices for generation
            rand_top_ex_split_indices = torch.randperm(n_top_ex)
            top_gen_indices = rand_top_ex_split_indices[:n_top_ex_for_generation]
            
            rand_iw_split_indices = torch.randperm(n_iw_sampled_ex)
            iw_gen_indices = rand_iw_split_indices[:n_iw_sampled_ex_for_generation]
            
            examples = []
            
            # Add top examples
            for j in top_gen_indices:
                examples.append(Example(
                    toks=top_toks[j].tolist(),
                    acts=top_values[j].tolist(),
                    act_threshold=act_threshold,
                    model=model,
                ))
            
            # Add IW examples
            for j in iw_gen_indices:
                examples.append(Example(
                    toks=iw_toks[j].tolist(),
                    acts=iw_values[j].tolist(),
                    act_threshold=act_threshold,
                    model=model,
                ))
        
        generation_examples[latent] = Examples(examples)
    
    return generation_examples


# ============================================================================
# Feature sparsity calculation (for filtering dead features)
# ============================================================================

def calculate_feature_sparsity(
    tokenized_dataset: torch.Tensor,
    model: HookedTransformer,
    sae,
    batch_size: int = 32
) -> torch.Tensor:
    """
    Calculate sparsity for all features in the SAE.
    Based on run_eval_single_sae from autointerp/main.py
    """
    print("\n" + "="*70)
    print("PASS 1: Calculating feature activation frequencies")
    print("="*70)
    print(f"Dataset shape: {tokenized_dataset.shape[0]:,} sequences × {tokenized_dataset.shape[1]} tokens")
    print(f"Total features in SAE: {sae.cfg.d_sae:,}")
    print(f"\nThis pass estimates how often each feature activates to identify")
    print(f"'alive' features (those that activate frequently enough).\n")
    
    try:
        from sae_bench.sae_bench_utils import activation_collection
    except ImportError:
        raise ImportError("sae_bench required. Install with: pip install sae-bench")
    
    device = next(model.parameters()).device
    tokenized_dataset = tokenized_dataset.to(device)
    
    # Note: get_feature_activation_sparsity internally uses tqdm
    sparsity = activation_collection.get_feature_activation_sparsity(
        tokenized_dataset,
        model,
        sae,
        batch_size,
        sae.cfg.hook_layer,
        sae.cfg.hook_name,
        mask_bos_pad_eos_tokens=True,
    )
    
    return sparsity


# ============================================================================
# Conversion to DataFrame format
# ============================================================================

def examples_to_dataframe(examples_dict: Dict[int, Examples]) -> pd.DataFrame:
    """
    Convert Examples dict to DataFrame format matching SAEBench.
    """
    rows = []
    for feature_idx, examples in examples_dict.items():
        for example in examples:
            rows.append({
                'feature_index': feature_idx,
                'Top_act': max(example.acts),
                'Sequence': example.to_str(mark_toks=True)
            })
    
    df = pd.DataFrame(rows)
    # Sort by feature_index and then by Top_act (descending)
    df = df.sort_values(['feature_index', 'Top_act'], ascending=[True, False])
    return df


# ============================================================================
# Main entry point
# ============================================================================

def collect_feature_examples(
    model: HookedTransformer,
    sae,
    sae_name: str,
    output_path: str,
    features: Optional[int] = 50,
    dataset_name: str = "monology/pile-uncopyrighted",
    total_tokens: int = 2_000_000,
    context_size: int = 128,
    dead_feature_threshold: int = 15,
    n_top_ex_for_generation: int = 10,
    n_iw_ex_for_generation: int = 5,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Main entry point for fresh example collection using AutoInterp methodology.
    
    This implements the two-pass approach:
    1. First pass: Calculate sparsity to identify alive features
    2. Second pass: Collect examples only for alive features
    
    Args:
        model: Transformer model
        sae: Sparse autoencoder
        sae_name: Name of SAE configuration
        output_path: Output directory path
        features: Number of features to sample (None for all alive features)
        dataset_name: HuggingFace dataset name (e.g. "monology/pile-uncopyrighted", "Skylion007/openwebtext")
        total_tokens: Total tokens to process
        context_size: Context window size
        dead_feature_threshold: Minimum activations to consider feature alive
        n_top_ex_for_generation: Number of top examples per feature
        n_iw_ex_for_generation: Number of importance-weighted examples per feature
        batch_size: Batch size for processing
    
    Returns:
        DataFrame with columns: feature_index, Top_act, Sequence
    """
    
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    
    # 1. Load/cache tokenized dataset
    tokenized_dataset = load_and_cache_dataset(
        model, dataset_name, total_tokens, context_size
    )
    
    # 2. First pass: Calculate feature sparsity
    sparsity = calculate_feature_sparsity(
        tokenized_dataset, model, sae, batch_size
    )
    
    # 3. Select alive features based on sparsity
    sparsity_counts = sparsity * total_tokens
    alive_features = (
        torch.nonzero(sparsity_counts > dead_feature_threshold)
        .squeeze(1)
        .tolist()
    )
    
    print(f"\n✓ Pass 1 complete: Found {len(alive_features):,} alive features")
    print(f"  (features with >{dead_feature_threshold} activations in {total_tokens:,} tokens)\n")
    
    if len(alive_features) == 0:
        print("WARNING: No alive features found! Try lowering dead_feature_threshold.")
        return pd.DataFrame(columns=['feature_index', 'Top_act', 'Sequence'])
    
    # Sample requested number of features from alive features
    if features is not None and features < len(alive_features):
        selected_features = sorted(random.sample(alive_features, features))
        print(f"Randomly sampling {features} features from {len(alive_features):,} alive features")
    else:
        selected_features = alive_features
        print(f"Using all {len(alive_features):,} alive features")
    
    # 4. Second pass: Collect examples for selected features
    # Following AutoInterp's default parameters
    n_top_ex = n_top_ex_for_generation + 2  # Extra for potential scoring phase
    n_iw_sampled_ex = n_iw_ex_for_generation + 2  # Extra for potential scoring phase
    
    examples_dict = gather_activating_examples(
        tokenized_dataset,
        model,
        sae,
        selected_features,
        n_top_ex=n_top_ex,
        n_iw_sampled_ex=n_iw_sampled_ex,
        n_top_ex_for_generation=n_top_ex_for_generation,
        n_iw_sampled_ex_for_generation=n_iw_ex_for_generation,
        batch_size=batch_size,
    )
    
    # 5. Convert to DataFrame format
    examples_df = examples_to_dataframe(examples_dict)
    
    print(f"\n✓ Pass 2 complete: Collected {len(examples_dict)} features with examples")
    
    # 6. Save to file
    sae_dir = os.path.join(".", output_path, sae_name)
    os.makedirs(sae_dir, exist_ok=True)
    output_file = os.path.join(sae_dir, "feature_examples.csv")
    examples_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved {len(examples_df)} total examples to {output_file}")
    print(f"  Average: {len(examples_df) / len(examples_dict):.1f} examples per feature")
    
    return examples_df