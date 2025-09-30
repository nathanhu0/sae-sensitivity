import sys
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd

# Ensure src is on path
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / 'src'))

# Import the main functions from each module
from saebench_utils import bootstrap_saebench_examples
from collect_examples import collect_feature_examples
from generate import generate_samples_from_examples
from sae_inference import run_sae_inference_on_samples





def aggregate_feature_stats(generated_df: pd.DataFrame, original_df: pd.DataFrame, 
                           sae_name: str, output_path: str, eps=1e-6) -> pd.DataFrame:
    """Calculate sensitivity for both datasets."""
    # Ensure feature_index is int in both DataFrames
    generated_df['feature_index'] = generated_df['feature_index'].astype(int)
    original_df['feature_index'] = original_df['feature_index'].astype(int)
    
    # Calculate sensitivities using max_activation field
    generated_df['feature_is_active'] = generated_df['max_activation'] > eps
    original_df['feature_is_active'] = original_df['max_activation'] > eps
    
    # Group by feature_index and calculate mean activation rate (sensitivity)
    gen_sens = generated_df.groupby('feature_index')['feature_is_active'].mean().reset_index()
    gen_sens.columns = ['feature_index', 'generated_sensitivity']
    
    orig_sens = original_df.groupby('feature_index')['feature_is_active'].mean().reset_index()
    orig_sens.columns = ['feature_index', 'original_sensitivity']
    
    # Merge on feature_index to properly align sensitivities
    stats_df = gen_sens.merge(orig_sens, on='feature_index', how='outer')

    # Save
    sae_dir = os.path.join(".", output_path, sae_name)
    stats_df.to_csv(os.path.join(sae_dir, "feature_stats.csv"), index=False)
    
    return stats_df


def process_single_sae(sae_name: str, output_path: str, features: Optional[int], 
                       llm_model: str, max_workers: int, model, use_saebench_examples: bool = True,
                       verbose: bool = True, dataset_name: str = "monology/pile-uncopyrighted", total_tokens: int = 2_000_000,
                       context_size: int = 128, n_top_ex_for_generation: int = 10,
                       n_iw_ex_for_generation: int = 5) -> None:
    """Process a single SAE through the entire pipeline.
    
    Args:
        sae_name: SAE configuration name
        output_path: Output directory path  
        features: Number of features to sample
        llm_model: OpenAI model for text generation
        max_workers: Max concurrent API requests
        model: The loaded transformer model
        use_saebench_examples: Whether to use pre-computed SAEBench examples vs compute fresh
        verbose: Whether to print progress
    """
    
    print(f"Processing {sae_name}")
    print("-" * 40)
    
    # Create output directory for this SAE
    sae_dir = os.path.join(".", output_path, sae_name)
    os.makedirs(sae_dir, exist_ok=True)
    
    # Load SAE using simplified interface
    from sae_inference import load_sae
    device = next(model.parameters()).device
    sae = load_sae(sae_name, device=device.type)
    
    # 1. Collect feature examples
    if verbose: print(f"Step 1: Collecting feature examples ({'from SAEBench' if use_saebench_examples else 'computing fresh'})...")
    if use_saebench_examples:
        examples_df = bootstrap_saebench_examples(sae_name, output_path, features)
    else:
        examples_df = collect_feature_examples(
            model, sae, sae_name, output_path, 
            features=features,
            dataset_name=dataset_name,
            total_tokens=total_tokens,
            context_size=context_size,
            n_top_ex_for_generation=n_top_ex_for_generation,
            n_iw_ex_for_generation=n_iw_ex_for_generation
        )
    
    # 2. Generate samples
    if verbose: print("Step 2: Generating samples via LLM...")
    _, generated_samples_df = generate_samples_from_examples(
        examples_df, sae_name, output_path, llm_model, max_workers
    )
    
    # 3. Run inference on generated samples
    if verbose: print("Step 3: Running SAE inference on generated samples...")
    generated_with_acts = run_sae_inference_on_samples(
        generated_samples_df, model, sae, sae_name, output_path,
        "generated_samples_with_activations.csv"
    )
    
    # 4. Run inference on original examples
    if verbose: print("Step 4: Running SAE inference on original examples...")
    original_with_acts = run_sae_inference_on_samples(
        examples_df, model, sae, sae_name, output_path,
        "original_examples_with_activations.csv"
    )
    
    # 5. Aggregate stats
    if verbose: print("Step 5: Aggregating feature statistics...")
    aggregate_feature_stats(
        generated_with_acts, original_with_acts, sae_name, output_path
    )


def main():
    parser = argparse.ArgumentParser(description="Run SAE evaluation pipeline")
    
    # Model configuration
    parser.add_argument("--model", type=str, required=True,
                       choices=["pythia-160m-deduped", "gemma-2-2b"],
                       help="Model to use for SAE evaluation")
    
    # SAE configuration - just pass explicit SAE names
    parser.add_argument("--sae-names", type=str, nargs="+", required=True,
                       help="SAE names to evaluate (e.g., py_4k_btopk_s0 gem_16k_gated_s2 gscope_16k_l0_128)")
    
    # Output path
    parser.add_argument("--output-path", type=str, default=None,
                       help="Output path for results (default: evals/{sae_names}_{date})")
    
    # Example collection
    parser.add_argument("--use-saebench-examples", action="store_true", default=False,
                       help="Use pre-computed SAEBench examples vs compute fresh (default: False)")
    parser.add_argument("--dataset-name", type=str, default="monology/pile-uncopyrighted",
                       help="HuggingFace dataset for fresh example collection (e.g. 'monology/pile-uncopyrighted', 'Skylion007/openwebtext')")
    parser.add_argument("--total-tokens", type=int, default=2_000_000,
                       help="Total tokens to process for fresh collection")
    parser.add_argument("--context-size", type=int, default=128,
                       help="Context size for example collection")
    parser.add_argument("--n-top-ex-for-generation", type=int, default=10,
                       help="Number of top activating examples per feature")
    parser.add_argument("--n-iw-ex-for-generation", type=int, default=5,
                       help="Number of importance-weighted examples per feature")
    
    # Generation parameters
    parser.add_argument("--features", type=int, default=50, 
                       help="Number of features to sample (None for all)")
    parser.add_argument("--llm-model", type=str, default="gpt-4.1-mini",
                       help="OpenAI model for text generation")
    parser.add_argument("--max-workers", type=int, default=50,
                       help="Max concurrent API requests")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Print progress for each step (default: True)")
    
    args = parser.parse_args()
    
    # Safety check for SAEBench examples
    if args.use_saebench_examples:
        if (args.total_tokens != 2_000_000 or 
            args.n_top_ex_for_generation != 10 or 
            args.n_iw_ex_for_generation != 5 or
            args.dataset_name != 'Skylion007/openwebtext'):
            print("ERROR: When using --use-saebench-examples, must use SAEBench's exact data settings")
            print("(2M tokens, 10 top examples, 5 IW examples, openwebtext dataset)")
            sys.exit(1)
    
    # Use the explicit SAE names provided
    sae_configs = args.sae_names
    
    # Load model once
    from sae_inference import init_model
    print(f"Loading model {args.model}...")
    model = init_model(model_name=args.model)
    
    # Set output path
    if args.output_path is None:
        # Create default path with SAE info and date
        sae_str = "_".join(args.sae_names[:2]) if len(args.sae_names) > 1 else args.sae_names[0]
        if len(args.sae_names) > 2:
            sae_str += f"_and_{len(args.sae_names)-2}_more"
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"evals/{sae_str}_{date_str}"
    else:
        output_path = args.output_path
    
    # Print configuration summary
    print(f"\n{'='*60}")
    print(f"SAE Evaluation Pipeline")
    print(f"{'='*60}")
    print(f"Output path: {output_path}")
    print(f"Features to sample: {args.features}")
    print(f"LLM model: {args.llm_model}")
    print(f"Configurations to process: {len(sae_configs)}")
    for sae_name in sae_configs:
        print(f"  - {sae_name}")
    print(f"{'='*60}\n")
    
    # Process each SAE completely before moving to the next
    print("\n=== PROCESSING SAEs ===\n")
    
    for i, sae_name in enumerate(sae_configs, 1):
        print(f"[{i}/{len(sae_configs)}] ", end="")
        process_single_sae(
            sae_name=sae_name,
            output_path=output_path,
            features=args.features,
            llm_model=args.llm_model,
            max_workers=args.max_workers,
            model=model,
            use_saebench_examples=args.use_saebench_examples,
            verbose=args.verbose,
            dataset_name=args.dataset_name,
            total_tokens=args.total_tokens,
            context_size=args.context_size,
            n_top_ex_for_generation=args.n_top_ex_for_generation,
            n_iw_ex_for_generation=args.n_iw_ex_for_generation
        )
        print()
    
    print(f"\n{'='*60}")
    print(f"Pipeline completed successfully!")
    print(f"Results saved to: {output_path}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()