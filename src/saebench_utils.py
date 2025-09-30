"""
Bootstrap feature examples from pre-computed SAEBench results.
This module is SAEBench-specific - it downloads and parses results from SAEBench's HuggingFace repository.
"""
import os
import re
import json
import random
from typing import Optional, Dict, Any
import pandas as pd
from huggingface_hub import hf_hub_download


def build_sae_map(type: str = "autointerp") -> Dict[str, str]:
    """
    Build mapping from sae_name -> relative path in SAEBench HuggingFace repository.
    
    This is SAEBench-specific: maps to adamkarvonen/sae_bench_results_0125 structure.
    """
    width_map = {"4k": "2pow12", "16k": "2pow14", "65k": "2pow16"}
    sae_type_to_folder = {
        "matbtopk": "MatryoshkaBatchTopK",
        "btopk": "BatchTopK",
        "jumprelu": "JumpRelu",
        "gated": "GatedSAE",
        "topk": "TopK",
        "relu": "Standard",
        "pan": "PAnneal",
    }
    model_to_folder = {"gem": "gemma-2-2b", "py": "pythia-160m-deduped"}
    type_to_folder = {"autointerp": "autointerp_with_generations", "core": "core_with_feature_statistics"}

    sae_map: Dict[str, str] = {}
    for model in ["gem", "py"]:
        layer = 12 if model == "gem" else 8
        for width in ["4k", "16k", "65k"]:
            for sae_type in ["btopk", "matbtopk", "gated", "jumprelu", "topk", "relu", "pan"]:
                for sparsity in range(6):
                    sae_name = f"{model}_{width}_{sae_type}_s{sparsity}"
                    folder_name = f"saebench_{model_to_folder[model]}_width-{width_map[width]}_date-0108"
                    sae_map[sae_name] = (
                        f"{type_to_folder[type]}/{folder_name}/"
                        f"{folder_name}_{sae_type_to_folder[sae_type]}_{model_to_folder[model]}__0108"
                        f"_resid_post_layer_{layer}_trainer_{sparsity}_eval_results.json"
                    )
    return sae_map


def parse_saebench_json(file_path: str) -> pd.DataFrame:
    """
    Parse SAEBench AutoInterp JSON to extract feature examples.
    
    This is SAEBench-specific: parses the exact JSON structure from SAEBench evaluations.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)

    if 'eval_result_unstructured' not in data:
        raise ValueError("No 'eval_result_unstructured' field found in JSON")

    all_rows = []
    
    for feature_idx, feature_data in data['eval_result_unstructured'].items():
        if 'logs' not in feature_data:
            continue
            
        logs = feature_data['logs']
        
        # Find generation examples in SAEBench's log format
        if "Generation phase" not in logs:
            continue
            
        gen_start = logs.find("Generation phase")
        gen_end = logs.find("Scoring phase") if "Scoring phase" in logs else len(logs)
        gen_section = logs[gen_start:gen_end]
        
        # Parse SAEBench's table format: │ <number> │ <text> │
        lines = gen_section.split('\n')
        for line in lines:
            if '│' not in line or 'Top act' in line or '─' in line:
                continue
                
            line = line.strip()
            if line.startswith('│'):
                line = line[1:]
            if line.endswith('│'):
                line = line[:-1]
                
            parts = [p.strip() for p in line.split('│')]
            if len(parts) >= 2:
                try:
                    top_act = float(parts[0])
                    sequence = parts[1]
                    # Convert SAEBench's <<>> to our {{}}
                    sequence = sequence.replace("<<", "{{").replace(">>", "}}")
                    
                    all_rows.append({
                        'feature_index': feature_idx,
                        'Top_act': top_act,
                        'Sequence': sequence
                    })
                except (ValueError, IndexError):
                    continue

    if not all_rows:
        return pd.DataFrame(columns=['feature_index', 'Top_act', 'Sequence'])

    df = pd.DataFrame(all_rows)
    return df[['feature_index', 'Top_act', 'Sequence']]


def bootstrap_saebench_examples(sae_name: str, output_path: str, features: Optional[int] = None) -> pd.DataFrame:
    """
    Download and parse feature examples from SAEBench pre-computed results.
    
    This entire function is SAEBench-specific: downloads from adamkarvonen/sae_bench_results_0125.
    
    Args:
        sae_name: SAE configuration name (e.g., "py_4k_btopk_s0")
        output_path: Output directory path
        features: Number of features to sample (None for all)
    
    Returns:
        DataFrame with columns: feature_index, Top_act, Sequence
    """
    # Build SAEBench repository map
    sae_map = build_sae_map()
    if sae_name not in sae_map:
        available = list(sae_map.keys())[:5]
        raise ValueError(f"Unknown sae_name: {sae_name}. Examples: {available}")
    
    # Download from SAEBench HuggingFace repository
    filename = sae_map[sae_name]
    json_path = hf_hub_download(
        repo_id='adamkarvonen/sae_bench_results_0125',
        filename=filename,
        repo_type='dataset'
    )
    
    # Parse SAEBench JSON format
    examples_df = parse_saebench_json(json_path)
    
    if examples_df.empty:
        raise ValueError(f"No examples found in SAEBench results for {sae_name}")
    
    # Sample features if requested
    if features is not None:
        unique_features = examples_df['feature_index'].unique()
        total = len(unique_features)
        if features < total:
            random.seed(42)
            selected = set(random.sample(list(unique_features), features))
            examples_df = examples_df[examples_df['feature_index'].isin(selected)]
            print(f"Sampled {features} features from {total} available")
    
    # Save to file
    sae_dir = os.path.join(".", output_path, sae_name)
    os.makedirs(sae_dir, exist_ok=True)
    output_file = os.path.join(sae_dir, "feature_examples.csv")
    examples_df.to_csv(output_file, index=False)
    print(f"Saved {len(examples_df)} examples to {output_file}")
    
    return examples_df