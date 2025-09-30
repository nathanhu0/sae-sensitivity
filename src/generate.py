"""
Module for generating samples from feature examples using LLM.
"""
import os
from typing import Tuple
import pandas as pd
from sae_gen_prompts import (
    get_system_prompt,
    build_core_prompt,
    build_user_prompt,
)
from sae_gen_samples import generate_samples


def generate_prompts_from_examples(examples_df: pd.DataFrame, sae_name: str, eval_name: str,
                                   num_per_call: int = 10, separator: str = "<SAMPLE_SEPARATOR/>") -> pd.DataFrame:
    """Generate prompts DataFrame from examples.
    
    Args:
        examples_df: DataFrame with columns ['feature_index', 'Top_act', 'Sequence']
        sae_name: SAE configuration name
        eval_name: Evaluation name  
        num_per_call: Number of samples per API call
        separator: Separator token between samples
    
    Returns:
        DataFrame with prompt information
    """
    # Group by feature_index
    feature_groups = examples_df.groupby('feature_index')
    
    rows = []
    for feature_idx, group in feature_groups:
        # Convert to list of dicts for prompt generation
        activations = []
        for _, row in group.iterrows():
            activations.append({
                'Top_act': row.get('Top_act', 0),
                'Sequence': row['Sequence']
            })
        
        # Generate prompts
        system_prompt = get_system_prompt("feature")
        core_prompt = build_core_prompt(
            sample_type="highly_relevant",
            num_per_call=num_per_call,
            feature_description=None,
            separator=separator
        )
        user_prompt = build_user_prompt(core_prompt, activations, separator)
        
        rows.append({
            'sae_name': sae_name,
            'eval_name': eval_name,
            'feature_index': int(feature_idx),
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'separator': separator,
            'num_per_call': num_per_call
        })
    
    return pd.DataFrame(rows)


def generate_samples_from_examples(examples_df: pd.DataFrame, sae_name: str, output_path: str, 
                                   llm_model: str, max_workers: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate samples - returns llm_queries_df and generated_samples_df.
    
    Args:
        examples_df: DataFrame with feature examples
        sae_name: SAE configuration name
        output_path: Output directory path
        llm_model: OpenAI model to use
        max_workers: Max concurrent API requests
    
    Returns:
        Tuple of (prompts_with_responses, generated_samples_df)
    """
    
    # Generate prompts from the examples DataFrame
    print(f"Generating prompts from examples...")
    prompts_df = generate_prompts_from_examples(
        examples_df, sae_name, output_path,
        num_per_call=10, separator="<SAMPLE_SEPARATOR/>"
    )
    
    print(f"Generated {len(prompts_df)} prompts for {len(prompts_df['feature_index'].unique())} features")
    
    # Call generate_samples with the prompts DataFrame
    samples_df, prompts_with_responses = generate_samples(
        eval_name=output_path,
        sae_name=sae_name,
        output_dir=".",
        model=llm_model,
        max_workers=max_workers,
        prompts_df=prompts_df  # Pass the prompts DataFrame
    )
    
    return prompts_with_responses, samples_df