import os
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def load_prompts(prompts_csv_path: str) -> Dict[Tuple[str, str, int], Tuple[str, str]]:
    """Load prompts from CSV into a mapping: (sae_name, eval_name, feature_index) -> (system, user)."""
    prompts: Dict[Tuple[str, str, int], Tuple[str, str]] = {}
    with open(prompts_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"sae_name", "eval_name", "feature_index", "system_prompt", "user_prompt"}
        missing = [c for c in required_cols if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns in prompts CSV: {missing}")

        for row in reader:
            try:
                sae_name = row["sae_name"].strip()
                eval_name = row["eval_name"].strip()
                feature_index = int(row["feature_index"])  # type: ignore[arg-type]
                system_prompt = row["system_prompt"]
                user_prompt = row["user_prompt"]
                prompts[(sae_name, eval_name, feature_index)] = (system_prompt, user_prompt)
            except Exception as e:
                print(f"Skipping malformed row: {e}")
                continue
    return prompts


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(file_path)) or "."
    os.makedirs(parent, exist_ok=True)


def parse_generated_examples(response_text: str, expected_count: int) -> List[str]:
    """
    Parse LLM response and extract individual examples based on expected count.

    Args:
        response_text: raw LLM response text
        expected_count: expected number of examples (e.g., 10 or 11)

    Returns:
        List of individual example texts
    """
    examples: List[str] = []
    # Try to split on <SAMPLE_SEPARATOR/> if present
    # Otherwise split on double newlines or numbered lists

    # Assuming <SAMPLE_SEPARATOR/> is the standard
    SEPARATOR = "<SAMPLE_SEPARATOR/>"
    if SEPARATOR in response_text:
        raw_splits = response_text.split(SEPARATOR)
        for ex in raw_splits:
            ex_clean = ex.strip()
            if ex_clean:
                examples.append(ex_clean)
    else:
        # Fallback: try double newline splits
        double_newline_splits = response_text.split("\n\n")
        for ex in double_newline_splits:
            ex_clean = ex.strip()
            if ex_clean:
                examples.append(ex_clean)

    # Simple filter: remove overly short examples (likely parsing artifacts)
    # but allow legitimate short examples (e.g., single words/tokens)
    examples = [ex for ex in examples if len(ex) > 0]

    # Check if we got roughly the right number (allow +/- 1)
    # Some LLMs occasionally produce 11 when asked for 10
    if abs(len(examples) - expected_count) <= 1:
        return examples[:expected_count]  # Trim to expected if over

    # If we have significantly fewer/more, return what we have
    # (better than nothing, and we log the discrepancy elsewhere)
    return examples


def process_feature_with_retries(
    client,
    prompts: Dict[Tuple[str, str, int], Tuple[str, str]],
    feature_triple: Tuple[str, str, int],
    model: str,
    num_per_call: int,
    max_attempts: int = 10,
    request_timeout: float = 60.0,
) -> Tuple[bool, Optional[List[str]]]:
    """Process a single feature with built-in retry logic."""
    sae_name, eval_name, feature_index = feature_triple
    if feature_triple not in prompts:
        print(f"Warning: No prompt for {feature_triple}")
        return False, None

    system_prompt, user_prompt = prompts[feature_triple]

    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1.0,
                max_tokens=4096,
                timeout=request_timeout,
            )
            content = response.choices[0].message.content
            examples = parse_generated_examples(content, num_per_call)
            if examples:
                return True, examples
            # If no examples parsed, retry
        except Exception as e:
            # Retry on any exception
            if attempt == max_attempts - 1:
                # Last attempt, log the error
                print(f"Failed {feature_triple} after {max_attempts} attempts: {e}")
        
        # Small backoff between retries
        if attempt < max_attempts - 1:
            time.sleep(min(2 ** attempt, 10))  # Exponential backoff, max 10 seconds
    
    return False, None


def run_scheduler(
    client,
    features: List[Tuple[str, str, int]],
    prompts: Dict[Tuple[str, str, int], Tuple[str, str]],
    model: str,
    num_per_call: int,
    max_workers: int,
    max_attempts: int = 10,
    request_timeout: float = 60.0,
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]], Dict[Tuple[str, str, int], List[str]]]:
    """
    Process all features with parallel workers. Retry logic is built into each task.

    Returns:
        (succeeded_features, failed_features, results_dict)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    succeeded: List[Tuple[str, str, int]] = []
    failed: List[Tuple[str, str, int]] = []
    results: Dict[Tuple[str, str, int], List[str]] = {}
    
    total = len(features)
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all features
        future_to_feature = {
            executor.submit(
                process_feature_with_retries,
                client,
                prompts,
                feature,
                model,
                num_per_call,
                max_attempts,
                request_timeout,
            ): feature
            for feature in features
        }
        
        # Process as they complete
        completed = 0
        for future in as_completed(future_to_feature):
            feature = future_to_feature[future]
            completed += 1
            
            try:
                ok, result_examples = future.result()
                if ok and result_examples:
                    succeeded.append(feature)
                    results[feature] = result_examples
                else:
                    failed.append(feature)
            except Exception as e:
                print(f"Unexpected error for {feature}: {e}")
                failed.append(feature)
            
            # Progress logging every 10 completions or at the end
            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - start_time
                print(f"[{elapsed:.1f}s] Completed {completed}/{total} features ({len(succeeded)} successful)", flush=True)
    
    return succeeded, failed, results


def generate_samples(eval_name, sae_name, output_dir="evals", model="gpt-4.1-mini", 
                    max_workers=50, max_attempts=10, request_timeout=60.0, prompts_df=None):
    """Generate samples using OpenAI API.
    
    Args:
        eval_name: Name of the evaluation run
        sae_name: SAE configuration name
        output_dir: Base output directory
        model: OpenAI model to use
        max_workers: Max concurrent API requests
        max_attempts: Max retry attempts per feature
        request_timeout: Timeout per API request in seconds
        prompts_df: Optional DataFrame with prompts. If None, loads from file.
    
    Returns:
        tuple: (samples_df, prompts_with_responses)
    """
    import os
    import time
    
    sae_dir = os.path.join(output_dir, eval_name, sae_name)
    prompts_path = os.path.join(sae_dir, "generation_queries_with_responses.csv")
    samples_path = os.path.join(sae_dir, "generated_samples.csv")
    
    print(f"Generating samples for {sae_name} using {model}...")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Load prompts from DataFrame or file
    if prompts_df is not None:
        # Convert DataFrame to the expected format
        prompts = {}
        for _, row in prompts_df.iterrows():
            key = (row['sae_name'], row['eval_name'], int(row['feature_index']))
            prompts[key] = (row['system_prompt'], row['user_prompt'])
        features = list(prompts.keys())
        print(f"Using {len(features)} features from provided DataFrame")
    else:
        # Load from file (backward compatibility)
        prompts = load_prompts(prompts_path)
        features = list(prompts.keys())
        print(f"Loaded {len(features)} features from {prompts_path}")
    
    # Run generation
    start_time = time.time()
    succeeded, failed, results = run_scheduler(
        client=client,
        features=features,
        prompts=prompts,
        model=model,
        num_per_call=10,
        max_workers=max_workers,
        max_attempts=max_attempts,
        request_timeout=request_timeout
    )
    elapsed = time.time() - start_time
    
    print(f"Completed in {elapsed:.1f}s - Succeeded: {len(succeeded)} | Failed: {len(failed)}")
    
    # Create two DataFrames:
    # 1. prompts_with_responses: includes full LLM responses
    # 2. samples_df: split individual samples
    
    import pandas as pd
    
    # Update prompts DataFrame with responses
    if prompts_df is None:
        # Load from file if not provided
        prompts_df = pd.read_csv(prompts_path)
    
    response_map = {}
    for (sae_name, eval_name, feature_index), examples in results.items():
        if (sae_name, eval_name, feature_index) in succeeded:
            # Join examples back with separator for full response
            separator = prompts_df['separator'].iloc[0] if 'separator' in prompts_df.columns else '\n'
            full_response = separator.join(examples) if len(examples) > 1 else examples[0] if examples else ""
            response_map[feature_index] = full_response
    
    prompts_with_responses = prompts_df.copy()
    prompts_with_responses['llm_response'] = prompts_with_responses['feature_index'].map(response_map)
    prompts_with_responses['status'] = prompts_with_responses['feature_index'].apply(
        lambda x: f"Success with {model}" if x in response_map else "Failed"
    )
    
    # Create samples DataFrame with split samples
    rows = []
    if succeeded:
        for (sae_name, eval_name, feature_index), examples in sorted(results.items()):
            if (sae_name, eval_name, feature_index) in succeeded:
                for i, text in enumerate(examples):
                    rows.append({
                        "feature_index": feature_index,
                        "sample_index": i,
                        "text": text,
                    })
    
    samples_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    
    # Save to files
    if not samples_df.empty:
        samples_df.to_csv(samples_path, index=False)
        print(f"Wrote {len(samples_df)} samples to {samples_path}")
    
    # Save prompts with responses
    prompts_with_responses.to_csv(prompts_path, index=False)
    
    return samples_df, prompts_with_responses