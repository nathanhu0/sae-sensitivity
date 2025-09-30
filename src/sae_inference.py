import json
import os
from typing import Optional, Literal
from dotenv import load_dotenv

load_dotenv()

import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download, login
from transformer_lens import HookedTransformer
import pandas as pd
from sae_utils import parse_sae_config

# SAE bench imports
import sae_bench.custom_saes.batch_topk_sae as batch_topk_sae
import sae_bench.custom_saes.gated_sae as gated_sae
import sae_bench.custom_saes.jumprelu_sae as jumprelu_sae
import sae_bench.custom_saes.relu_sae as relu_sae
import sae_bench.custom_saes.topk_sae as topk_sae

torch.set_grad_enabled(False);

# Repository mapping
REPO_MAPPING = {
    ("pythia-160m-deduped", "2pow12"): "adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108",
    ("pythia-160m-deduped", "2pow14"): "adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108",
    ("pythia-160m-deduped", "2pow16"): "adamkarvonen/saebench_pythia-160m-deduped_width-2pow16_date-0108",
    ("gemma-2-2b", "2pow12"): "adamkarvonen/saebench_gemma-2-2b_width-2pow12_date-0108",
    ("gemma-2-2b", "2pow14"): "canrager/saebench_gemma-2-2b_width-2pow14_date-0107",
    ("gemma-2-2b", "2pow16"): "canrager/saebench_gemma-2-2b_width-2pow16_date-0107",
}

# Loader mapping
LOADERS = {
    "matryoshka_batch_topk": batch_topk_sae.load_dictionary_learning_matryoshka_batch_topk_sae,
    "batch_topk": batch_topk_sae.load_dictionary_learning_batch_topk_sae,
    "topk": topk_sae.load_dictionary_learning_topk_sae,
    "relu": relu_sae.load_dictionary_learning_relu_sae,
    "jumprelu": jumprelu_sae.load_dictionary_learning_jump_relu_sae,
    "gated": gated_sae.load_dictionary_learning_gated_sae,
    "pan": relu_sae.load_dictionary_learning_relu_sae,
}


def init_model(model_name="pythia-160m-deduped"):
    """Initialize the transformer model for SAE evaluation."""
    # Disable tokenizer parallelism to avoid warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Disable gradients for inference
    torch.set_grad_enabled(False)

    # Login to HuggingFace (token sourced from environment)
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        login(token=hf_token)

    # Auto-detect device
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model with appropriate dtype
    if model_name == "pythia-160m-deduped":
        # Pythia uses float32
        model = HookedTransformer.from_pretrained("EleutherAI/pythia-160m-deduped", 
                                                 device=device,
                                                 dtype=torch.float32)
    elif model_name == "gemma-2-2b":
        # Gemma uses bfloat16 for efficiency
        model = HookedTransformer.from_pretrained("google/gemma-2-2b", 
                                                 device=device,
                                                 dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def load_saebench_sae(
    model: Literal["pythia-160m-deduped", "gemma-2-2b"],
    layer: int,
    width: Literal["2pow12", "2pow14", "2pow16"],
    sae_type: Literal["matryoshka_batch_topk", "batch_topk", "topk", "relu", "jumprelu", "gated", "pan"],
    trainer: int,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    local_dir: str = "downloaded_saes",
):
    """
    Load a SAEBench SAE with minimal interface.

    Args:
        model: "pythia-160m-deduped" or "gemma-2-2b"
        layer: Layer number
        width: "2pow12", "2pow14", or "2pow16"
        sae_type: "matryoshka_batch_topk", "batch_topk", "topk", "relu", "jumprelu", "gated", "pan"
        trainer: Trainer number
        device: Device ("cuda"/"cpu", auto-detects if None)
        dtype: Data type (auto-detects if None)
        local_dir: Local directory for downloads (default: "downloaded_saes")
    """

    # Get repo and loader
    repo_id = REPO_MAPPING[(model, width)]
    loader_func = LOADERS[sae_type]

    # Auto-detect defaults
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")
    if dtype is None:
        dtype = torch.bfloat16 if "gemma" in model else torch.float32

    # Construct filename based on repo owner naming conventions
    if "adamkarvonen" in repo_id:
        # Adam's naming: "MatryoshkaBatchTopK_model__date" (works for both pythia and gemma)
        sae_type_to_folder = {
            "matryoshka_batch_topk": "MatryoshkaBatchTopK",
            "batch_topk": "BatchTopK",
            "topk": "TopK",
            "relu": "Standard",
            "jumprelu": "JumpRelu",
            "gated": "GatedSAE",
            "pan": "PAnneal"
        }
        # Extract date from repo_id
        date_suffix = repo_id.split("date-")[1] if "date-" in repo_id else "0108"
        folder_name = f"{sae_type_to_folder[sae_type]}_{model}__{date_suffix}"
    else:  # canrager repos
        # Can's naming: "gemma-2-2b_matryoshka_batch_top_k_width-2pow16_date-0107"
        sae_type_to_folder = {
            "matryoshka_batch_topk": "matryoshka_batch_top_k",
            "batch_topk": "batch_top_k",
            "topk": "top_k",
            "relu": "standard_new",
            "jumprelu": "jump_relu",
            "gated": "gated",
            "pan": "p_anneal"
        }
        # Extract date from repo_id
        date_suffix = repo_id.split("date-")[1] if "date-" in repo_id else "0107"
        folder_name = f"gemma-2-2b_{sae_type_to_folder[sae_type]}_width-{width}_date-{date_suffix}"

    filename = f"{folder_name}/resid_post_layer_{layer}/trainer_{trainer}/ae.pt"

    print(f"Downloading SAE to: {local_dir}/")

    # Load SAE
    sae = loader_func(
        repo_id=repo_id,
        filename=filename,
        model_name=model,
        device=torch.device(device),
        dtype=dtype,
        layer=layer,
        local_dir=local_dir,
    )

    # Load and print eval info for sanity checking
    try:
        eval_filename = filename.replace("/ae.pt", "/eval_results.json")
        eval_path = hf_hub_download(
            repo_id=repo_id,
            filename=eval_filename,
            local_dir=local_dir
        )
        with open(eval_path) as f:
            eval_info = json.load(f)

        print(f"✓ Loaded SAE: {sae_type} (trainer {trainer})")
        print(f"  L0 sparsity: {eval_info.get('l0', 'N/A'):.1f}")
        print(f"  Reconstruction loss: {eval_info.get('frac_variance_explained', 'N/A'):.4f}")

    except Exception as e:
        print(f"✓ Loaded SAE: {sae_type} (trainer {trainer})")
        print(f"  Warning: Could not load eval info")

    return sae


def load_gemmascope_sae(
    layer: int,
    width: str,  # e.g., "16k", "128k", "1m"
    l0: int,  # target L0 value
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    local_dir: str = "downloaded_saes",
):
    """
    Load a GemmaScope SAE.
    
    Args:
        layer: Layer number (typically 12 for gemma-2-2b)
        width: Width string like "16k", "128k", "1m"
        l0: Target L0 sparsity value
        device: Device ("cuda"/"cpu", auto-detects if None)
        dtype: Data type (auto-detects if None)
        local_dir: Local directory for downloads
    """
    
    # Auto-detect defaults
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")
    
    if dtype is None:
        dtype = torch.bfloat16  # GemmaScope typically uses bfloat16
    
    # Construct filename
    filename = f"layer_{layer}/width_{width}/average_l0_{l0}/params.npz"
    
    print(f"Loading GemmaScope SAE (layer {layer}, width {width}, L0={l0})...")
    
    # Load using jumprelu loader
    sae = jumprelu_sae.load_gemma_scope_jumprelu_sae(
        repo_id="google/gemma-scope-2b-pt-res",
        filename=filename,
        layer=layer,
        model_name="gemma-2-2b",
        device=torch.device(device),
        dtype=dtype,
        local_dir=local_dir,
    )
    
    print(f"✓ Loaded GemmaScope SAE: width={width}, L0={l0}")
    
    return sae


def load_sae(
    sae_name: str,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    local_dir: str = "downloaded_saes",
):
    """
    Load SAE from name string, auto-detecting type.
    
    SAEBench format: {model}_{width}_{type}_s{sparsity}
        Examples: py_4k_btopk_s0, gem_16k_relu_s2
        
    GemmaScope format: gscope_{width}_l0_{value} or gscope_layer{N}_{width}_l0_{value}
        Examples: gscope_16k_l0_128, gscope_layer12_1m_l0_256
    
    Args:
        sae_name: SAE identifier string
        device: Device ("cuda"/"cpu", auto-detects if None)
        dtype: Data type (auto-detects if None)
        local_dir: Local directory for downloads
    """
    
    if sae_name.startswith("gscope_"):
        # Parse GemmaScope format
        parts = sae_name.split("_")
        
        if len(parts) < 4:
            raise ValueError(f"Invalid GemmaScope SAE name: {sae_name}")
        
        if parts[1].startswith("layer"):
            # Format: gscope_layer12_16k_l0_128
            layer = int(parts[1][5:])
            width = parts[2]
            if parts[3] != "l0":
                raise ValueError(f"Expected 'l0' in position 3, got: {parts[3]}")
            l0 = int(parts[4])
        else:
            # Format: gscope_16k_l0_128 (default layer 12)
            layer = 12
            width = parts[1]
            if parts[2] != "l0":
                raise ValueError(f"Expected 'l0' in position 2, got: {parts[2]}")
            l0 = int(parts[3])
        
        return load_gemmascope_sae(layer, width, l0, device, dtype, local_dir)
    
    else:
        # Parse SAEBench format (existing logic)
        config = parse_sae_config(sae_name)
        return load_saebench_sae(
            model=config["model"],
            layer=config["layer"],
            width=config["width"],
            sae_type=config["sae_type"],
            trainer=config["trainer"],
            device=device,
            dtype=dtype,
            local_dir=local_dir,
        )


def run_sae(text, model, sae, max_length=128, max_batch_size=4, zero_bos_padding=True):
    """Analyze text using the SAE and return activations and features.

    Args:
        text (str or list): The input text(s) to analyze. Can be a single string
            or a list of strings.
        model: The HookedTransformer model
        sae: the (saebench) SAE. Expected to have `encode()` method and `cfg.hook_name` attribute.
        max_length (int, optional): Maximum number of tokens to process. If the
            tokenized text exceeds this length, it will be truncated. Defaults to 128.
        max_batch_size (int, optional): Maximum number of texts to process at once.
            If more texts are provided, they will be processed in batches. Defaults to 4.
        zero_bos_padding (bool, optional): Zero out activations/features at BOS and padding tokens. Defaults to True.

    Returns:
        dict: A dictionary containing:
            - 'tokens': torch.Tensor of shape (batch_size, sequence_length)
                containing the tokenized input.
            - 'activations': torch.Tensor containing the model's internal activations
                at the SAE's hook point.
            - 'features': torch.Tensor containing the SAE-encoded features extracted
                from the activations.
    """
    model_device = next(model.parameters()).device
    # Handle single string input
    if isinstance(text, str):
        text = [text]

    # Tokenize all texts at once
    tokens = model.to_tokens(text, move_to_device=False)
    if tokens.shape[1] > max_length:
        tokens = tokens[:, :max_length]

    # Process in batches
    all_activations = []
    all_features = []

    for i in range(0, len(text), max_batch_size):
        batch_tokens = tokens[i:i + max_batch_size].to(model_device)

        # Run model and get activations
        with torch.no_grad():
            _, cache = model.run_with_cache(batch_tokens, names_filter=[sae.cfg.hook_name])
            activations = cache[sae.cfg.hook_name]

            # Cast activations to the SAE's dtype and ensure correct device
            activations = activations.to(sae.W_enc.dtype).to(sae.W_enc.device)

            features = sae.encode(activations)

            # Zero out BOS and padding tokens if requested
            if zero_bos_padding:
                mask = (batch_tokens == model.tokenizer.bos_token_id) | (batch_tokens == model.tokenizer.pad_token_id)
                mask = mask.to(activations.device)
                activations = activations * (~mask).unsqueeze(-1)
                features = features * (~mask).unsqueeze(-1)

        all_activations.append(activations.cpu())
        all_features.append(features.cpu())

    # Concatenate results (tokens don't need concatenation)
    return {
        'tokens': tokens,
        'activations': torch.cat(all_activations, dim=0),
        'features': torch.cat(all_features, dim=0),
    }




# parse_sae_config moved to sae_utils.py

 
# def load_samples_from_csv(file_path):
#     """
#     Load samples from CSV file.

#     Args:
#         file_path: Path to the CSV file (all_samples.csv)

#     Returns:
#         pandas.DataFrame: DataFrame containing the sample data
#     """
#     try:
#         # Load CSV with explicit data types to ensure feature_index is treated as int
#         df = pd.read_csv(file_path, dtype={'feature_index': 'int64'})
#         print(f"Loaded {len(df)} samples from {file_path}")
#         print(f"Feature index range: {df['feature_index'].min()} to {df['feature_index'].max()}")
#         print(f"Feature index dtype: {df['feature_index'].dtype}")

#         return df
#     except Exception as e:
#         print(f"Error loading CSV file {file_path}: {e}")
#         return None


def run_sae_inference_on_samples(samples_df: pd.DataFrame, model, sae, sae_name: str, output_path: str, 
                                 output_file: str, batch_size: int = 4, max_length: int = 128) -> pd.DataFrame:
    """Run SAE inference and return DataFrame with activations.
    
    Args:
        samples_df: DataFrame with samples to run inference on
        model: The loaded transformer model
        sae: The loaded SAE
        sae_name: SAE configuration name
        output_path: Output directory path
        output_file: Name of output CSV file
        batch_size: Batch size for processing
        max_length: Maximum token length
    
    Returns:
        DataFrame with added columns: text_clean, tokenized_text, max_activation, all_activations
    """
    # Standardize format
    samples_for_inference = samples_df.copy()
    if 'Sequence' in samples_for_inference.columns:
        samples_for_inference = samples_for_inference.rename(columns={'Sequence': 'text'})
    
    # Compute activations with batch processing
    samples_with_activations = compute_sae_activations(
        samples_for_inference, sae, model, batch_size=batch_size, max_length=max_length
    )
    
    # Save results
    sae_dir = os.path.join(".", output_path, sae_name)
    output_path_full = os.path.join(sae_dir, output_file)
    samples_with_activations.to_csv(output_path_full, index=False)
    
    return samples_with_activations


def compute_sae_activations(df, sae, model, batch_size=4, max_length=128):
    """
    Compute SAE feature activations for texts in a DataFrame.
    
    Args:
        df: DataFrame with columns 'feature_index' and 'text' 
        sae: The SAE model
        model: The transformer model
        batch_size: Batch size for processing
        max_length: Maximum token length
        
    Returns:
        DataFrame with added columns: text_clean, tokenized_text, 
        max_activation, all_activations (with BOS token zeroed out)
    """
    results = []
    texts = df['text'].tolist()
    feature_indices = df['feature_index'].tolist()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing activations"):
        batch_texts = texts[i:i + batch_size]
        batch_features = feature_indices[i:i + batch_size]
        
        # Clean texts (remove {{}} markers)
        batch_texts_clean = [text.replace('{{', '').replace('}}', '') for text in batch_texts]
        
        # Run SAE on clean texts with BOS tokens zeroed out (zero_bos_padding=True is default)
        sae_results = run_sae(batch_texts_clean, model, sae, max_length, batch_size, zero_bos_padding=True)
        
        # Process each item in batch
        for j in range(len(batch_texts)):
            feature_idx = int(batch_features[j])
            
            # Get original row data
            row_idx = i + j
            result_row = df.iloc[row_idx].to_dict()
            
            # Add text versions
            result_row['text_annotated'] = batch_texts[j]  # Original with {{}}
            result_row['text_clean'] = batch_texts_clean[j]  # Without {{}}
            
            # Add tokenized text
            result_row['tokenized_text'] = sae_results['tokens'][j].cpu().numpy().tolist()
            
            # Extract feature activations for this specific feature across the sequence
            # Note: BOS token activation is already zeroed out by run_sae with zero_bos_padding=True
            feature_activations = sae_results['features'][j, :, feature_idx].cpu().float().numpy()
            
            # Store activation statistics (BOS already zeroed, so max is meaningful)
            result_row['max_activation'] = float(feature_activations.max())
            result_row['all_activations'] = feature_activations.tolist()
            
            # Remove original 'text' column to avoid confusion
            if 'text' in result_row:
                del result_row['text']
            
            results.append(result_row)
    
    return pd.DataFrame(results)

# make_sae_names moved to sae_utils.py








