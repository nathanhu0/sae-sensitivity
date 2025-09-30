"""
Core SAE utilities for naming and configuration.
"""

def make_sae_names(model="py", widths=["4k"], types=["btopk"], sparsities=[0,1,2,3,4,5]):
    """Generate SAE names for all combinations of parameters.
    
    SAEBench naming convention: {model}_{width}_{type}_s{sparsity}
    - model: py (pythia-160m-deduped), gem (gemma-2-2b)  
    - width: 4k, 16k, 65k
    - type: btopk, matbtopk, gated, jumprelu, topk, relu, pan
    - sparsity: 0-5
    """
    names = []
    for width in widths:
        for sae_type in types:
            for sparsity in sparsities:
                names.append(f"{model}_{width}_{sae_type}_s{sparsity}")
    return names


def parse_sae_config(sae_name):
    """
    Parse SAE name to extract configuration.

    Example: py_65k_relu_s0 -> {model: pythia-160m-deduped, width: 2pow16, type: relu, trainer: 0}
    """
    parts = sae_name.split('_')

    if len(parts) < 4:
        raise ValueError(f"Invalid SAE name format: {sae_name}")

    # Extract components
    model_prefix = parts[0]  # py or gem
    width = parts[1]         # 4k, 16k, 65k
    sae_type = parts[2]      # relu, topk, etc
    sparsity = parts[3]      # s0, s1, etc

    # Map to SAE config
    model = "pythia-160m-deduped" if model_prefix == "py" else "gemma-2-2b"
    layer = 8 if model_prefix == "py" else 12

    width_map = {"4k": "2pow12", "16k": "2pow14", "65k": "2pow16"}
    type_map = {
        "btopk": "batch_topk",
        "matbtopk": "matryoshka_batch_topk",
        "gated": "gated",
        "jumprelu": "jumprelu",
        "topk": "topk",
        "relu": "relu",
        "pan": "pan"
    }

    trainer = int(sparsity[1:])  # Extract number from s0, s1, etc

    return {
        "model": model,
        "layer": layer,
        "width": width_map[width],
        "sae_type": type_map[sae_type],
        "trainer": trainer
    }