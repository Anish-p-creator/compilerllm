"""
LLM model presets and parameter counting.

Values are taken from the official Hugging Face `config.json` files:
  * LLaMA-7B   : meta-llama/Llama-2-7b-hf (same shape as original LLaMA-7B)
  * Mistral-7B : mistralai/Mistral-7B-v0.1

To add a model, add one more ModelConfig entry to MODEL_PRESETS.
"""

from common import ModelConfig

MODEL_PRESETS: dict[str, ModelConfig] = {
    "llama-7b": ModelConfig(
        name="LLaMA-7B",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=32,           # standard multi-head attention
        intermediate_size=11008,
        vocab_size=32000,
        reported_params=6_740_000_000,
    ),
    "mistral-7b": ModelConfig(
        name="Mistral-7B",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,            # Grouped Query Attention: 4 query heads share 1 KV head
        intermediate_size=14336,
        vocab_size=32000,
        reported_params=7_240_000_000,
    ),
}


def get_model(name: str) -> ModelConfig:
    """Look up a model preset by (case-insensitive) key, e.g. 'llama-7b'."""
    key = name.strip().lower().replace("_", "-")
    if key not in MODEL_PRESETS:
        valid = ", ".join(MODEL_PRESETS)
        raise ValueError(f"Unknown model '{name}'. Valid options: {valid}")
    return MODEL_PRESETS[key]


def attention_params_per_layer(model: ModelConfig) -> int:
    """
    W_q: d x d,  W_k: d x d_kv,  W_v: d x d_kv,  W_o: d x d
    => d * (2d + 2 d_kv)       (no biases in LLaMA / Mistral)
    """
    d, d_kv = model.hidden_size, model.kv_dim
    return d * (2 * d + 2 * d_kv)


def mlp_params_per_layer(model: ModelConfig) -> int:
    """SwiGLU: gate, up (d x f) and down (f x d) => 3 * d * f."""
    return model.mlp_matrices * model.hidden_size * model.intermediate_size


def count_parameters(model: ModelConfig) -> int:
    """
    Parameter count computed from the architecture:

        P = V*d                         (token embedding)
          + L * (attention + MLP + 2d)  (2 RMSNorm weight vectors per layer)
          + d                           (final RMSNorm)
          + V*d                         (LM head, unless tied to embedding)
    """
    d, vocab = model.hidden_size, model.vocab_size
    norms_per_layer = 2 * d
    per_layer = attention_params_per_layer(model) + mlp_params_per_layer(model) + norms_per_layer

    embedding = vocab * d
    lm_head = 0 if model.tied_embeddings else vocab * d
    final_norm = d
    return embedding + model.num_layers * per_layer + final_norm + lm_head
