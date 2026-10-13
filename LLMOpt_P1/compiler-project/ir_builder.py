"""
Build the IR (a flat list of IROp) for one PREFILL forward pass of a
decoder-only transformer: the whole prompt of `sequence_length` tokens is
processed at once for `batch_size` sequences.

Notation (used in every formula below)
--------------------------------------
    B    = batch size
    S    = sequence length
    T    = B * S                 number of tokens processed
    d    = hidden size
    h    = number of query heads
    h_kv = number of key/value heads   (h_kv < h means GQA)
    d_h  = d / h                 head dimension
    d_kv = h_kv * d_h            width of K and V
    f    = FFN intermediate size

FLOP rule for a matrix multiply (M x K) @ (K x N):
    every one of the M*N outputs needs K multiplies + K adds
    => 2 * M * K * N FLOPs

Memory-traffic model (analytical, not a CUDA kernel simulation)
---------------------------------------------------------------
Each op is treated as a separate, UNFUSED kernel that
    * reads its input activations from GPU memory,
    * reads its weights from GPU memory (once, assuming perfect on-chip reuse),
    * writes its output activations back to GPU memory.
All tensors (weights, activations, KV) use the same `precision`.
This "naive unfused" baseline is deliberate: later optimization passes
(e.g. fusing softmax into attention like FlashAttention) can then show
their benefit as a reduction in bytes.
"""

from common import IROp, ModelConfig, Op, bytes_for_elements, normalize_precision

FLOPS_PER_MAC = 2  # one multiply + one add

# Softmax over one score x_i, computed in the numerically stable way
#   y_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
# costs approximately per element:
#   1 (max compare) + 1 (subtract) + 1 (exp) + 1 (sum add) + 1 (divide) = 5
# exp is counted as 1 FLOP (it is really a few instructions; this is an
# approximation). The 1/sqrt(d_h) scaling is assumed folded into W_q.
SOFTMAX_FLOPS_PER_ELEMENT = 5


def matmul_flops(m: int, k: int, n: int) -> int:
    """FLOPs of (m x k) @ (k x n) = 2 * m * k * n."""
    return FLOPS_PER_MAC * m * k * n


# --------------------------------------------------------------------------- #
# One function per op. Each returns (flops, bytes).
# --------------------------------------------------------------------------- #

def qkv_cost(model: ModelConfig, tokens: int, precision: str) -> tuple[int, int]:
    """
    Q = X W_q : (T x d) @ (d x d)     -> 2 * T * d * d
    K = X W_k : (T x d) @ (d x d_kv)  -> 2 * T * d * d_kv
    V = X W_v : (T x d) @ (d x d_kv)  -> 2 * T * d * d_kv
    Total FLOPs = 2 * T * d * (d + 2 * d_kv)
    With GQA d_kv < d, so K and V projections are cheaper.

    Bytes = input X (T*d) + weights d*(d + 2 d_kv) + outputs Q,K,V T*(d + 2 d_kv)
    """
    d, d_kv = model.hidden_size, model.kv_dim
    qkv_width = d + 2 * d_kv

    flops = matmul_flops(tokens, d, qkv_width)
    elements = tokens * d + d * qkv_width + tokens * qkv_width
    return flops, bytes_for_elements(elements, precision)


def attention_cost(model: ModelConfig, batch_size: int, seq_len: int,
                   precision: str) -> tuple[int, int]:
    """
    For every batch element and every QUERY head (B * h of them):
        scores = Q_head @ K_head^T : (S x d_h) @ (d_h x S) -> 2 * S * S * d_h
        out    = P @ V_head        : (S x S)   @ (S x d_h) -> 2 * S * S * d_h
    Total FLOPs = 4 * B * h * S^2 * d_h

    GQA does NOT reduce these FLOPs: each query head still does its own
    matmuls, it just shares K/V with other query heads.
    Causal masking is ignored (dense kernels compute the full S x S matrix).

    Bytes (unfused: scores go to memory, softmax runs, probabilities come back):
        read Q             : T * d
        read K and V       : 2 * T * d_kv          (GQA -> smaller)
        write scores       : B * h * S^2
        read probabilities : B * h * S^2
        write output       : T * d
    """
    d, d_kv, d_h = model.hidden_size, model.kv_dim, model.head_dim
    heads = model.num_attention_heads
    tokens = batch_size * seq_len
    score_elements = batch_size * heads * seq_len * seq_len

    flops_per_head = matmul_flops(seq_len, d_h, seq_len) + matmul_flops(seq_len, seq_len, d_h)
    flops = batch_size * heads * flops_per_head

    elements = (tokens * d             # Q
                + 2 * tokens * d_kv    # K, V
                + 2 * score_elements   # write scores + read probabilities
                + tokens * d)          # output
    return flops, bytes_for_elements(elements, precision)


def softmax_cost(model: ModelConfig, batch_size: int, seq_len: int,
                 precision: str) -> tuple[int, int]:
    """
    Softmax over each row of the B * h attention score matrices (S x S each).
    FLOPs = SOFTMAX_FLOPS_PER_ELEMENT * B * h * S^2
    Bytes = read scores + write probabilities = 2 * B * h * S^2 elements
    """
    score_elements = batch_size * model.num_attention_heads * seq_len * seq_len
    flops = SOFTMAX_FLOPS_PER_ELEMENT * score_elements
    return flops, bytes_for_elements(2 * score_elements, precision)


def out_proj_cost(model: ModelConfig, tokens: int, precision: str) -> tuple[int, int]:
    """
    O = A W_o : (T x d) @ (d x d)  -> FLOPs = 2 * T * d^2
    Bytes = input T*d + weights d*d + output T*d
    """
    d = model.hidden_size
    flops = matmul_flops(tokens, d, d)
    elements = tokens * d + d * d + tokens * d
    return flops, bytes_for_elements(elements, precision)


def mlp_cost(model: ModelConfig, tokens: int, precision: str) -> tuple[int, int]:
    """
    SwiGLU MLP (LLaMA / Mistral):
        gate = X W_gate : (T x d) @ (d x f) -> 2 * T * d * f
        up   = X W_up   : (T x d) @ (d x f) -> 2 * T * d * f
        H    = SiLU(gate) * up              -> O(T*f), ignored (tiny vs matmuls)
        Y    = H W_down : (T x f) @ (f x d) -> 2 * T * f * d
    Total FLOPs = 2 * m * T * d * f, where m = 3 matrices (2 for a plain MLP).

    Bytes = input T*d + weights m*d*f + output T*d
          + hidden H written once and read once (2 * T * f)
            (gate/up/activation assumed fused into one kernel)
    """
    d, f = model.hidden_size, model.intermediate_size
    matrices = model.mlp_matrices

    flops = matrices * matmul_flops(tokens, d, f)
    elements = tokens * d + matrices * d * f + tokens * d + 2 * tokens * f
    return flops, bytes_for_elements(elements, precision)


# --------------------------------------------------------------------------- #
# IR construction
# --------------------------------------------------------------------------- #

def build_layer_ir(model: ModelConfig, layer: int, sequence_length: int,
                   batch_size: int, precision: str) -> list[IROp]:
    """IR for a single transformer layer: QKV -> Attention -> Softmax -> OutProj -> MLP."""
    tokens = batch_size * sequence_length
    costs = [
        (Op.QKV, qkv_cost(model, tokens, precision)),
        (Op.ATTENTION, attention_cost(model, batch_size, sequence_length, precision)),
        (Op.SOFTMAX, softmax_cost(model, batch_size, sequence_length, precision)),
        (Op.OUT_PROJ, out_proj_cost(model, tokens, precision)),
        (Op.MLP, mlp_cost(model, tokens, precision)),
    ]
    return [IROp(layer=layer, op=op, flops=flops, bytes=nbytes)
            for op, (flops, nbytes) in costs]


def build_ir(model: ModelConfig, sequence_length: int, batch_size: int = 1,
             precision: str = "fp16") -> list[IROp]:
    """
    Convert a model configuration into a flat IR list with 5 ops per layer,
    in execution order (layer 0 first).

    Not included (documented limitation): embedding lookup, RMSNorms,
    residual adds, rotary embeddings, and the final LM head.
    """
    if sequence_length <= 0 or batch_size <= 0:
        raise ValueError("sequence_length and batch_size must be positive")
    precision = normalize_precision(precision)

    ir: list[IROp] = []
    for layer in range(model.num_layers):
        ir.extend(build_layer_ir(model, layer, sequence_length, batch_size, precision))
    return ir
