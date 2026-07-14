"""Blood Bowl value/policy network — PyTorch trainer + ONNX export (plan 017).

Tensor layout is defined authoritatively by the Rust encoder
(`botbowl-nn/src/encode.rs`); this package only consumes the prepared
`.npy` batches and never parses a `GameState`.
"""

from .model import BBNet, SPATIAL_CHANNELS, GLOBAL_FEATURES, POLICY_CHANNELS, masked_policy_logits

__all__ = [
    "BBNet",
    "SPATIAL_CHANNELS",
    "GLOBAL_FEATURES",
    "POLICY_CHANNELS",
    "masked_policy_logits",
]
