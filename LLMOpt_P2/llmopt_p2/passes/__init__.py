"""P2 passes. Each pass exposes `name` and `run(ctx) -> PassResult`.
Teammates' passes (fusion, scheduling, auto-tuning) can be appended to ALL_PASSES."""
from passes.kv_cache import KVCachePass
from passes.prefix_share import PrefixSharePass

ALL_PASSES = [KVCachePass(), PrefixSharePass()]
