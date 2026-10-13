"""P2 pass 2 - prefix sharing.

A compressed (radix) trie over the prompts' token ids finds the prefixes several requests
have in common (system prompt, few-shot block, ...). Shared prefix KV is computed and stored
once, in `chunk_size`-token chunks (ChunkAttention-style), instead of once per request.

The candidate reports the memory saved (GiB of KV) and the prefill compute saved (FLOPs,
prefill tokens)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from cost_model import prefill_flops
from ir import GIB, Candidate, Context, PassResult, Rejection, resolve_prompts


class _Node:
    __slots__ = ("label", "count", "children")

    def __init__(self, label: Tuple[int, ...], count: int):
        self.label = label          # tokens on the edge leading into this node
        self.count = count          # number of prompts that pass through / end on this edge
        self.children: Dict[int, "_Node"] = {}


class RadixTrie:
    """Compressed trie keyed by token ids; edges are split lazily on divergence."""

    def __init__(self) -> None:
        self.root = _Node((), 0)
        self.num_seqs = 0

    def insert(self, seq: Sequence[int]) -> int:
        """Insert seq; return the length of its longest prefix already in the trie."""
        node, i, n, matched = self.root, 0, len(seq), 0
        while i < n:
            child = node.children.get(seq[i])
            if child is None:
                node.children[seq[i]] = _Node(tuple(seq[i:]), 1)
                break
            lab = child.label
            j, m = 0, min(len(lab), n - i)
            while j < m and lab[j] == seq[i + j]:
                j += 1
            matched += j
            if j < len(lab):                       # diverged inside the edge: split it
                mid = _Node(lab[:j], child.count + 1)
                child.label = lab[j:]
                mid.children[child.label[0]] = child
                node.children[seq[i]] = mid
                if i + j < n:
                    mid.children[seq[i + j]] = _Node(tuple(seq[i + j:]), 1)
                break
            child.count += 1
            node, i = child, i + j
        self.num_seqs += 1
        return matched

    def _walk(self):
        """Yield (node, start_depth, head_tokens) for every non-root node."""
        stack = [(c, 0, ()) for c in self.root.children.values()]
        while stack:
            node, start, head = stack.pop()
            head = (head + node.label)[:8]
            yield node, start, head
            end = start + len(node.label)
            stack.extend((c, end, head) for c in node.children.values())

    def stored_tokens(self) -> int:
        """Tokens physically stored once the shared prefixes are deduplicated."""
        return sum(len(n.label) for n, _, _ in self._walk())

    def shared_segments(self) -> List[dict]:
        """Trie segments used by >= 2 prompts: [start, start+length) shared by `count` prompts."""
        return [dict(start=s, length=len(n.label), count=n.count, head=h)
                for n, s, h in self._walk() if n.count >= 2]

    def common_prefix_all(self) -> int:
        """Length of the prefix shared by ALL inserted prompts."""
        node, depth = self.root, 0
        while len(node.children) == 1:
            (child,) = node.children.values()
            if child.count != self.num_seqs:
                break
            depth += len(child.label)
            node = child
        return depth


def analyze_prompts(seqs: Sequence[Sequence[int]], chunk_size: int = 16, topk: int = 5) -> dict:
    """Trie statistics + sequential-cache reuse (prompt i reuses what prompts 0..i-1 cached)."""
    trie = RadixTrie()
    reuse: List[int] = []
    for s in seqs:
        matched = trie.insert(s)
        reuse.append(min((matched // chunk_size) * chunk_size, len(s)))
    total = sum(len(s) for s in seqs)
    segs = trie.shared_segments()
    best = max(segs, key=lambda d: d["start"] + d["length"], default=None)
    top = sorted(segs, key=lambda d: d["length"] * (d["count"] - 1), reverse=True)[:topk]
    return dict(
        num_prompts=len(seqs), total_tokens=total, stored_tokens=trie.stored_tokens(),
        shareable_tokens_exact=total - trie.stored_tokens(),
        reuse_per_prompt=reuse, reuse_tokens=sum(reuse),
        longest_shared_prefix=(best["start"] + best["length"]) if best else 0,
        longest_shared_prefix_prompts=best["count"] if best else 0,
        longest_shared_prefix_head=best["head"] if best else (),
        common_prefix_all=trie.common_prefix_all(), top_segments=top)


class PrefixSharePass:
    name = "prefix_share"

    def run(self, ctx: Context) -> PassResult:
        res = PassResult(self.name)
        m, w, o = ctx.model, ctx.workload, ctx.opts
        data = resolve_prompts(w)
        if data is None:
            res.rejections.append(Rejection("prefix_share", "no prompts given (add workload.prompts or workload.prompt_spec)"))
            return res
        seqs = data["sequences"]
        if len(seqs) < 2:
            res.rejections.append(Rejection("prefix_share", "need at least 2 prompts to share a prefix"))
            return res

        st = analyze_prompts(seqs, o.chunk_size)
        res.info.update(prompt_source=data["source"], **{k: st[k] for k in (
            "num_prompts", "total_tokens", "stored_tokens", "longest_shared_prefix",
            "longest_shared_prefix_prompts", "common_prefix_all")})
        if st["reuse_tokens"] == 0:
            res.rejections.append(Rejection(
                "prefix_share", f"no prefix of >= {o.chunk_size} tokens is shared by two prompts "
                                f"(longest shared prefix: {st['longest_shared_prefix']} tokens)"))
            return res

        # Savings, exact for the given prompts, expressed as ratios so they can be applied to
        # the workload's B x P prompt tokens when the sample prompts differ in number/length.
        r = st["reuse_tokens"] / st["total_tokens"]
        flops_saved = sum(prefill_flops(m, n) for n in st["reuse_per_prompt"]) - 0.0
        # prefix tokens are positions 0..n-1 of their prompt, so prefill_flops(m, n) is exact
        flops_total = sum(prefill_flops(m, len(s)) for s in seqs)
        f = flops_saved / flops_total
        shared_tokens = r * w.batch_size * w.prompt_length
        mem_saved = shared_tokens * m.kv_bytes_per_token()
        wl_flops = w.batch_size * prefill_flops(m, w.prompt_length)

        vrev: Optional[dict] = data["vocab_rev"]
        head = st["longest_shared_prefix_head"]
        preview = (" ".join(vrev.get(t, "?") for t in head) if vrev else " ".join(map(str, head))) + \
                  (" ..." if st["longest_shared_prefix"] > len(head) else "")
        res.candidates.append(Candidate(
            name="prefix_share", pass_name=self.name,
            description=f"shared-prefix KV cache ({o.chunk_size}-token chunks, radix trie)",
            memory_saved_bytes=mem_saved, latency_delta_pct=0.0, accuracy_risk="none",
            prefill_compute_saved_flops=f * wl_flops,
            rationale=(f"longest shared prefix = {st['longest_shared_prefix']} tokens "
                       f"(shared by {st['longest_shared_prefix_prompts']} of {st['num_prompts']} prompts); "
                       f"{r * 100:.1f}% of prompt tokens never need recomputing or re-storing"),
            details=dict(
                prompt_source=data["source"], extrapolated=not data["synthetic"],
                num_prompts=st["num_prompts"], total_prompt_tokens=st["total_tokens"],
                trie_stored_tokens=st["stored_tokens"], exact_shareable_tokens=st["shareable_tokens_exact"],
                reuse_tokens_chunk_aligned=st["reuse_tokens"], chunk_size=o.chunk_size,
                longest_shared_prefix=st["longest_shared_prefix"],
                longest_shared_prefix_prompts=st["longest_shared_prefix_prompts"],
                longest_shared_prefix_preview=preview, common_prefix_all=st["common_prefix_all"],
                top_segments=st["top_segments"],
                token_reuse_ratio=r, flops_saved_fraction=f,
                prefill_tokens_saved=shared_tokens, memory_saved_gib=mem_saved / GIB,
                compute_saved_tflop=f * wl_flops / 1e12)))
        return res
