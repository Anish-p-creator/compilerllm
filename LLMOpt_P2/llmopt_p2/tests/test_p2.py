import json, os, subprocess, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cost_model import evaluate, int8_ratio, kv_pressure, kv_reserved_bytes, paged_tokens
from ir import GIB, load_context, load_context_file
from passes.kv_cache import KVCachePass, simulate_int8_error
from passes.prefix_share import PrefixSharePass, RadixTrie, analyze_prompts
from planner import build_plan, run_passes


def sample(name):
    return load_context_file(os.path.join(ROOT, "samples", name))


class TestTrie(unittest.TestCase):
    def test_longest_prefix_and_counts(self):
        t = RadixTrie()
        self.assertEqual(t.insert([1, 2, 3, 4]), 0)
        self.assertEqual(t.insert([1, 2, 3, 9, 9]), 3)
        self.assertEqual(t.insert([1, 2, 7]), 2)
        self.assertEqual(t.insert([5]), 0)
        self.assertEqual(t.insert([1, 2, 3, 4]), 4)          # duplicate prompt
        self.assertEqual(t.stored_tokens(), 8)   # 1,2,3,4 | 9,9 | 7 | 5

    def test_stored_tokens_exact(self):
        seqs = [[1, 2, 3, 4], [1, 2, 3, 9, 9], [1, 2, 7], [5]]
        t = RadixTrie()
        for s in seqs:
            t.insert(s)
        # unique trie tokens: 1,2,3,4 | 9,9 | 7 | 5  => 4 + 2 + 1 + 1
        self.assertEqual(t.stored_tokens(), 8)
        self.assertEqual(t.common_prefix_all(), 0)

    def test_common_prefix_all(self):
        t = RadixTrie()
        for s in ([1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 6, 7]):
            t.insert(s)
        self.assertEqual(t.common_prefix_all(), 3)

    def test_savings_identity(self):
        seqs = [[1, 2, 3, 4, 5], [1, 2, 3, 9], [1, 2, 8], [1, 2, 3, 4, 5, 6]]
        st = analyze_prompts(seqs, chunk_size=1)
        self.assertEqual(st["reuse_tokens"], st["shareable_tokens_exact"])   # chunk=1 -> token level
        self.assertEqual(st["total_tokens"] - st["stored_tokens"], st["shareable_tokens_exact"])

    def test_chunk_alignment(self):
        st = analyze_prompts([list(range(40)), list(range(40)) ], chunk_size=16)
        self.assertEqual(st["reuse_per_prompt"], [0, 32])


class TestKV(unittest.TestCase):
    def test_high_pressure_proposes_both(self):
        r = KVCachePass().run(sample("high_pressure.json"))
        self.assertEqual({c.name for c in r.candidates}, {"paged_kv", "int8_kv"})
        self.assertGreater(r.info["memory_pressure"], 0.6)

    def test_low_pressure_rejects_both(self):
        r = KVCachePass().run(sample("low_pressure.json"))
        self.assertEqual(r.candidates, [])
        self.assertEqual({x.name for x in r.rejections}, {"paged_kv", "int8_kv"})

    def test_paged_numbers(self):
        ctx = sample("high_pressure.json")
        c = {c.name: c for c in KVCachePass().run(ctx).candidates}["paged_kv"]
        self.assertAlmostEqual(c.memory_saved_bytes / GIB, 24.0, places=3)   # (4096-2560)*32*512KiB
        self.assertEqual(c.latency_delta_pct, 3.0)

    def test_int8_halves_memory_and_is_faster(self):
        ctx = sample("high_pressure.json")
        c = {c.name: c for c in KVCachePass().run(ctx).candidates}["int8_kv"]
        self.assertAlmostEqual(int8_ratio(ctx.model), 0.5078125)
        self.assertAlmostEqual(c.memory_saved_bytes / kv_reserved_bytes(ctx), 1 - 0.5078125, places=6)
        self.assertLess(c.latency_delta_pct, -40)
        self.assertIn(c.accuracy_risk, ("low", "medium"))

    def test_int8_error_real_attention(self):
        a = simulate_int8_error(128)
        self.assertLess(a["rel_error"], 0.03)
        self.assertGreater(a["mean_cosine"], 0.999)

    def test_threshold_boundary(self):
        ctx = sample("high_pressure.json")
        ctx.opts.pressure_threshold = kv_pressure(ctx)          # not strictly greater -> rejected
        self.assertEqual(KVCachePass().run(ctx).candidates, [])


class TestPrefixPass(unittest.TestCase):
    def test_synthetic_exact(self):
        r = PrefixSharePass().run(sample("high_pressure.json"))
        d = r.candidates[0].details
        self.assertEqual(d["reuse_tokens_chunk_aligned"], 1024 + 30 * 1280)
        self.assertEqual(d["common_prefix_all"], 1024)
        self.assertEqual(d["longest_shared_prefix"], 1280)
        self.assertGreater(r.candidates[0].memory_saved_bytes, 0)
        self.assertGreater(r.candidates[0].prefill_compute_saved_flops, 0)

    def test_no_prompts_rejected(self):
        ctx = sample("high_pressure.json")
        ctx.workload.prompt_spec = None
        r = PrefixSharePass().run(ctx)
        self.assertEqual(r.candidates, [])
        self.assertEqual(len(r.rejections), 1)


class TestPlanner(unittest.TestCase):
    def test_all_samples_improve(self):
        for n in ("high_pressure.json", "low_pressure.json", "consumer_gpu.json"):
            ctx = sample(n)
            plan = build_plan(ctx, run_passes(ctx))
            self.assertLessEqual(plan.optimized.total_gib, plan.baseline.total_gib)
            self.assertLessEqual(plan.optimized.tpot_ms, plan.baseline.tpot_ms * 1.05)

    def test_oom_baseline_becomes_fit(self):
        ctx = sample("high_pressure.json")
        plan = build_plan(ctx, run_passes(ctx))
        self.assertFalse(plan.baseline.fits)
        self.assertTrue(plan.optimized.fits)

    def test_evaluate_baseline_matches_hand_calc(self):
        ctx = sample("high_pressure.json")
        m = evaluate(ctx)
        self.assertAlmostEqual(m.kv_gib, 64.0, places=6)
        self.assertEqual(paged_tokens(ctx), 32 * 2560)


class TestInputAndCLI(unittest.TestCase):
    def test_bad_input(self):
        with self.assertRaises(ValueError):
            load_context({"model": "llama-7b", "hardware": "H100",
                          "workload": {"batch_size": 1, "prompt_length": 100, "generation_length": 100, "max_context": 50}})

    def test_cli_json(self):
        out = os.path.join(ROOT, "tests", "_out.json")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "main.py"), "-i",
                            os.path.join(ROOT, "samples", "high_pressure.json"), "--json", out],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(out) as f:
            data = json.load(f)
        os.remove(out)
        self.assertEqual(data["plan"]["selected"], ["int8_kv", "paged_kv", "prefix_share"])


if __name__ == "__main__":
    unittest.main()
