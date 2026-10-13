"""
Unit tests for the IR builder and analyzer.

Expected values are derived by hand from the formulas (see README), so these
tests check the math, not just that the code runs.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import (analyze, estimate_op, fits_in_memory, kv_cache_bytes,
                      kv_cache_gb, weights_bytes, weights_gb)
from common import Bound, Op, bytes_for_elements
from hardware import get_hardware
from ir_builder import build_ir
from main import main
from models import count_parameters, get_model

LLAMA = get_model("llama-7b")
MISTRAL = get_model("mistral-7b")
OPS_PER_LAYER = 5


def ops_of_type(ir, op_type):
    return [o for o in ir if o.op is op_type]


class TestModelPresets(unittest.TestCase):
    def test_llama_layer_count(self):
        self.assertEqual(LLAMA.num_layers, 32)

    def test_mistral_layer_count(self):
        self.assertEqual(MISTRAL.num_layers, 32)

    def test_mistral_uses_gqa(self):
        self.assertEqual(MISTRAL.num_kv_heads, 8)
        self.assertEqual(MISTRAL.kv_dim, 8 * 128)
        self.assertEqual(LLAMA.kv_dim, LLAMA.hidden_size)  # MHA

    def test_llama_parameter_count_exact(self):
        # per layer: 4*d^2 (attn) + 3*d*f (MLP) + 2*d (norms)
        d, f, v = 4096, 11008, 32000
        per_layer = 4 * d * d + 3 * d * f + 2 * d
        expected = 32 * per_layer + 2 * v * d + d
        self.assertEqual(count_parameters(LLAMA), expected)
        self.assertEqual(expected, 6_738_415_616)

    def test_computed_params_match_reported(self):
        for model in (LLAMA, MISTRAL):
            rel_error = abs(count_parameters(model) - model.reported_params) / model.reported_params
            self.assertLess(rel_error, 0.01, model.name)

    def test_unknown_model_and_hardware(self):
        with self.assertRaises(ValueError):
            get_model("gpt-5")
        with self.assertRaises(ValueError):
            get_hardware("v100")


class TestIR(unittest.TestCase):
    def test_number_of_ops(self):
        ir = build_ir(LLAMA, sequence_length=2048)
        self.assertEqual(len(ir), LLAMA.num_layers * OPS_PER_LAYER)

    def test_op_order_per_layer(self):
        ir = build_ir(LLAMA, sequence_length=16)
        expected = [Op.QKV, Op.ATTENTION, Op.SOFTMAX, Op.OUT_PROJ, Op.MLP]
        self.assertEqual([o.op for o in ir[:OPS_PER_LAYER]], expected)
        self.assertEqual({o.layer for o in ir}, set(range(32)))

    def test_flops_and_bytes_positive(self):
        for model in (LLAMA, MISTRAL):
            for op in build_ir(model, sequence_length=128, batch_size=2):
                self.assertGreater(op.flops, 0)
                self.assertGreater(op.bytes, 0)

    def test_qkv_flops_formula_mha_and_gqa(self):
        seq, batch = 2048, 2
        tokens, d = seq * batch, 4096
        llama_qkv = ops_of_type(build_ir(LLAMA, seq, batch), Op.QKV)[0]
        mistral_qkv = ops_of_type(build_ir(MISTRAL, seq, batch), Op.QKV)[0]
        # 2 * T * d * (d + 2*d_kv)
        self.assertEqual(llama_qkv.flops, 2 * tokens * d * (d + 2 * 4096))
        self.assertEqual(mistral_qkv.flops, 2 * tokens * d * (d + 2 * 1024))
        self.assertLess(mistral_qkv.flops, llama_qkv.flops)

    def test_attention_softmax_formulas(self):
        seq, batch = 1024, 3
        ir = build_ir(MISTRAL, seq, batch)
        attention = ops_of_type(ir, Op.ATTENTION)[0]
        softmax = ops_of_type(ir, Op.SOFTMAX)[0]
        scores = batch * 32 * seq * seq
        # 4 * B * h * S^2 * d_h (GQA does not reduce attention FLOPs)
        self.assertEqual(attention.flops, 4 * scores * 128)
        self.assertEqual(softmax.flops, 5 * scores)
        self.assertEqual(softmax.bytes, 2 * scores * 2)  # read + write, fp16

    def test_outproj_and_mlp_formulas(self):
        seq = 512
        ir = build_ir(LLAMA, seq)
        self.assertEqual(ops_of_type(ir, Op.OUT_PROJ)[0].flops, 2 * seq * 4096 * 4096)
        self.assertEqual(ops_of_type(ir, Op.MLP)[0].flops, 3 * 2 * seq * 4096 * 11008)

    def test_qkv_bytes_formula(self):
        seq, d, width = 100, 4096, 4096 + 2 * 1024
        qkv = ops_of_type(build_ir(MISTRAL, seq), Op.QKV)[0]
        elements = seq * d + d * width + seq * width
        self.assertEqual(qkv.bytes, elements * 2)

    def test_linear_flops_match_2_params_per_token(self):
        # The projections + MLP do 2 FLOPs per weight per token.
        seq = 256
        ir = build_ir(LLAMA, seq)
        linear_ops = (Op.QKV, Op.OUT_PROJ, Op.MLP)
        linear_flops = sum(o.flops for o in ir if o.op in linear_ops)
        layer_weights = 4 * 4096 * 4096 + 3 * 4096 * 11008
        self.assertEqual(linear_flops, 2 * seq * 32 * layer_weights)

    def test_invalid_precision(self):
        with self.assertRaises(ValueError):
            build_ir(LLAMA, 128, precision="fp8")
        with self.assertRaises(ValueError):
            weights_gb(LLAMA, "float16")

    def test_invalid_sizes(self):
        with self.assertRaises(ValueError):
            build_ir(LLAMA, 0)


class TestMemory(unittest.TestCase):
    def test_weight_memory(self):
        params = count_parameters(LLAMA)
        self.assertEqual(weights_bytes(LLAMA, "fp16"), params * 2)
        self.assertEqual(weights_bytes(LLAMA, "fp32"), params * 4)
        self.assertEqual(weights_bytes(LLAMA, "int8"), params)
        self.assertAlmostEqual(weights_gb(LLAMA, "fp16"), params * 2 / 1e9)
        self.assertAlmostEqual(weights_gb(LLAMA, "fp16"), 13.48, places=2)

    def test_int4_is_exactly_half_a_byte(self):
        params = count_parameters(LLAMA)
        self.assertEqual(weights_bytes(LLAMA, "int4"), params // 2)
        self.assertEqual(bytes_for_elements(3, "int4"), 2)  # 12 bits -> 2 bytes

    def test_kv_cache_formula(self):
        # 2 * L * B * S * h_kv * d_h * bytes
        self.assertEqual(kv_cache_bytes(LLAMA, 2048, 1, "fp16"), 2 * 32 * 1 * 2048 * 32 * 128 * 2)
        self.assertEqual(kv_cache_bytes(LLAMA, 2048, 1, "fp16"), 1_073_741_824)
        self.assertEqual(kv_cache_bytes(MISTRAL, 4096, 1, "fp16"), 536_870_912)
        self.assertAlmostEqual(kv_cache_gb(MISTRAL, 4096, 1, "fp16"), 0.536870912)

    def test_kv_cache_scales_linearly(self):
        base = kv_cache_bytes(LLAMA, 1024, 1, "fp16")
        self.assertEqual(kv_cache_bytes(LLAMA, 2048, 1, "fp16"), 2 * base)
        self.assertEqual(kv_cache_bytes(LLAMA, 1024, 4, "fp16"), 4 * base)

    def test_gqa_reduces_kv_cache(self):
        llama_kv = kv_cache_bytes(LLAMA, 4096, 1, "fp16")
        mistral_kv = kv_cache_bytes(MISTRAL, 4096, 1, "fp16")
        # same L, d_h; h_kv 32 vs 8 -> exactly 4x smaller
        self.assertEqual(llama_kv, 4 * mistral_kv)


class TestFitAndRoofline(unittest.TestCase):
    def test_fit_boundary(self):
        t4 = get_hardware("t4")
        self.assertTrue(fits_in_memory(16e9, t4))
        self.assertFalse(fits_in_memory(16e9 + 1, t4))

    def test_fit_decisions(self):
        t4, h100 = get_hardware("t4"), get_hardware("h100")
        # 13.48 GB + 1.07 GB = 14.55 GB <= 16 GB
        self.assertTrue(analyze(LLAMA, t4, 2048, 1, "fp16").fits)
        # fp32 weights alone are 26.95 GB > 16 GB
        self.assertFalse(analyze(LLAMA, t4, 2048, 1, "fp32").fits)
        # KV for B=8, S=4096: 2*32*8*4096*4096*2 = 17.18 GB -> 30.66 GB total
        big = analyze(LLAMA, t4, 4096, 8, "fp16")
        self.assertEqual(big.total_memory_bytes,
                         weights_bytes(LLAMA, "fp16") + 2 * 32 * 8 * 4096 * 4096 * 2)
        self.assertFalse(big.fits)
        self.assertTrue(analyze(LLAMA, h100, 4096, 8, "fp16").fits)

    def test_roofline_times(self):
        a100 = get_hardware("a100")
        op = build_ir(LLAMA, 2048)[0]
        est = estimate_op(op, a100)
        self.assertAlmostEqual(est.compute_time_s, op.flops / 312e12)
        self.assertAlmostEqual(est.memory_time_s, op.bytes / 2039e9)
        self.assertEqual(est.estimated_time_s, max(est.compute_time_s, est.memory_time_s))

    def test_bound_matches_ridge_point(self):
        a100 = get_hardware("a100")
        for op in build_ir(MISTRAL, 2048):
            expected = (Bound.COMPUTE_BOUND if op.arithmetic_intensity >= a100.ridge_point
                        else Bound.MEMORY_BOUND)
            self.assertEqual(estimate_op(op, a100).bound, expected)

    def test_softmax_is_memory_bound(self):
        report = analyze(LLAMA, get_hardware("a100"), 2048)
        softmax = [e for e in report.estimates if e.op.op is Op.SOFTMAX][0]
        self.assertEqual(softmax.bound, Bound.MEMORY_BOUND)

    def test_report_totals(self):
        report = analyze(LLAMA, get_hardware("a100"), 512, 2)
        ir = build_ir(LLAMA, 512, 2)
        self.assertEqual(report.total_flops, sum(o.flops for o in ir))
        self.assertEqual(report.total_bytes, sum(o.bytes for o in ir))
        self.assertAlmostEqual(report.arithmetic_intensity, report.total_flops / report.total_bytes)


class TestCLI(unittest.TestCase):
    def test_invalid_names_return_error_code(self):
        devnull = open(os.devnull, "w")
        old_stderr, sys.stderr = sys.stderr, devnull
        try:
            self.assertEqual(main(["--precision", "fp8"]), 2)
            self.assertEqual(main(["--model", "gpt-5"]), 2)
            self.assertEqual(main(["--hardware", "v100"]), 2)
        finally:
            sys.stderr = old_stderr
            devnull.close()


if __name__ == "__main__":
    unittest.main()
