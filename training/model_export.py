"""
Model Export Pipeline for Financial Trading RL Gym.
Exports trained PPO/Qwen models to ONNX, quantized formats, and Triton-compatible.

Usage:
    python training/model_export.py --model models/qwen_final_model.zip
    python training/model_export.py --model models/qwen_final_model.zip --quantize fp16 --benchmark
    python training/model_export.py --model models/qwen_final_model_200k.zip --output-dir models/exported/v2
"""
import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

warnings.filterwarnings("ignore")

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def export_to_onnx(
    model_path: str,
    output_path: str = "models/exported/model.onnx",
    obs_dim: int = 49,
    verbose: bool = True,
) -> Optional[str]:
    """
    Export a trained SB3 model's policy network to ONNX format.

    Args:
        model_path: Path to the SB3 model zip file (with or without .zip)
        output_path: Destination path for the ONNX model
        obs_dim: Observation dimension (default 49 for financial trading env)
        verbose: Whether to print progress

    Returns:
        Path to exported ONNX file, or None on failure
    """
    from stable_baselines3 import PPO
    from gymnasium import spaces

    if verbose:
        print(f"[ONNX] Loading model from {model_path}")

    # Ensure model path has .zip
    model_zip = model_path if model_path.endswith(".zip") else model_path + ".zip"
    if not os.path.exists(model_zip):
        print(f"[ERROR] Model not found: {model_zip}")
        return None

    try:
        model = PPO.load(model_zip)
        policy = model.policy
        policy.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load PPO model: {e}")
        print("[INFO] Attempting alternative loading...")
        try:
            model = PPO.load(model_path)
            policy = model.policy
            policy.eval()
        except Exception as e2:
            print(f"[ERROR] Alternative loading also failed: {e2}")
            return None

    # Export policy to ONNX
    dummy_input = torch.randn(1, obs_dim)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    try:
        torch.onnx.export(
            policy,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["observation"],
            output_names=["action", "value"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action": {0: "batch_size"},
                "value": {0: "batch_size"},
            },
        )
    except Exception as e:
        print(f"[ERROR] ONNX export failed: {e}")
        return None

    if verbose:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[ONNX] Model exported to {output_path} ({size_mb:.2f} MB)")

    return output_path


def quantize_onnx(
    onnx_path: str,
    output_path: Optional[str] = None,
    quantize_type: str = "int8",
) -> Optional[str]:
    """
    Quantize an ONNX model to int8, uint8, or fp16.

    Args:
        onnx_path: Path to the ONNX model
        output_path: Destination path (auto-generated if None)
        quantize_type: 'int8', 'uint8', or 'fp16'

    Returns:
        Path to quantized model, or None on failure
    """
    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX model not found: {onnx_path}")
        return None

    if output_path is None:
        base = os.path.splitext(onnx_path)[0]
        output_path = f"{base}_{quantize_type}.onnx"

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    try:
        if quantize_type in ("int8", "uint8"):
            from onnxruntime.quantization import quantize_dynamic, QuantType

            weight_type = QuantType.QInt8 if quantize_type == "int8" else QuantType.QUInt8

            quantize_dynamic(
                onnx_path,
                output_path,
                weight_type=weight_type,
            )
        elif quantize_type == "fp16":
            import onnx
            from onnxconverter_common import float16

            model = onnx.load(onnx_path)
            model_fp16 = float16.convert_float_to_float16(model)
            onnx.save(model_fp16, output_path)
        else:
            print(f"[ERROR] Unknown quantize type: {quantize_type}")
            return None

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        orig_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        ratio = size_mb / orig_size_mb if orig_size_mb > 0 else 1.0

        print(f"[QUANT] Quantized ({quantize_type}) model saved to {output_path}")
        print(f"  Size: {orig_size_mb:.2f} MB -> {size_mb:.2f} MB ({ratio*100:.0f}% of original)")
        return output_path

    except ImportError as e:
        print(f"[ERROR] Quantization tools not installed: {e}")
        print("  Install with: pip install onnx onnxruntime onnxconverter-common")
        return None
    except Exception as e:
        print(f"[ERROR] Quantization failed: {e}")
        return None


def benchmark_inference(
    model_path: str,
    n_runs: int = 1000,
    obs_dim: int = 49,
) -> Optional[Dict[str, float]]:
    """
    Benchmark inference latency of an ONNX model.

    Args:
        model_path: Path to the ONNX model
        n_runs: Number of inference runs for benchmarking
        obs_dim: Observation dimension

    Returns:
        Dictionary with latency and throughput metrics
    """
    try:
        import onnxruntime as ort

        if not os.path.exists(model_path):
            print(f"[ERROR] Model not found: {model_path}")
            return None

        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name

        dummy = np.random.randn(1, obs_dim).astype(np.float32)

        # Warmup runs
        for _ in range(100):
            session.run(None, {input_name: dummy})

        # Benchmark
        start = time.perf_counter()
        for _ in range(n_runs):
            session.run(None, {input_name: dummy})
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / n_runs) * 1000.0
        throughput = n_runs / elapsed

        print(f"[BENCH] Model: {model_path}")
        print(f"  Avg latency: {avg_latency_ms:.2f} ms")
        print(f"  Throughput: {throughput:.0f} inferences/sec")
        print(f"  Runs: {n_runs}")

        # Validate output
        outputs = session.run(None, {input_name: dummy})
        action = outputs[0]
        value = outputs[1] if len(outputs) > 1 else None
        print(f"  Action shape: {action.shape}, Value: {value[0][0]:.4f}" if value is not None else f"  Action shape: {action.shape}")

        return {
            "latency_ms": round(avg_latency_ms, 2),
            "throughput": round(throughput, 0),
            "model_path": model_path,
            "n_runs": n_runs,
            "obs_dim": obs_dim,
        }

    except ImportError:
        print("[ERROR] ONNX Runtime not installed. Install with: pip install onnxruntime")
        return None
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {e}")
        return None


def verify_onnx_model(onnx_path: str, obs_dim: int = 49) -> bool:
    """
    Verify an ONNX model by running a full inference pass and checking outputs.

    Args:
        onnx_path: Path to the ONNX model
        obs_dim: Observation dimension

    Returns:
        True if verification passes, False otherwise
    """
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)

        # Check input/output names
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]

        print(f"[VERIFY] Inputs: {input_names}")
        print(f"  Outputs: {output_names}")

        # Run verification
        dummy = np.random.randn(1, obs_dim).astype(np.float32)
        outputs = session.run(None, {input_names[0]: dummy})

        print(f"  Output shapes: {[o.shape for o in outputs]}")
        print(f"  Action sample: {outputs[0][0][:3]}")
        if len(outputs) > 1:
            print(f"  Value estimate: {outputs[1][0][0]:.4f}")

        print(f"[VERIFY] Model verification PASSED")
        return True

    except Exception as e:
        print(f"[VERIFY] Model verification FAILED: {e}")
        return False


def export_artifacts_summary(
    artifacts: Dict[str, Any],
    output_dir: str,
):
    """Save a summary JSON of all exported artifacts."""
    summary_path = os.path.join(output_dir, "export_summary.json")
    with open(summary_path, "w") as f:
        json.dump(artifacts, f, indent=2, default=str)
    print(f"[INFO] Export summary saved to {summary_path}")


IMPORT_HELP = """
Required packages for full export pipeline:
    pip install onnx onnxruntime onnxconverter-common
"""


if __name__ == "__main__":
    import argparse

    # Late import to avoid circular issues; torch is only needed for export
    import torch

    parser = argparse.ArgumentParser(
        description="Model export pipeline for RL trading gym",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=IMPORT_HELP,
    )
    parser.add_argument("--model", default="models/qwen_final_model.zip",
                        help="Path to SB3 model (default: models/qwen_final_model.zip)")
    parser.add_argument("--output-dir", default="models/exported",
                        help="Output directory for exported models")
    parser.add_argument("--quantize", choices=["int8", "uint8", "fp16", "none"],
                        default="int8", help="Quantization type (default: int8)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark inference latency after export")
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX model after export")
    parser.add_argument("--obs-dim", type=int, default=49,
                        help="Observation dimension (default: 49)")
    parser.add_argument("--benchmark-runs", type=int, default=1000,
                        help="Number of inference runs for benchmark (default: 1000)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("MODEL EXPORT PIPELINE")
    print(f"Input: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Quantize: {args.quantize}")
    print("=" * 60)

    artifacts = {
        "source_model": args.model,
        "output_dir": args.output_dir,
        "export_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": [],
    }

    # Step 1: Export to ONNX
    print("\n[STEP 1/3] Exporting to ONNX...")
    onnx_path = export_to_onnx(
        args.model,
        output_path=os.path.join(args.output_dir, "model.onnx"),
        obs_dim=args.obs_dim,
    )

    if onnx_path is None:
        print("[FAILED] ONNX export failed. Exiting.")
        export_artifacts_summary(artifacts, args.output_dir)
        sys.exit(1)

    artifacts["files"].append({"format": "onnx", "path": onnx_path, "size_mb": round(os.path.getsize(onnx_path) / (1024 * 1024), 2)})

    # Step 2: Quantize
    quant_path = onnx_path
    if args.quantize != "none":
        print(f"\n[STEP 2/3] Quantizing to {args.quantize}...")
        quant_path = quantize_onnx(
            onnx_path,
            quantize_type=args.quantize,
        )
        if quant_path:
            artifacts["files"].append({
                "format": f"onnx_{args.quantize}",
                "path": quant_path,
                "size_mb": round(os.path.getsize(quant_path) / (1024 * 1024), 2),
            })

    # Step 3: Verify
    if args.verify and quant_path:
        print(f"\n[STEP 3/3] Verifying model...")
        verify_onnx_model(quant_path, obs_dim=args.obs_dim)

    # Step 4: Benchmark (if requested)
    if args.benchmark and quant_path:
        print(f"\n[BENCHMARK] Benchmarking inference...")
        bench_results = benchmark_inference(
            quant_path,
            n_runs=args.benchmark_runs,
            obs_dim=args.obs_dim,
        )
        if bench_results:
            artifacts["benchmark"] = bench_results

    # Save summary
    print("\n" + "=" * 60)
    export_artifacts_summary(artifacts, args.output_dir)
    print("Export pipeline complete!")
    print("=" * 60)
