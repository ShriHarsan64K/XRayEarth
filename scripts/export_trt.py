"""
XRayEarth — export_trt.py
Seeing through disaster with satellite vision.

Pipeline:
    Trained .pth checkpoint
         │
         ▼
    PyTorch model (eval mode)
         │
         ▼
    ONNX export (opset 17)
         │
         ▼
    TensorRT engine (.engine)
         │
         ▼
    Latency benchmark
    (PyTorch vs TensorRT)

Usage:
    python scripts/export_trt.py \
        --checkpoint outputs/checkpoints/v10_best.pth \
        --config     configs/v10.yaml \
        --output-dir outputs/trt \
        --benchmark

NOTE: Run on Machine B only (requires TensorRT + pycuda)
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import (
    load_config,
    load_checkpoint,
    get_device,
    print_banner,
    console,
)
from model import build_model


# ═══════════════════════════════════════════════════════════
#  1. ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XRayEarth — TensorRT Export"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained .pth checkpoint"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment config"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="outputs/trt",
        help="Directory to save ONNX and TRT engine"
    )
    parser.add_argument(
        "--tile-size", type=int, default=None,
        help="Input tile size (overrides config)"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run latency benchmark after export"
    )
    parser.add_argument(
        "--benchmark-runs", type=int, default=100,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True,
        help="Use FP16 precision in TensorRT (default: True)"
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════
#  2. ONNX EXPORT
# ═══════════════════════════════════════════════════════════

def export_to_onnx(
    model:      torch.nn.Module,
    output_path: str,
    tile_size:  int,
    device:     torch.device,
) -> str:
    """
    Export XRayEarth model to ONNX format.

    For Siamese models, we export the POST image stream only
    (single-image ONNX) since TensorRT has limited support
    for dynamic multi-input Siamese architectures.

    For deployment, pre-image features can be cached separately.

    Args:
        model:       Trained XRayEarthModel in eval mode
        output_path: Path to save .onnx file
        tile_size:   Input tile size
        device:      CUDA device

    Returns:
        Path to saved ONNX file
    """
    import torch.onnx

    console.log("[bold]Exporting to ONNX...[/bold]")

    model.eval()

    # Dummy input: batch of 1 tile
    dummy_pre  = [torch.rand(3, tile_size, tile_size).to(device)]
    dummy_post = [torch.rand(3, tile_size, tile_size).to(device)]

    output_path = str(output_path)

    # Export using torch.onnx
    # We trace the detector backbone + FPN only
    # (RPN + ROI heads are exported as-is)
    with torch.no_grad():
        torch.onnx.export(
            model.detector,
            (dummy_post,),         # single image input for ONNX
            output_path,
            opset_version    = 17,
            input_names      = ["images"],
            output_names     = ["boxes", "labels", "scores", "masks"],
            dynamic_axes     = {
                "images": {0: "batch_size"},
            },
            verbose          = False,
            do_constant_folding = True,
        )

    file_size = Path(output_path).stat().st_size / 1e6
    console.log(
        f"[green]✓[/green] ONNX exported: "
        f"[bold]{output_path}[/bold] ({file_size:.1f} MB)"
    )
    return output_path


# ═══════════════════════════════════════════════════════════
#  3. ONNX VALIDATION
# ═══════════════════════════════════════════════════════════

def validate_onnx(onnx_path: str, tile_size: int) -> bool:
    """
    Validate ONNX model using onnxruntime.

    Args:
        onnx_path: Path to .onnx file
        tile_size: Input tile size

    Returns:
        True if validation passes
    """
    try:
        import onnx
        import onnxruntime as ort

        console.log("[bold]Validating ONNX model...[/bold]")

        # Check model structure
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)

        # Run inference with onnxruntime
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session   = ort.InferenceSession(onnx_path, providers=providers)

        dummy = np.random.rand(1, 3, tile_size, tile_size).astype(np.float32)
        input_name = session.get_inputs()[0].name

        outputs = session.run(None, {input_name: dummy})

        console.log(
            f"[green]✓[/green] ONNX validation passed "
            f"({len(outputs)} outputs)"
        )
        return True

    except Exception as e:
        console.log(f"[red]✗[/red] ONNX validation failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════
#  4. TENSORRT CONVERSION
# ═══════════════════════════════════════════════════════════

def convert_to_tensorrt(
    onnx_path:   str,
    engine_path: str,
    tile_size:   int,
    fp16:        bool = True,
) -> str:
    """
    Convert ONNX model to TensorRT engine.

    Optimizations applied:
        - Layer fusion
        - Kernel auto-tuning
        - FP16 precision (if enabled)
        - Dynamic shape support

    Args:
        onnx_path:   Path to .onnx file
        engine_path: Path to save .engine file
        tile_size:   Input tile size
        fp16:        Enable FP16 precision

    Returns:
        Path to saved TensorRT engine
    """
    try:
        import tensorrt as trt
    except ImportError:
        console.log(
            "[red]✗[/red] TensorRT not installed. "
            "Install with: pip install tensorrt==8.6.1"
        )
        raise

    console.log(
        f"[bold]Converting to TensorRT "
        f"({'FP16' if fp16 else 'FP32'})...[/bold]"
    )

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser  = trt.OnnxParser(network, TRT_LOGGER)
    config  = builder.create_builder_config()

    # Set memory pool (4GB for RTX 5060)
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30)
    )

    # Enable FP16
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        console.log("[cyan]→[/cyan] FP16 mode enabled")

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                console.log(f"[red]TRT Parse Error: {parser.get_error(i)}[/red]")
            raise RuntimeError("Failed to parse ONNX model")

    # Dynamic shape profile
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images",
        min=(1, 3, tile_size, tile_size),
        opt=(2, 3, tile_size, tile_size),
        max=(4, 3, tile_size, tile_size),
    )
    config.add_optimization_profile(profile)

    # Build engine (this takes a few minutes)
    console.log("[yellow]→[/yellow] Building TRT engine (may take 2-5 min)...")
    serialized = builder.build_serialized_network(network, config)

    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized)

    file_size = Path(engine_path).stat().st_size / 1e6
    console.log(
        f"[green]✓[/green] TensorRT engine saved: "
        f"[bold]{engine_path}[/bold] ({file_size:.1f} MB)"
    )
    return engine_path


# ═══════════════════════════════════════════════════════════
#  5. LATENCY BENCHMARK
# ═══════════════════════════════════════════════════════════

def benchmark_pytorch(
    model:     torch.nn.Module,
    tile_size: int,
    device:    torch.device,
    n_runs:    int = 100,
) -> Dict:
    """
    Benchmark PyTorch model inference latency.

    Args:
        model:     Model in eval mode
        tile_size: Input tile size
        device:    CUDA device
        n_runs:    Number of timed runs

    Returns:
        Dict with mean, std, min, max latency in ms
    """
    from typing import Dict

    model.eval()
    dummy = [torch.rand(3, tile_size, tile_size).to(device)]

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy, dummy)

    # Timed runs
    torch.cuda.synchronize()
    latencies = []

    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(dummy, dummy)
            torch.cuda.synchronize()
            end   = time.perf_counter()
            latencies.append((end - start) * 1000)

    return {
        "mean_ms": np.mean(latencies),
        "std_ms":  np.std(latencies),
        "min_ms":  np.min(latencies),
        "max_ms":  np.max(latencies),
        "fps":     1000 / np.mean(latencies),
    }


def benchmark_tensorrt(
    engine_path: str,
    tile_size:   int,
    n_runs:      int = 100,
) -> dict:
    """
    Benchmark TensorRT engine inference latency.

    Args:
        engine_path: Path to .engine file
        tile_size:   Input tile size
        n_runs:      Number of timed runs

    Returns:
        Dict with mean, std, min, max latency in ms
    """
    try:
        import tensorrt as trt
        import pycuda.driver   as cuda
        import pycuda.autoinit
    except ImportError:
        console.log("[red]✗[/red] pycuda not installed")
        return {}

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime    = trt.Runtime(TRT_LOGGER)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate buffers
    dummy   = np.random.rand(1, 3, tile_size, tile_size).astype(np.float32)
    d_input = cuda.mem_alloc(dummy.nbytes)
    cuda.memcpy_htod(d_input, dummy)

    # Warmup
    for _ in range(10):
        context.execute_v2([int(d_input)])

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        context.execute_v2([int(d_input)])
        cuda.Context.synchronize()
        end   = time.perf_counter()
        latencies.append((end - start) * 1000)

    return {
        "mean_ms": np.mean(latencies),
        "std_ms":  np.std(latencies),
        "min_ms":  np.min(latencies),
        "max_ms":  np.max(latencies),
        "fps":     1000 / np.mean(latencies),
    }


def print_benchmark_results(
    pytorch_stats: dict,
    trt_stats:     dict,
) -> None:
    """Print side-by-side benchmark comparison."""
    from rich.table import Table
    from utils import console

    table = Table(title="🚀 Latency Benchmark — PyTorch vs TensorRT")
    table.add_column("Metric",    style="cyan")
    table.add_column("PyTorch",   style="yellow")
    table.add_column("TensorRT",  style="green")
    table.add_column("Speedup",   style="bold magenta")

    metrics = ["mean_ms", "min_ms", "max_ms", "fps"]
    labels  = ["Mean latency", "Min latency", "Max latency", "Throughput"]
    units   = ["ms", "ms", "ms", "FPS"]

    for metric, label, unit in zip(metrics, labels, units):
        pt_val  = pytorch_stats.get(metric, 0)
        trt_val = trt_stats.get(metric, 0)

        if metric == "fps":
            speedup = f"{trt_val / (pt_val + 1e-8):.2f}×"
        else:
            speedup = f"{pt_val / (trt_val + 1e-8):.2f}×"

        table.add_row(
            f"{label} ({unit})",
            f"{pt_val:.2f}",
            f"{trt_val:.2f}",
            speedup,
        )

    console.print(table)


# ═══════════════════════════════════════════════════════════
#  6. MAIN EXPORT PIPELINE
# ═══════════════════════════════════════════════════════════

def main() -> None:
    print_banner()
    args   = parse_args()
    cfg    = load_config(args.config)
    device = get_device()

    tile_size  = args.tile_size or cfg.dataset.tile_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    version    = cfg.project.version
    onnx_path  = output_dir / f"{version}.onnx"
    engine_path = output_dir / f"{version}.engine"

    console.rule(f"[bold blue]XRayEarth TRT Export — {version}[/bold blue]")
    console.log(f"  Checkpoint : {args.checkpoint}")
    console.log(f"  Tile size  : {tile_size}")
    console.log(f"  FP16       : {args.fp16}")
    console.log(f"  Output dir : {output_dir}")

    # ── Load model ────────────────────────────────────────
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    # ── ONNX Export ───────────────────────────────────────
    export_to_onnx(model, str(onnx_path), tile_size, device)
    validate_onnx(str(onnx_path), tile_size)

    # ── TensorRT Conversion ───────────────────────────────
    convert_to_tensorrt(
        str(onnx_path),
        str(engine_path),
        tile_size,
        fp16=args.fp16,
    )

    # ── Benchmark ─────────────────────────────────────────
    if args.benchmark:
        console.rule("[bold]Latency Benchmark[/bold]")

        console.log("[yellow]→[/yellow] Benchmarking PyTorch...")
        pt_stats = benchmark_pytorch(
            model, tile_size, device, args.benchmark_runs
        )

        console.log("[yellow]→[/yellow] Benchmarking TensorRT...")
        trt_stats = benchmark_tensorrt(
            str(engine_path), tile_size, args.benchmark_runs
        )

        print_benchmark_results(pt_stats, trt_stats)

    console.rule("[bold green]Export Complete[/bold green]")
    console.log(f"[green]✓[/green] ONNX:   {onnx_path}")
    console.log(f"[green]✓[/green] Engine: {engine_path}")


if __name__ == "__main__":
    main()
