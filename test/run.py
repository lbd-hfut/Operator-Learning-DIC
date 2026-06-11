"""CLI entry point for FEM-based irregular-ROI test cases.

Usage:
    # Run all three cases (generate + predict)
    python -m test.run --all

    # Single case
    python -m test.run --case circle
    python -m test.run --case ring
    python -m test.run --case notch

    # Generate only, no prediction
    python -m test.run --case circle --generate-only

    # Predict only (use previously generated images)
    python -m test.run --case circle --predict-only

    # Custom parameters
    python -m test.run --case circle --image_size 256 --disp_range -4 4 --seed 42
    python -m test.run --all --speckle_source path/to/speckles
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


DEFAULT_OUTPUT_DIR = _project_root / "test" / "output"
DEFAULT_CKPT_A = _project_root / "checkpoints" / "route_a" / "best.pt"
DEFAULT_CKPT_B = _project_root / "checkpoints" / "route_b" / "best.pt"
DEFAULT_CKPT_C = _project_root / "checkpoints" / "route_c" / "best.pt"
DEFAULT_CKPT_D = _project_root / "checkpoints" / "route_d" / "best.pt"

CASES = ["circle", "ring", "notch", "full"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="FEM-based irregular-ROI DIC test cases (Route A/B/C/D)"
    )
    # Case selection
    parser.add_argument("--all", action="store_true",
                        help="Run all three cases (circle, ring, notch)")
    parser.add_argument("--case", type=str, default=None,
                        choices=CASES,
                        help="Single case to run")

    # Mode
    parser.add_argument("--generate-only", action="store_true",
                        help="Generate images only, skip prediction")
    parser.add_argument("--predict-only", action="store_true",
                        help="Predict only, use existing generated images")

    # Parameters
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size (square, default: 256)")
    parser.add_argument("--disp_range", type=float, nargs=2, default=[-4, 4],
                        help="Displacement range in pixels (default: -4 4)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--speckle_source", type=str, default=None,
                        help="Path to speckle image (default: test/speckle.png)")

    # Checkpoints
    parser.add_argument("--ckpt_a", type=str, default=str(DEFAULT_CKPT_A),
                        help="Route A checkpoint path")
    parser.add_argument("--ckpt_b", type=str, default=str(DEFAULT_CKPT_B),
                        help="Route B checkpoint path")
    parser.add_argument("--ckpt_c", type=str, default=str(DEFAULT_CKPT_C),
                        help="Route C checkpoint path")
    parser.add_argument("--ckpt_d", type=str, default=str(DEFAULT_CKPT_D),
                        help="Route D checkpoint path")

    # Output
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory (default: test/output)")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Inference device (default: cuda)")

    args = parser.parse_args()

    if not args.all and args.case is None:
        parser.error("Must specify --all or --case {circle,ring,notch}")

    return args


def main():
    args = parse_args()

    cases_to_run = CASES if args.all else [args.case]
    image_size = (args.image_size, args.image_size)

    print(f"{'='*60}")
    print(f"FEM-based Irregular-ROI DIC Test")
    print(f"  Cases: {cases_to_run}")
    print(f"  Image size: {image_size}")
    print(f"  Displacement range: {args.disp_range}")
    print(f"  Seed: {args.seed}")
    print(f"  Mode: {'generate-only' if args.generate_only else 'predict-only' if args.predict_only else 'full'}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")

    all_results = {}

    for case in cases_to_run:
        case_output_dir = Path(args.output_dir) / case

        if not args.predict_only:
            # ── Generate ──────────────────────────────────────────
            print(f"\n{'─'*60}")
            print(f"Generating: {case}")
            from test.generate import generate_test_case

            data = generate_test_case(
                case=case,
                output_dir=case_output_dir,
                image_size=image_size,
                disp_range=tuple(args.disp_range),
                seed=args.seed,
                speckle_source=args.speckle_source,
            )
        else:
            # ── Load existing ─────────────────────────────────────
            print(f"\n{'─'*60}")
            print(f"Loading existing: {case}")
            from PIL import Image

            ref_path = case_output_dir / "ref.png"
            tar_path = case_output_dir / "tar.png"
            u_path = case_output_dir / "u_field.npy"
            roi_path = case_output_dir / "roi_mask.png"

            if not ref_path.exists():
                print(f"  ERROR: {ref_path} not found. Run without --predict-only first.")
                continue

            import numpy as np
            ref = np.array(Image.open(ref_path), dtype=np.float32) / 255.0
            tar = np.array(Image.open(tar_path), dtype=np.float32) / 255.0
            u_field = np.load(u_path)
            roi = np.array(Image.open(roi_path)) > 127

            data = {
                "ref_array": ref,
                "tar_array": tar,
                "u_field_array": u_field,
                "roi_array": roi,
            }

        if not args.generate_only:
            # ── Predict ───────────────────────────────────────────
            from test.predict import evaluate_case

            save_plot = case_output_dir / f"{case}_results.png"
            results = evaluate_case(
                case_name=case,
                data_dict=data,
                ckpt_a=args.ckpt_a,
                ckpt_b=args.ckpt_b,
                ckpt_c=args.ckpt_c,
                ckpt_d=args.ckpt_d,
                device=args.device,
                save_plot=str(save_plot),
            )
            all_results[case] = results

    # ── Summary ────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        route_order = ["A", "B", "C", "D"]
        for case, results in all_results.items():
            print(f"\n{case}:")
            for route in route_order:
                if route in results:
                    _, mae, mse = results[route]
                    print(f"  Route {route}: MAE={mae:.4f}  MSE={mse:.6f}")

    print(f"\nDone. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
