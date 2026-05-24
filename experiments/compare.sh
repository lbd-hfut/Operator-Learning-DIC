#!/bin/bash
# Compare Route A and Route B on the same test dataset

echo "=== Comparing DIC Solver Operator vs Deformation Inverse Operator ==="

ROUTE_A_CKPT=${1:-"checkpoints/solver_operator/best.pt"}
ROUTE_B_CKPT=${2:-"checkpoints/inverse_operator/best.pt"}

echo ""
echo "--- Route A: DIC Solver Operator ---"
python -m dic_solver_operator.eval --checkpoint "$ROUTE_A_CKPT" --n_samples 1000

echo ""
echo "--- Route B: Deformation Inverse Operator ---"
python -m deformation_inverse_operator.eval --checkpoint "$ROUTE_B_CKPT" --n_samples 1000
