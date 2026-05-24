#!/bin/bash
# Launch DIC Solver Operator (Route A) training
# Single GPU: bash experiments/launch_route_a.sh
# Multi GPU:  bash experiments/launch_route_a.sh --ddp 4

N_GPU=${1:-1}

if [ "$N_GPU" -gt 1 ]; then
    echo "Launching Route A training with $N_GPU GPUs (DDP)..."
    torchrun --nproc_per_node=$N_GPU -m dic_solver_operator.train --use_ddp
else
    echo "Launching Route A training (single GPU)..."
    python -m dic_solver_operator.train
fi
