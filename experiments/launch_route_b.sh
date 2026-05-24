#!/bin/bash
# Launch Deformation Inverse Operator (Route B) training
# Single GPU: bash experiments/launch_route_b.sh
# Multi GPU:  bash experiments/launch_route_b.sh --ddp 4

N_GPU=${1:-1}

if [ "$N_GPU" -gt 1 ]; then
    echo "Launching Route B training with $N_GPU GPUs (DDP)..."
    torchrun --nproc_per_node=$N_GPU -m deformation_inverse_operator.train --use_ddp
else
    echo "Launching Route B training (single GPU)..."
    python -m deformation_inverse_operator.train
fi
