#!/usr/bin/env bash
python3 main.py simple_tag MA_DD_SNR50 \
    --agent_alg DDPG \
    --adversary_alg MADDPG \
    --noisy_SNR 50
    --validate_every_n_eps 100 \
    --run_n_eps_in_validation 10
