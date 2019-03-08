#!/usr/bin/env bash
# MA+DD pairs
for SNR in 50 0 -5 -20 ; do
    echo "***** MA+DD SNR${SNR} *****" 
    rm -rf ./models/simple_tag/tag_MA_DD_SNR${SNR}/ 
    python3 main.py simple_tag tag_MA_DD_SNR${SNR} \
    --agent_alg DDPG \
    --adversary_alg MADDPG \
    --noisy_SNR ${SNR} \
    --validate_every_n_eps 100 \
    --run_n_eps_in_validation 10
done

# only 1 DD+DD pair
echo "***** DD+DD *****"
rm -rf ./models/simple_tag/tag_DD_DD
python3 main.py simple_tag tag_DD_DD \
    --agent_alg DDPG \
    --adversary_alg DDPG \
    --validate_every_n_eps 100 \
    --run_n_eps_in_validation 10
