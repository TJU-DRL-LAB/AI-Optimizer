CUDA_VISIBLE_DEVICES=1  python3 -m run_scripts.run_cadm_pets --dataset ant --policy_type CEM --n_candidate 200 \
--normalize_flag --ensemble_size 1 --n_particles 1 --deterministic_flag 1 --history_length 10 \
--future_length 10 --seed 0 &

#CUDA_VISIBLE_DEVICES=1  python3 -m run_scripts.run_cadm_pets --dataset slim_humanoid --policy_type CEM --n_candidate 200 \
#--normalize_flag --ensemble_size 1 --n_particles 1 --deterministic_flag 1 --history_length 10 \
#--future_length 10 --seed 0 &     #PE-TS + CaDM
#
#CUDA_VISIBLE_DEVICES=1  python3 -m run_scripts.run_cadm_pets --dataset slim_humanoid --policy_type CEM --n_candidate 200 \
#--normalize_flag --ensemble_size 5 --n_particles 20 --deterministic_flag 0 --history_length 10 \
#--future_length 10 --seed 0 &     #PE-TS + CaDM
#
#CUDA_VISIBLE_DEVICES=1 python -m run_scripts.run_pets --dataset slim_humanoid --policy_type CEM --n_candidate 200 \
#--normalize_flag --ensemble_size 1 --n_particles 1 --deterministic_flag 1 --seed 0      #Vanilla DM

#CUDA_VISIBLE_DEVICES=1 python -m run_scripts.run_pets --dataset slim_humanoid --policy_type CEM --n_candidate 200 \
#--normalize_flag --ensemble_size 5 --n_particles 20 --deterministic_flag 0 --seed 0  &    #PE-TS

#CUDA_VISIBLE_DEVICES=1 python -m run_scripts.model_free.run_ppo_cadm --entropy_coeff 0.0 --lr 0.0005 \
#--num_rollouts 10 --num_steps 200 --num_minibatches 4 --policy_type CEM --n_candidate 200 \
#--normalize_flag --deterministic_flag 1 --ensemble_size 1 --n_particles 1 --history_length 10 \
#--future_length 10 --load_path 'data/HALFCHEETAH/NORMALIZED/CaDM/model_free/vanilla' --seed 0  &    #PPO + (Vanilla + CaDM)

#CUDA_VISIBLE_DEVICES=0 python -m run_scripts.model_free.run_ppo_cadm --entropy_coeff 0.0 --lr 0.0005 \
#--num_rollouts 10 --num_steps 200 --num_minibatches 4 --policy_type CEM --n_candidate 200 \
#--normalize_flag --deterministic_flag 0 --ensemble_size 5 --n_particles 20 --history_length 10 \
#--future_length 10 --load_path 'data/HALFCHEETAH/NORMALIZED/CaDM/model_free/pets' --seed 0  &    #PPO + (PE-TS + CaDM)


#CUDA_VISIBLE_DEVICES=0 python -m run_scripts.model_free.run_ppo_cadm --entropy_coeff 0.0 --lr 0.0005 \
#--num_rollouts 10 --num_steps 200 --num_minibatches 4 --policy_type CEM --n_candidate 200 \
#--normalize_flag --deterministic_flag 0 --ensemble_size 5 --n_particles 20 --history_length 10 \
#--future_length 10 --load_path 'data/SLIM_HUMANOID/NORMALIZED/CaDM/DET/CEM/CAND_200/H_10/F_10/BACK_COEFF_0.5/DIFF/BATCH_256/EPOCH_5/hidden_200_lr_0.001_horizon_30_seed_0/checkpoints/params_epoch_9' --seed 0  &    #PPO + (PE-TS + CaDM)