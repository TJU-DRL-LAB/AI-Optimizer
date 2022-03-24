# examplary commands to run mujoco tasks with the algorithms



# 1. Evaluation
python mujoco_run_vdfp.py --env=HalfCheetah-v1 --seq_length=64 --min_seq_length=16 --kl_weight=1000 --clip_value=0.2 --seed=111 > ./run_logs/HalfCheetah_vdfp_seqlength_64_16_kl_1000_c_02_s111.log
python mujoco_run_vdfp.py --env=Walker2d-v1 --seq_length=256 --min_seq_length=0 --kl_weight=1000 --clip_value=0.2 --seed=111 > ./run_logs/Walker2d_vdfp_seqlength_256_0_kl_1000_c_02_s111.log
python mujoco_run_vdfp.py --env=InvertedDoublePendulum-v1 --seq_length=256 --min_seq_length=0 --kl_weight=1000 --clip_value=0.2 --seed=111 > ./run_logs/InvertedDoublePendulum_vdfp_seqlength_256_0_kl_1000_c_02_s111.log

python mujoco_run_vdfp_ppo.py --env=Ant-v1 --ti=2 --seq_length=256 --min_seq_length=0 --lamb=0 --kl_weight=1000 --clip_value=0.2 --seed=111 > ./run_logs/Ant_vdfp_ppo_seqlength_256_0_kl_1000_c_02_ti2_s111.log
python mujoco_run_vdfp_ppo.py --env=Hopper-v1 --ti=10 --seq_length=256 --min_seq_length=0 --lamb=0 --kl_weight=1000 --clip_value=0.2 --seed=111 > ./run_logs/Hopper_vdfp_ppo_seqlength_256_0_kl_1000_c_02_ti110_s111.log

python ./evaluation/mujoco_run_ddpg.py --env=HalfCheetah-v1 --seed=111 > ./run_logs/HalfCheetah_ddpg_s111.log
python ./evaluation/mujoco_run_ddpg.py --env=Walker2d-v1 --seed=111 > ./run_logs/Walker2d_ddpg_s111.log

python ./evaluation/mujoco_run_ddsr.py --env=HalfCheetah-v1 --seed=111 > ./run_logs/HalfCheetah_ddsr_s111.log
python ./evaluation/mujoco_run_ddsr.py --env=Walker2d-v1 --seed=111 > ./run_logs/Walker2d_ddsr_s111.log

python ./evaluation/mujoco_run_ppo.py --env=HalfCheetah-v1 --ti=2 --lamb=0.95 --seed=111 > ./run_logs/HalfCheetah_ppo_ti_2_lambda_95_s111.log
python ./evaluation/mujoco_run_ppo.py --env=Walker2d-v1 --ti=5 --lamb=0.95 --seed=111 > ./run_logs/Walker2d_ppo_ti_5_lambda_95_s111.log

python ./evaluation/mujoco_run_a2c.py --env=HalfCheetah-v1 --ti=2 --lamb=0.95 --seed=111 > ./run_logs/HalfCheetah_a2c_ti_2_lambda_95_s111.log
python ./evaluation/mujoco_run_a2c.py --env=Walker2d-v1 --ti=5 --lamb=0.95 --seed=111 > ./run_logs/Walker2d_a2c_ti_5_lambda_95_s111.log



# 2. Ablation
python ./ablation/mujoco_run_vdfp_mlp.py --env=HalfCheetah-v1 --seq_length=64 --min_seq_length=16 --seed=111 > ./run_logs/HalfCheetah_vdfp_mlp_seqlength_64_16_s111.log
python ./ablation/mujoco_run_vdfp_lstm.py --env=HalfCheetah-v1 --seq_length=64 --min_seq_length=16 --kl_weight=1000 --clip_value=0.2 --seed=111 > ./run_logs/HalfCheetah_vdfp_lstm_seqlength_64_16_kl_1000_c_02_s111.log
python ./ablation/mujoco_run_vdfp_concat.py --env=HalfCheetah-v1 --seq_length=64 --min_seq_length=16 --kl_weight=1000 --clip_value=0.2 --seed=111 > ./run_logs/HalfCheetah_vdfp_concat_seqlength_64_16_kl_1000_c_02_s111.log
python ./ablation/mujoco_run_vdfp_relu.py --env=HalfCheetah-v1 --seq_length=64 --min_seq_length=16 --slope=0.8 --kl_weight=1000 --clip_value=0.2 --seed=111 > ./run_logs/HalfCheetah_vdfp_relu_seqlength_64_16_slope_08_kl_1000_c_02_s111.log
python ./ablation/mujoco_run_vdfp_icnn.py --env=HalfCheetah-v1 --seq_length=64 --min_seq_length=16 --icnn-type=0 --kl_weight=1000 --clip_value=0.2 --seed=111 > ./run_logs/HalfCheetah_vdfp_icnn_seqlength_64_16_icnn_type0_kl_1000_c_02_s111.log

python mujoco_run_vdfp.py --env=HalfCheetah-v1 --seq_length=64 --min_seq_length=16 --kl_weight=100 --clip_value=0.2 --seed=111 > ./run_logs/HalfCheetah_vdfp_seqlength_64_16_kl_100_c_02_s111.log
python mujoco_run_vdfp.py --env=HalfCheetah-v1 --seq_length=64 --min_seq_length=16 --kl_weight=10 --clip_value=0.2 --seed=111 > ./run_logs/HalfCheetah_vdfp_seqlength_64_16_kl_10_c_02_s111.log

python mujoco_run_vdfp.py --env=HalfCheetah-v1 --seq_length=64 --min_seq_length=16 --kl_weight=1000 --clip_value=0 --seed=111 > ./run_logs/HalfCheetah_vdfp_seqlength_64_16_kl_1000_c_0_s111.log
python mujoco_run_vdfp.py --env=HalfCheetah-v1 --seq_length=64 --min_seq_length=16 --kl_weight=1000 --clip_value=1 --seed=111 > ./run_logs/HalfCheetah_vdfp_seqlength_64_16_kl_1000_c_1_s111.log
python mujoco_run_vdfp.py --env=HalfCheetah-v1 --seq_length=64 --min_seq_length=16 --kl_weight=1000 --clip_value=2 --seed=111 > ./run_logs/HalfCheetah_vdfp_seqlength_64_16_kl_1000_c_2_s111.log



# 3. Delay Reward

## Setting1
python ./delay_reward/mujoco_run_vdfp_delay.py --delay_type=1 --delay_step=16 --env=HalfCheetah-v1 --seq_length=64 --min_seq_length=16 --kl_weight=1000 --clip_value=0.2 --seed=111 > ./run_logs/HalfCheetah_delay1_16_vdfp_seqlength_64_16_kl_1000_c_02_s111.log
python ./delay_reward/mujoco_run_vdfp_delay.py --delay_type=1 --delay_step=64 --env=Walker2d-v1 --seq_length=256 --min_seq_length=0 --kl_weight=1000 --clip_value=0.2 --seed=111 > ./run_logs/Walker2d_delay1_64_vdfp_seqlength_256_0_kl_1000_c_02_s111.log

python ./delay_reward/mujoco_run_ddpg_delay.py --delay_type=1 --delay_step=16 --env=HalfCheetah-v1 --seed=111 > ./run_logs/HalfCheetah_delay1_16_ddpg_s111.log
python ./delay_reward/mujoco_run_ddsr_delay.py --delay_type=1 --delay_step=32 --env=Walker2d-v1 --seed=111 > ./run_logs/Walker2d_delay1_32_ddsr_s111.log
python ./delay_reward/mujoco_run_ppo_delay.py --delay_type=1 --delay_step=64 --env=HalfCheetah-v1 --ti=2 --lamb=0.95 --seed=111 > ./run_logs/HalfCheetah_delay1_64_ppo_ti_2_lambda_95_s111.log
python ./delay_reward/mujoco_run_a2c_delay.py --delay_type=1 --delay_step=128 --env=Walker2d-v1 --ti=5 --lamb=0.95 --seed=111 > ./run_logs/Walker2d_delay1_128_a2c_ti_5_lambda_95_s111.log

## Setting2
python ./delay_reward/mujoco_run_vdfp_delay.py --delay_type=2 --delay_step=32 --env=HalfCheetah-v1 --seq_length=256 --min_seq_length=0 --kl_weight=1000 --clip_value=0.2 --seed=111 > ./run_logs/HalfCheetah_delay2_32_vdfp_seqlength_256_0_kl_1000_c_02_s111.log
python ./delay_reward/mujoco_run_vdfp_delay.py --delay_type=2 --delay_step=128 --env=Walker2d-v1 --seq_length=256 --min_seq_length=0 --kl_weight=1000 --clip_value=0.2 --seed=111 > ./run_logs/Walker2d_delay2_128_vdfp_seqlength_256_0_kl_1000_c_02_s111.log

python ./delay_reward/mujoco_run_ddpg_delay.py --delay_type=2 --delay_step=16 --env=HalfCheetah-v1 --seed=111 > ./run_logs/HalfCheetah_delay2_16_ddpg_s111.log
python ./delay_reward/mujoco_run_ddsr_delay.py --delay_type=2 --delay_step=32 --env=Walker2d-v1 --seed=111 > ./run_logs/Walker2d_delay2_32_ddsr_s111.log
python ./delay_reward/mujoco_run_ppo_delay.py --delay_type=2 --delay_step=64 --env=HalfCheetah-v1 --ti=2 --lamb=0.95 --seed=111 > ./run_logs/HalfCheetah_delay2_64_ppo_ti_2_lambda_95_s111.log
python ./delay_reward/mujoco_run_a2c_delay.py --delay_type=2 --delay_step=128 --env=Walker2d-v1 --ti=5 --lamb=0.95 --seed=111 > ./run_logs/Walker2d_delay2_128_a2c_ti_5_lambda_95_s111.log

