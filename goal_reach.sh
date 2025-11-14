python main.py \
  phc/env=env_im_vae_goal \
  phc.learning.params.seed=12 \
  phc.env.num_envs=1  \
  phc.headless=True \
  phc.env.episode_length=900 \
  diffusion_forcing/algorithm=df_humanoid_goal \
  diffusion_forcing.load="output/UniPhys/checkpoints/uniphys_T32.ckpt" \
  diffusion_forcing.algorithm.diffusion.use_ema=False \
  diffusion_forcing.task="interact" \
  +diffusion_forcing.name=play_goal_reaching \
  +phc.env.collect_trajectories=True \
  +phc.env.maxGoalsToCollect=1000 \
  +diffusion_forcing.sequential_goal=False \
  phc.env.tar_min=5 \
  phc.env.tar_max=5