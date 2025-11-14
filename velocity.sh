python main.py \
  phc/env=env_im_vae_steer \
  phc.learning.params.seed=12 \
  phc.env.num_envs=1  \
  phc.headless=True \
  phc.env.episode_length=3000 \
  phc.no_virtual_display=True \
  diffusion_forcing/algorithm=df_humanoid_steer \
  diffusion_forcing.load="output/UniPhys/checkpoints/uniphys_T32.ckpt" \
  diffusion_forcing.algorithm.diffusion.use_ema=False \
  diffusion_forcing.task="interact" \
  +diffusion_forcing.name=play_steering \
  phc.env.tar_speed_min=0.0 phc.env.tar_speed_max=3.0 \
  phc.env.change_steps_min=200 phc.env.change_steps_max=201 \
  +phc.env.collect_trajectories=True \
  +phc.env.maxEpisodesToCollect=1000 \
  +phc.env.resetSceneOnSpeedChange=True \
  phc.env.cycle_motion=True