from gym.envs.registration import register

register(
    id='SimKubeEnv-v0',
    entry_point='kube_sim_gym.envs:SimKubeEnv',
    max_episode_steps=None, 
)