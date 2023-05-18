from gym.envs.registration import register

register(
    id='RealKubeEnv-v0',
    entry_point='kube_gym.envs:RealKubeEnv',
    max_episode_steps=None, 
)