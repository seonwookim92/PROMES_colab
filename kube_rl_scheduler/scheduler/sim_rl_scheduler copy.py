from stable_baselines3 import DQN, PPO

import os, sys
base_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(base_path)

class SimRlScheduler:
    def __init__(self, env, model_fname='ppo_1st.zip'):
        self.env = env
        self.model_name = model_fname.split('.')[0]
        print(f"Model file name: {self.model_name}")
        self.model_fpath = os.path.join(base_path, 'kube_rl_scheduler', 'strategies', 'model', self.model_name)
        self.model_type = model_fname.split('_')[0]
        self.model = None

        if self.model_type == 'DQN':
            self.model = DQN.load(self.model_fpath, env=self.env)
        elif self.model_type == 'PPO':
            self.model = PPO.load(self.model_fpath, env=self.env)

    def decision(self, env):
        state = env.get_state()
        action = self.model.predict(state)[0]
        return action