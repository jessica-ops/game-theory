import gym
import gym.spaces as spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import numpy as np

RESULTS = [[(4, 4), (0, 5)], [(5, 0), (2, 2)]]
STRATEGIES = ["c", "d", "r", "tft", "gt", "pav", "tf2t", "2tft", "g", "sm", "hm", "np", "rp", "sg", "p"]
MODELS = ["memory_model", "opponent_agent", "diff_rewards", "opponent_agent2"]

class GamePlayEnv(gym.Env):
    def __init__(self):
        # agent's action
        self.action_space = spaces.Discrete(2)
        # opponent's action 
        # self.observation_space = spaces.Dict({'opponent_action': spaces.MultiDiscrete([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), 
        # 'agent_action': spaces.MultiDiscrete([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), 'opponent_name': spaces.Discrete(len(STRATEGIES) + len(MODELS))})
        self.observation_space = spaces.Dict({'opponent_action': spaces.MultiDiscrete([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), 
        'agent_action': spaces.MultiDiscrete([3, 3, 3, 3, 3, 3, 3, 3, 3, 3])})
        self.agent_reward = 0
        self.opponent_reward = 0

    def step(self, action):
        prompt = input("Enter Action:\n")
        if prompt == 'Total':
            print("Agent's total reward:", self.agent_reward)
            print("Your total reward:", self.opponent_reward)
            prompt = input("Enter Action:\n")
        opponent_action = int(prompt)
        print("Agent's action was:", action)
        agent_reward, opponent_reward  = RESULTS[action][opponent_action]
        print("your reward, agent reward:", opponent_reward, ",", agent_reward, "\n")

        self.agent_obvs[:-1] = self.agent_obvs[1:]
        self.agent_obvs[-1] = action + 1
        self.opponent_obvs[:-1] = self.opponent_obvs[1:]
        self.opponent_obvs[-1] = opponent_action + 1
        self.agent_reward+=agent_reward
        self.opponent_reward+=opponent_reward
        
        # return {'opponent_action': self.opponent_obvs, 'agent_action': self.agent_obvs, 'opponent_name': len(STRATEGIES)}, agent_reward, False, {}
        return {'opponent_action': self.opponent_obvs, 'agent_action': self.agent_obvs}, agent_reward, False, {}

    def reset(self):
        self.opponent_obvs = np.zeros(shape = (10,), dtype=np.int32)
        self.agent_obvs = np.zeros(shape = (10,), dtype=np.int32)

        # return {'opponent_action': self.opponent_obvs, 'agent_action': self.agent_obvs, 'opponent_name': len(STRATEGIES)}
        return {'opponent_action': self.opponent_obvs, 'agent_action': self.agent_obvs}

def main():
    env = GamePlayEnv()
    # check_env(env)
    model = PPO.load("/home/jops/game_theory/results/models/individual_policies/gt")
    obs = env.reset()
    print("loaded env, model...")
    tot = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        tot += rewards
    
    print("result:", tot)
    

if __name__ == "__main__":
    main()