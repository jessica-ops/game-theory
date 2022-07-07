import gym
import gym.spaces as spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import numpy as np
import sys
sys.path.append('..')
from strategies.strategy import *
import random

PAYOFF = [[(10, 10), (0, 20)], [(20, 0), (5, 5)]]
STRATEGIES = ["c", "d", "r", "tft", "gt", "tf2t", "2tft", "grad", "sm", "hm", "rp", "sg", "p"]
MODELS = ["memory_model", "opponent_agent", "diff_rewards", "opponent_agent2"]
ROUNDS = 30

class GameEnv(gym.Env):
    def __init__(self, opponents, play_model):
        self.opponents = opponents # list of opponents
        self.play_model = play_model # playing against fixed strategy vs trained model
    
        # Action space 
        self.action_space = spaces.Discrete(2)
        # Observation space 
        self.observation_space = spaces.Dict({'opponent_action': spaces.MultiDiscrete([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), 
        'agent_action': spaces.MultiDiscrete([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), 'opponent_defect_avg': spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32), 
        'agent_defect_avg': spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32), 'round_num': spaces.Box(low=0, high=ROUNDS, shape=(1,), dtype=np.int32)})

        # Initialize round num and episode num
        self.round_num = 0
        self.episode_num = 0

        #logging info
        self.average_rewards = []
        self.strategies = []
        self.actions = []

        self.info = {}

    def step(self, action):
        """
        Given an action, play one rounds of the game 
        """
        self.round_num+=1

        if self.play_model: # opponent action (model)
            opponent_action, _states = self.opponent_model.predict({'opponent_action': self.agent_obvs, 'agent_action': self.opponent_obvs})
        else: # opponent action (strategy)
            opponent_action = self.opponent.choose(self.opponent_history, self.agent_history, self.round_num)

        # payoff
        agent_reward, opponent_reward  = PAYOFF[action][opponent_action]

        # update full history
        self.opponent_history[self.round_num - 1] = opponent_action
        self.agent_history[self.round_num - 1] = action

        # update observations (last 10 rounds)
        self.agent_obvs[:-1] = self.agent_obvs[1:]
        self.agent_obvs[-1] = action + 1
        self.opponent_obvs[:-1] = self.opponent_obvs[1:]
        self.opponent_obvs[-1] = opponent_action + 1

        # update defect averages
        opponent_defect_avg = np.sum(self.opponent_history) / self.round_num
        agent_defect_avg = np.sum(self.agent_history) / self.round_num

        # update round info
        self.reward+=agent_reward
        self.actions.append(action)

        if self.round_num == ROUNDS: # terminal state 
            terminal = True
            self.average_rewards.append(self.reward / float(ROUNDS))
            self.strategies.append(self.s)
        else: # non terminal
            terminal = False

        return {'opponent_action': self.opponent_obvs, 'agent_action': self.agent_obvs, 'opponent_defect_avg': np.array([opponent_defect_avg]), 
        'agent_defect_avg': np.array([agent_defect_avg]), 'round_num': np.array(self.round_num)}, agent_reward, terminal, self.info

    def reset(self):
        """
        Reset after / before an episode
        """
        # pick new opponent 
        if self.play_model:
            self.random_model()
        else: 
            self.random_strategy()

        # reset episode info 
        self.round_num = 0
        self.reward = 0
        self.agent_history = np.empty(shape = (ROUNDS), dtype=np.int32)
        self.opponent_history = np.empty(shape = (ROUNDS,), dtype=np.int32)
        self.opponent_obvs = np.zeros(shape = (10,), dtype=np.int32)
        self.agent_obvs = np.zeros(shape = (10,), dtype=np.int32)

        return {'opponent_action': self.opponent_obvs, 'agent_action': self.agent_obvs, 'opponent_defect_avg': np.array([0.]), 'agent_defect_avg': np.array([0.]), 'round_num': np.array([0])}

    def random_model(self):
        """
        Choose & load a random model from list of opponents 
        """
        self.s = random.choice(self.opponents)
        self.opponent_model = PPO.load("/home/jops/game_theory/results/models/" + self.s)

    def random_strategy(self):
        """
        Choose & initialize a random strategy from list of opponents 
        """
        self.s = random.choice(self.opponents)
        if self.s == "c": 
            self.opponent = Cooperater(0)
        elif self.s == "d":
            self.opponent = Defector(0)
        elif self.s == "r":
            self.opponent = Random([0.5, 0.5])
        elif self.s == "tft":
            self.opponent = TitForTat(0)
        elif self.s == "gt":
            self.opponent = GrimTrigger(0)
        elif self.s == "tf2t":
            self.opponent = TitForTwoTats(0)
        elif self.s == "2tft":
            self.opponent = TwoTitsForTats(0)
        elif self.s == "grad":
            self.opponent = Gradual(0)
        elif self.s == "sm":
            self.opponent = SoftMajority(0)
        elif self.s == "hm":
            self.opponent = HardMajority(0)
        elif self.s == "rp":
            self.opponent = RemorsefulProber(0)
        elif self.s == "sg":
            self.opponent = SoftGrudger(0)
        elif self.s == "p":
            self.opponent = Prober(0)
    
    def log(self, logdir):
        """
        Log rewards, actions, and strategies 
        """
        np.savetxt(logdir + "average_rewards.npy", self.average_rewards)
        np.savetxt(logdir + "actions.npy", self.actions)
        f = open(logdir + "strategies.txt", "w")
        for strat in self.strategies:
            f.write(strat + '\n')
        f.close()

if __name__ == "__main__":
    """
    For training polciies on many strategies/models
    """
    # env = GameEnv(MODELS, True)
    # model = PPO('MultiInputPolicy', env, verbose=1).learn(total_timesteps=1000000)
    # model.save("/home/jops/game_theory/results/models/opponent_agent")
    # env.log("/home/jops/game_theory/results/logs/opponent_agent")

    """
    For training policies on individual strategies 
    """
    for strat in STRATEGIES:
        print("strat: " + strat)
        env = GameEnv([strat], False)
        model = PPO('MultiInputPolicy', env, verbose=1).learn(total_timesteps=100000)
        model.save("/home/jops/game_theory/results/models/short_rounds/" + strat)
        env.log("/home/jops/game_theory/results/logs/short_rounds/" + strat + "/")