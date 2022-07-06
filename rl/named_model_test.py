import numpy as np
from stable_baselines3 import PPO
import gym
import gym.spaces as spaces
import random
from stable_baselines3.common.env_checker import check_env

MODELS = ["memory_model", "diff_rewards", "opponent_agent", "opponent_agent2"]
STRATEGIES = ["c", "d", "r", "tft", "gt", "pav", "tf2t", "2tft", "g", "sm", "hm", "np", "rp", "sg", "p"]
ROUNDS = 100
GAMES = 10
PAYOFF = [[(10, 10), (0, 20)], [(20, 0), (5, 5)]]
FILE = open("/home/jops/game_theory/results/logs/named_results.txt", "w")

def test_model(verbose, extra_verbose):
    FILE.write("-------------------------------------------------------------\n")
    FILE.write("\nTESTING LOADED 'NAMED' MODEL \n")
    model1 = PPO.load("models/strat_name")
    for m in MODELS: 
        model2 = PPO.load("models/" + m)
        model1_acts = np.zeros(shape = (10,), dtype=np.int32)
        model2_acts = np.zeros(shape = (10,), dtype=np.int32)
        model1_tally = 0
        model2_tally = 0
        FILE.write("\nstrat_names vs " + m + "\n")
        for x in range(GAMES):
            if verbose:
                FILE.write("\nGAME #" + str(x) + "\n")
            model1_total = 0
            model2_total = 0
            for y in range(ROUNDS):
                action1, _states = model1.predict({'opponent_action': model2_acts, 'agent_action': model1_acts, 'opponent_name': len(strat_names) + models.index(m)})
                action2, _states = model2.predict({'opponent_action': model1_acts, 'agent_action': model2_acts})
                model1_pay, model2_pay = PAYOFF[action1][action2]
                model1_total += model1_pay
                model2_total += model2_pay
                model1_acts[:-1] = model1_acts[1:]
                model1_acts[-1] = action1 + 1
                model2_acts[:-1] = model2_acts[1:]
                model2_acts[-1] = action2 + 1
                if verbose and extra_verbose:
                    FILE.write(str(action1) + " : " + str(model1_pay) + "    " + str(action2) + " : " + str(model2_pay) + "\n")
            
            if model1_total > model2_total: 
                if verbose:
                    FILE.write("strat_names won this game with total " + str(model1_total) + " vs " + str(model2_total) + "\n")
                model1_tally += 1
            elif model2_total > model1_total: 
                if verbose: 
                    FILE.write(m + " won this game with total " + str(model2_total) + " vs " + str(model1_total) + "\n")
                model2_tally += 1
            else:
                if verbose: 
                    FILE.write("there has been a tie! with total " + str(model1_total) + "\n")

        if model1_tally > model2_tally: 
            FILE.write("strat_name VICTORIOUS OVER " + m + "\n")
        elif model2_tally > model1_tally: 
            FILE.write(m + " VICTORIOUS OVER strat_name \n")
        else: 
            FILE.write("TIE BETWEEN strat_name AND " + m + "\n")

class NamedEnvTest(gym.Env):
    def __init__(self):
        self.round_num = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({'opponent_action': spaces.MultiDiscrete([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), 
        'agent_action': spaces.MultiDiscrete([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), 'opponent_name': spaces.Discrete(len(STRATEGIES) + len(MODELS))})

    def step(self, action):
        opponent_action, _states = self.opponent_model.predict({'opponent_action': self.agent_obvs, 'agent_action': self.opponent_obvs})
        agent_reward, opponent_reward  = PAYOFF[action][opponent_action]
        self.agent_reward_tot+=agent_reward
        self.opponnent_reward_tot+=opponent_reward

        self.agent_obvs[:-1] = self.agent_obvs[1:]
        self.agent_obvs[-1] = action + 1
        self.opponent_obvs[:-1] = self.opponent_obvs[1:]
        self.opponent_obvs[-1] = opponent_action + 1
        self.round_num+=1

        if self.round_num == ROUNDS:
            terminal = True
            if self.agent_reward_tot > self.opponnent_reward_tot:
                FILE.write("agent beat " + self.s + " with reward " + str(self.agent_reward_tot) + " vs " + str(self.opponnent_reward_tot) + "\n")
            elif self.opponnent_reward_tot > self.agent_reward_tot: 
                FILE.write("agent lost to  " + self.s + " with reward " + str(self.agent_reward_tot) + " vs " + str(self.opponnent_reward_tot) + "\n")
        else:
            terminal = False
        
        return {'opponent_action': self.opponent_obvs, 'agent_action': self.agent_obvs, 'opponent_name': len(STRATEGIES) + MODELS.index(self.s)}, agent_reward, terminal, {}

    def reset(self):
        self.agent_reward_tot = 0
        self.opponnent_reward_tot = 0
        self.round_num = 0
        self.random_model()
        self.opponent_obvs = np.zeros(shape = (10,), dtype=np.int32)
        self.agent_obvs = np.zeros(shape = (10,), dtype=np.int32)

        return {'opponent_action': self.opponent_obvs, 'agent_action': self.agent_obvs, 'opponent_name': len(STRATEGIES) + MODELS.index(self.s)}

    def random_model(self):
        self.s = random.choice(MODELS)
        self.opponent_model = PPO.load("/home/jops/game_theory/results/models/" + self.s)

def main():
    # test_model(False)
    model = PPO.load("/home/jops/game_theory/results/models/strat_name")
    env = NamedEnvTest()
    check_env(env)
    model.set_env(env)
    model.learn(total_timesteps=4000000)

if __name__ == "__main__":
    main()