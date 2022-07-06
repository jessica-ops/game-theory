from queue import Full
import numpy as np
import random

from strategies.strategy import *
from strategies.likelihood import Likelihood
from strategies.action_probability import ActionProbability
from strategies.full_likelihood import FullLikelihood

EPSILON = 0.01
FILE = open("/home/jops/game_theory/results/logs/bayesian_agent_results.txt", "w")
STRATS = [Cooperater(0), Defector(0), Random([0.5, 0.5]), TitForTat(0), GrimTrigger(0), TitForTwoTats(0), TwoTitsForTats(0), Gradual(0), 
SoftMajority(0), HardMajority(0), RemorsefulProber(0), SoftGrudger(0), Prober(0)]

class Agent:
    def __init__(self, rounds, exploration):
        self.possibilities = ["Cooperator", "Defector", "Random", "Tit for Tat", "Grim Trigger", "Tit for Two Tats", 
        "Two Tits for Tat", "Gradual", "Soft Majority", "Hard Majority", "Remorseful Prober", "Soft Grudger", "Prober"]
        lh = Likelihood(EPSILON)
        self.calculate = [lh.cooperator, lh.defector, lh.random, lh.tit_for_tat, 
        lh.grim_trigger, lh.tit_for_two_tats, lh.two_tits_for_tat, 
        lh.gradual, lh.soft_majority, lh.hard_majority, lh.remorseful_prober, lh.soft_grudger, 
        lh.prober]
        self.next_action = [ActionProbability.cooperator, ActionProbability.defector, ActionProbability.random, 
        ActionProbability.tit_for_tat, ActionProbability.grim_trigger, ActionProbability.tit_for_two_tats, 
        ActionProbability.two_tits_for_tat, ActionProbability.gradual, ActionProbability.soft_majority, 
        ActionProbability.hard_majority, ActionProbability.remorseful_prober, ActionProbability.soft_grudger, 
        ActionProbability.prober]
        self.backwards_lh = [FullLikelihood.cooperator, FullLikelihood.defector, FullLikelihood.random, 
        FullLikelihood.tit_for_tat, FullLikelihood.grim_trigger, FullLikelihood.tit_for_two_tats, 
        FullLikelihood.two_tits_for_tat, FullLikelihood.gradual, FullLikelihood.soft_majority, 
        FullLikelihood.hard_majority, FullLikelihood.remorseful_prober, FullLikelihood.soft_grudger, 
        FullLikelihood.prober]
        self.payoff = [[(10, 10), (0, 20)], [(20, 0), (5, 5)]]
        self.rounds = rounds
        self.exploration = exploration

    def reset(self):
        self.agent_history = []
        self.opponent_history = []
        self.prior = np.ones((len(self.possibilities), )) / len(self.possibilities)
        self.posterior = np.ones((len(self.possibilities), )) / len(self.possibilities)
        self.likelihood = np.ones((len(self.possibilities)))
        self.unexpected = np.zeros(len(self.possibilities))

    def game(self, opponent_idx):
        FILE.write("\n------------------------------------------------\n")
        FILE.write("Game against " + self.possibilities[opponent_idx] + "\n\n")
        opponent = STRATS[opponent_idx]
        for r in range(self.rounds):
            action = self.choose()
            opponent_action = opponent.choose(self.opponent_history, self.agent_history, r)
            FILE.write("Round #" + str(r) + " --- agent action: " + str(action) + "  strat action: " + str(opponent_action) + "\n")
            self.agent_history.append(action)
            self.opponent_history.append(opponent_action)
            for i in range(len(self.likelihood)):
                self.likelihood[i], self.unexpected[i] = self.calculate[i](self.opponent_history, self.agent_history, self.likelihood[i], self.unexpected[i])
            marginal = 0
            for i in range(len(self.likelihood)):
                marginal += self.likelihood[i] * self.prior[i]
            self.posterior = np.multiply(self.likelihood, self.prior) / marginal
            max_lhs_idx = np.where(self.posterior==np.max(self.posterior))
            for i in max_lhs_idx[0]:
                FILE.write("Strat guess: " + self.possibilities[i] + "   likelihood: " + str(self.posterior[i]) + "\n")

    def choose(self):
        action_probabilities = np.zeros((len(self.possibilities), 2))
        for i in range(len(self.possibilities)):
            action_probabilities[i] = self.next_action[i](self.opponent_history, self.agent_history, self.unexpected[i] * EPSILON)
        
        max_lhs_idx = np.where(self.posterior==np.max(self.posterior))
        if np.size(max_lhs_idx[0]) > 1 and random.random() <= self.exploration:
            zero_acts = []
            one_acts = []
            agent_history_0 = self.agent_history + [0]
            agent_history_1 = self.agent_history + [1]
            for i in max_lhs_idx[0]:
                # print("action probs ", action_probabilities[i], " at", i)
                # print("action: ", np.where(action_probabilities[i] == np.max(action_probabilities[i])))
                opponent_action = np.where(action_probabilities[i] == np.max(action_probabilities[i]))[0][0]
                next_0 = self.next_action[i](self.opponent_history + [opponent_action], agent_history_0, self.unexpected[i] * EPSILON)
                next_1 = self.next_action[i](self.opponent_history + [opponent_action], agent_history_1, self.unexpected[i] * EPSILON)
                zero_acts.append(np.where(next_0 == np.max(next_0))[0][0])
                one_acts.append(np.where(next_1 == np.max(next_1))[0][0])
            diff_0 = np.sum(zero_acts) * (np.size(zero_acts) - np.sum(zero_acts))
            diff_1 = np.sum(one_acts) * (np.size(one_acts) - np.sum(one_acts))
            if diff_0 > diff_1:
                FILE.write("\nChoosing to explore with action 0\n")
                return 0
            elif diff_1 > diff_0:
                FILE.write("\nChoosing to explore with action 1\n")
                return 1

        zero_util = 0
        for i in range(len(self.possibilities)):
            zero_util += (self.payoff[0][0][0] * action_probabilities[i][0] + self.payoff[0][1][0] * action_probabilities[i][1]) * self.posterior[i]
        one_util = 0
        for i in range(len(self.possibilities)):
            one_util += (self.payoff[1][0][0] * action_probabilities[i][0] + self.payoff[1][1][0] * action_probabilities[i][1]) * self.posterior[i]
        FILE.write("\nChoosing action... zero util = " + str(zero_util) + ", one util = " + str(one_util) + "\n")
        if one_util > zero_util:
            return 1
        return 0
    
    def compute_strategy(self, strat_idx):
        opt = np.empty(shape = (self.rounds, 2, 2))
        opt[self.rounds - 1] = [[10, 0], [20, 5]]
        actions = [[[[0], [0]], [[0], [1]]], [[[1], [0]], [[1], [1]]]]
        for r in range(self.rounds - 2, -1, -1):
            print("---------------------------------------------")
            print("round #" + str(r))
            new_actions = [[[[], []], [[], []]], [[[], []], [[], []]]]
            for my_act in range(2):
                for opp_act in range(2):
                    print("\nanalyzing potential move...")
                    print("my action is " + str(my_act) + " and opponent's action is " + str(opp_act))
                    max = -1
                    max_reward = 0
                    opt_my_acts = []
                    opt_opp_acts = []
                    for my_next_act in range(2):
                        for opp_next_act in range(2):
                            reward = opt[r + 1][my_next_act][opp_next_act] + self.payoff[my_act][opp_act][0]
                            my_acts = [my_act] + actions[my_next_act][opp_next_act][0]
                            opp_acts = [opp_act] + actions[my_next_act][opp_next_act][1]
                            lh = self.backwards_lh[strat_idx](opp_acts, my_acts, 0, r)
                            print("\npotential sequence of my next actions: ", my_acts)
                            print("potential sequence of opponent next actions: ", opp_acts)
                            print("with reward:", reward)
                            print("likelihood of this occuring is:", lh)
                            print("lh * reward = ", lh * reward)
                            if (lh * reward) > max:
                                print("found new max!")
                                max = lh * reward
                                max_reward = reward
                                opt_my_acts = my_acts
                                opt_opp_acts = opp_acts
                    
                    opt[r][my_act][opp_act] = max
                    new_actions[my_act][opp_act] = [opt_my_acts, opt_opp_acts]
            actions = new_actions
        max_idx = np.where(opt[0] == np.max(opt[0]))
        print("\n---------------------------------------")
        print("RESULT:")
        print("optimal reward from round 0:", opt[0][max_idx])
        print("optimal agent actions:", actions[max_idx[0][0]][max_idx[1][0]][0])
        print("optimal opponent actions:", actions[max_idx[0][0]][max_idx[1][0]][1])

def main():
    bayes_agent = Agent(10, 0.6)
    for i in range(len(STRATS)):
        bayes_agent.reset()
        bayes_agent.game(i)
    # bayes_agent.compute_strategy(3)

if __name__ == "__main__":
    main()