import sys
sys.path.append('..')
from strategies.strategy import *
from strategies.likelihood import Likelihood
from strategies.action_probability import ActionProbability
import numpy as np

EPSILON = 0.01

"""
Allows user to play against one of the strategies. 
Prints the likelihood of each strategy and probability distribution over their next move. 
"""
def main():
    # all strategies
    possibilities = ["Cooperator", "Defector", "Random", "Tit for Tat", "Grim Trigger", "Tit for Two Tats", 
    "Two Tits for Tat", "Gradual", "Soft Majority", "Hard Majority", "Remoreseful Prober", "Soft Grudger", 
    "Prober"]

    # functions for computing likelihood and probability of next action 
    lh = Likelihood(EPSILON)
    calculate = [lh.cooperator, lh.defector, lh.random, lh.tit_for_tat, 
    lh.grim_trigger, lh.tit_for_two_tats, lh.two_tits_for_tat, 
    lh.gradual, lh.soft_majority, lh.hard_majority, lh.remorseful_prober, 
    lh.soft_grudger, lh.prober]
    next_action = [ActionProbability.cooperator, ActionProbability.defector, ActionProbability.random, 
    ActionProbability.tit_for_tat, ActionProbability.grim_trigger, ActionProbability.tit_for_two_tats, 
    ActionProbability.two_tits_for_tat, ActionProbability.gradual, ActionProbability.soft_majority, 
    ActionProbability.hard_majority, ActionProbability.remorseful_prober, ActionProbability.soft_grudger, 
    ActionProbability.prober]

    prior = np.ones((len(possibilities), )) / len(possibilities) # flat prior
    likelihood = np.ones((len(possibilities))) # all strategies start with likelihood = 1
    unexpected = np.zeros(len(possibilities)) # keeps track of number of unexpected moves

    agent_history = []
    opponent_history = []
    round = 0

    # set the opponent here 
    opponent = GrimTrigger(0)
    while True: 
        print("----------------------------------------")
        print("Round #" + str(round))
        action = input("Enter action: ")
        action = int(action)
        opponent_action = opponent.choose(opponent_history, agent_history, round)
        print("opponent action:", opponent_action)
        print("")

        opponent_history.append(opponent_action)
        agent_history.append(action)
        
        # calculate likelihood for all strategies 
        for i in range(len(likelihood)):
            likelihood[i], unexpected[i] = calculate[i](opponent_history, agent_history, likelihood[i], unexpected[i])
        
        # calculate the marginal
        marginal = 0
        for i in range(len(likelihood)):
            marginal += likelihood[i] * prior[i]
        
        # calculate the posterior 
        posterior = np.multiply(likelihood, prior) / marginal

        # output results
        for i in range(len(possibilities)):
            print(possibilities[i] + " ----  posterior: ", posterior[i], "  unexpected: ", unexpected[i] * EPSILON)
        print("")
        # for i in range(len(next_action)):
        #     print(possibilities[i] + " : ", next_action[i](opponent_history, agent_history, unexpected[i] * EPSILON))
        
        round += 1

if __name__ == "__main__":
    main()