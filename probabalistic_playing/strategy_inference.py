from argparse import Action
from errno import EPIPE
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
    calculate = [Likelihood.cooperator, Likelihood.defector, Likelihood.random, Likelihood.tit_for_tat, 
    Likelihood.grim_trigger, Likelihood.tit_for_two_tats, Likelihood.two_tits_for_tat, 
    Likelihood.gradual, Likelihood.soft_majority, Likelihood.hard_majority, Likelihood.remorseful_prober, 
    Likelihood.soft_grudger, Likelihood.prober]
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
    opponent = Prober(0.1)
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
        for i in range(len(next_action)):
            print(possibilities[i] + " : ", next_action[i](opponent_history, agent_history, unexpected[i] * EPSILON))
        
        round += 1

if __name__ == "__main__":
    main()