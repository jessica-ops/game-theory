import numpy as np
import random
import sys
import matplotlib.pyplot as plt

sys.path.append('..')
from strategies.strategy import *
from strategies.action_probability import ActionProbability

PAYOFF = [[(10.0, 10.0), (0.0, 20.0)], [(20.0, 0.0), (5.0, 5.0)]]

def load_table(filename):
    """
    Load Q Table from saved location (filename). Return as a dictionary. 
    """
    table = {}
    f = open(filename)
    lines = f.readlines()

    for line in lines: 
        full_state = line[:line.index(":")]
        dic = line[line.index(":") + 2:-2]
        states = full_state.split(", ")
        state = (states[0], states[1])
        entries = dic.split(", ")
        if len(entries) == 1:
            entry_zero = entries[0].split(":")
            actions = {int(entry_zero[0]): float(entry_zero[1])}
        else:
            entry_zero = entries[0].split(":")
            entry_one = entries[1].split(":")
            actions = {int(entry_zero[0]): float(entry_zero[1]), int(entry_one[0]): float(entry_one[1])}
        table[state] = actions
    
    return table

def load_compressed_table(filename):
    """
    NOT BEING USED -- supposed to be used for compressed transition probabilities 
    returned by observe_policies 
    """
    table = {}
    f = open(filename)
    lines = f.readlines()

    for line in lines: 
        if line.index(":") == 0:
            state = ""
        else:
            state = line[:line.index(":")]
        dic = line[line.index(":") + 2:-2]
        entries = dic.split(", ")
        if len(entries) == 1:
            entry_zero = entries[0].split(":")
            actions = {int(entry_zero[0]): float(entry_zero[1])}
        else:
            entry_zero = entries[0].split(":")
            entry_one = entries[1].split(":")
            actions = {int(entry_zero[0]): float(entry_zero[1]), int(entry_one[0]): float(entry_one[1])}
        table[state] = actions
    
    return table

def act_prob_from_Q(Q, next_act_dist, opp_history, agent_history, opp_history_str, agent_history_str, epsilon, temperature, rounds):
    """
    Returns the probablity distirbution of playing 0 or 1 based on Q table + Boltzmann Distribution 
    with certain temperature. 
    """
    
    opp_next_act_dist = next_act_dist(opp_history, agent_history, epsilon)
    values = [[Q[(opp_history_str + "0", agent_history_str)][0], 
    Q[(opp_history_str + "1", agent_history_str)][0]], 
    [Q[(opp_history_str + "0", agent_history_str)][1], 
    Q[(opp_history_str + "1", agent_history_str)][1]]]

    if round != rounds - 1:
        values[0][0] *= opp_next_act_dist[0]
        values[1][0] *= opp_next_act_dist[0]
        values[0][1] *= opp_next_act_dist[1]
        values[1][1] *= opp_next_act_dist[1]
    
    action_value = np.array([sum(values[0]), sum(values[1])])
    norm = sum(np.exp(action_value / temperature))
    act_probs = np.exp(action_value / temperature) / norm

    return act_probs    

def io_play(Q, next_act_dist, epsilon, rounds, temperature, names, Qs, next_act_dists):
    """
    Personally play against a policy optimized against a certain opponent with ActionProbability
    next_act_dist and noise epsilon. 
    See likelihood of strategy which agent was optimized against for all strategies in names, Qs, and next_act_dists.
    """
    while True:
        prior = np.ones(shape = (len(names), )) / 0.5
        likelihood = np.ones(shape = (len(names), ))
        io_history = []
        agent_history = []
        io_history_str = ""
        agent_history_str = "None"
        for round in range(rounds):
            act_probs = act_prob_from_Q(Q, next_act_dist, io_history, agent_history, io_history_str, agent_history_str, epsilon, temperature, rounds)
            prob = random.random()
            if prob <= act_probs[0]:
                my_action = 0
            else:
                my_action = 1
            
            io_action = int(input("Enter an action: "))
            my_reward, io_reward = PAYOFF[my_action][io_action]

            print("my action:", my_action, "strat action:", io_action)
            print("my reward:", my_reward, "strat reward:", io_reward)
            print("")

            for i in range(len(names)):
                print(names[i][0])
                act_lh = act_prob_from_Q(Qs[i], next_act_dists[i], io_history, agent_history, io_history_str, agent_history_str, names[i][1], temperature, rounds)
                print(act_lh)
                likelihood[i] *= act_lh[my_action]

            agent_history.append(my_action)
            io_history.append(io_action)
            io_history_str = io_history_str + str(io_action)
            if agent_history_str == "None":
                agent_history_str = str(my_action)
            else:
                agent_history_str = agent_history_str + str(my_action)
        
        print("\n END OF THIS GAME -- INFERENCES: \n")
        posterior = np.multiply(prior, likelihood) / np.sum(np.multiply(prior, likelihood))
        for i in range(len(posterior)):
            print(names[i][0] + " = " + str(posterior[i]))
        print("")

def agent_vs_strat(Q, epsilon, temperature, opponent, next_act_dist, rounds):
    """
    Have an agent defined by their Q table (Q) and the characteristics they believe of their opponent
    (epsilon, next_act_dist) play against a certain opponent for a certain number of rounds. 
    Return the history of the game (both as list & string for agent & opponent)
    """
    strat_history = []
    agent_history = []
    strat_history_str = ""
    agent_history_str = "None"
    for round in range(rounds):
        act_probs = act_prob_from_Q(Q, next_act_dist, strat_history, agent_history, strat_history_str, agent_history_str, epsilon, temperature, rounds)
        prob = random.random()
        if prob <= act_probs[0]:
            my_action = 0
        else:
            my_action = 1
        
        strat_act = opponent.choose(strat_history, agent_history, round)
        agent_history.append(my_action)
        strat_history.append(strat_act)
        strat_history_str = strat_history_str + str(strat_act)
        if agent_history_str == "None":
            agent_history_str = str(my_action)
        else:
            agent_history_str = agent_history_str + str(my_action)
    
    return agent_history, strat_history, agent_history_str, strat_history_str

def analyze_game(agent_history, strat_history, agent_history_str, strat_history_str, names, potential_Qs, next_act_dists, temperature):
    """
    Given the history of a game from agent_vs_strat, compute likelihood of the policy being one in the 
    lits provided by names, potential_Qs, and next_act_dists. 
    Likelihood is returned in a list where each index matches with the appropriate agent based on the indicies of the inputs. 
    """
    likelihood = np.ones(shape = (len(names), ))
    for round in range(len(agent_history)):
        for i in range(len(names)):
            if round == 0:
                act_lh = act_prob_from_Q(potential_Qs[i], next_act_dists[i], strat_history[:round], agent_history[:round], strat_history_str[:round], "None", names[i][1], temperature, len(agent_history))
            else:
                act_lh = act_prob_from_Q(potential_Qs[i], next_act_dists[i], strat_history[:round], agent_history[:round], strat_history_str[:round], agent_history_str[:round], names[i][1], temperature, len(agent_history))
            likelihood[i] *= act_lh[agent_history[round]]
    
    return likelihood

def main():
    temperature = 2 # temperature I have been using for all agent 
    rounds = 10
    tournaments = 1000

    names = [("Cooperator", 0.2), ("Defector", 0.2), ("Tit for Tat", 0.2), ("Grim Trigger", 0.2),("Tit for Two Tats", 0.2),
    ("Two Tits for Tat", 0.2), ("Gradual", 0.2), ("Soft Majority", 0.2), ("Hard Majority", 0.2), ("Remorseful Prober", 0.2), 
    ("Soft Grudger", 0.2), ("Prober", 0.2)]
    opponents = [Cooperater(0.2), Defector(0.2), TitForTat(0.2), GrimTrigger(0.2), TitForTwoTats(0.2), TwoTitsForTats(0.2), Gradual(0.2), 
    SoftMajority(0.2), HardMajority(0.2), RemorsefulProber(0.2), SoftGrudger(0.2), Prober(0.2)]
    next_act_dists = [ActionProbability.cooperator, ActionProbability.defector,
    ActionProbability.tit_for_tat, ActionProbability.grim_trigger, ActionProbability.tit_for_two_tats, 
    ActionProbability.two_tits_for_tat, ActionProbability.gradual, ActionProbability.soft_majority, ActionProbability.hard_majority,
    ActionProbability.remorseful_prober, ActionProbability.soft_grudger, 
    ActionProbability.prober]

    # load all Q tables
    Qs = []
    for i in range(len(names)):
        folder = "policies/epsilon_" + str(names[i][1]) + "/"
        Qs.append(load_table(folder + names[i][0] + ".txt"))

    posteriors = [[] for i in range(len(names))]
    for t in range(tournaments):
        for i in range(len(names)): # Get the agent 
            prior = np.ones(len(names)) / len(names)
            likelihood = np.ones(len(names))
            Q = Qs[i]
            for j in range(len(names)): # Get the strategy
                # play agent vs strategy
                agent_history, strat_history, agent_history_str, strat_history_str = agent_vs_strat(Q, names[i][1], temperature, opponents[j], next_act_dists[i], rounds)
                # get likelihood from game 
                new_likelihood = analyze_game(agent_history, strat_history, agent_history_str, strat_history_str, names, Qs, next_act_dists, temperature)
                likelihood = np.multiply(likelihood, new_likelihood)
            posterior = np.multiply(prior, likelihood) / np.sum(np.multiply(prior, likelihood))
            posteriors[i].append(posterior)
    
    x = np.linspace(1, len(names) * 3, len(names))
    cmap = plt.cm.get_cmap('tab20', len(names))

    """
    Bar chart for all strategies together
    """
    # plt.figure(figsize=(20,10))
    # #print(posteriors)
    # plt.xticks(x, labels)
    # for i in range(len(names)):
    #     plt.plot(x, np.average(posteriors[i], axis=0), color = cmap(i), linewidth = 5.0)
    # plt.savefig("graphs/total")
    # plt.close()

    """
    Line graphs for all strategies individually 
    """
    # plt.figure(figsize=(20,10))
    for i in range(len(names)):
        plt.plot(x, np.average(posteriors[i], axis=0), color = cmap(i), linewidth = 5.0)
        plt.ylim([0.0, 1.0])
        plt.savefig("graphs/" + names[i][0])
        plt.close()
    
    """
    Line graph for all strategies together
    """
    # plt.figure(figsize=(20,10))
    # for i in range(len(names)):
    #     if names[i][0] == "Cooperator" or names[i][0] == "Defector" or names[i][0] == "Tit for Tat" or names[i][0] == "Tit for Two Tats":
    #         plt.plot(x, np.average(posteriors[i], axis=0), color = cmap(i), linewidth = 5.0)
    # plt.savefig("graphs/four dists")
    # plt.close()

if __name__ == "__main__":
    main()