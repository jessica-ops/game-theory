import io
import numpy as np
import math
import random
import sys
sys.path.append('..')
from strategies.action_probability import ActionProbability

PAYOFF = [[(10.0, 10.0), (0.0, 20.0)], [(20.0, 0.0), (5.0, 5.0)]]

def load_table(filename):
    table = {}
    f = open(filename)
    lines = f.readlines()

    for line in lines: 
        #print(line)
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
    table = {}
    f = open(filename)
    lines = f.readlines()

    for line in lines: 
        #print(line)
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

def io_play(Q, next_act_dist, epsilon, rounds, choice_param, names, distributions):
    prior = np.ones(shape = (len(distributions), )) / 0.5
    posterior = np.zeros(shape = (len(distributions), ))
    while True:
        io_history = []
        agent_history = []
        io_history_str = ""
        agent_history_str = "None"
        for round in range(rounds):
            opp_next_act_dist = next_act_dist(io_history, agent_history, epsilon)
            values = [[Q[(io_history_str + "0", agent_history_str)][0], 
            Q[(io_history_str + "1", agent_history_str)][0]], 
            [Q[(io_history_str + "0", agent_history_str)][1], 
            Q[(io_history_str + "1", agent_history_str)][1]]]
            # print("Q values:", values)
            if round != rounds - 1:
                values[0][0] *= opp_next_act_dist[0]
                values[1][0] *= opp_next_act_dist[0]
                values[0][1] *= opp_next_act_dist[1]
                values[1][1] *= opp_next_act_dist[1]
            # print("expected q values:", values)
            action_value = [sum(values[0]), sum(values[1])]
            # print("action values:", action_value)
            my_action = np.argmax(action_value)
            threshold = math.pow(choice_param, abs(action_value[0] - action_value[1])) / 2.0
            prob = random.random()
            if prob <= threshold:
                my_action = (my_action + 1) % 2
            
            io_action = int(input("Enter an action: "))
            my_reward, io_reward = PAYOFF[my_action][io_action]

            print("my action:", my_action, "strat action:", io_action)
            print("my reward:", my_reward, "strat reward:", io_reward)
            print("")
            print("P(agent action =" + str(my_action) + " | "+ io_history_str + ", strat = _____)")
            for i in range(len(names)):
                if io_history_str in distributions[i]:
                    if my_action in distributions[i][io_history_str]:
                        # print(distributions[i][io_history_str])
                        print(names[i] + " =", distributions[i][io_history_str][my_action])
                    else: 
                        print(names[i] + " = 0.0")
                else:
                    print(names[i] + " has not been observed with (" + io_history_str + ")")
            print("")
            agent_history.append(my_action)
            io_history.append(io_action)
            io_history_str = io_history_str + str(io_action)
            if agent_history_str == "None":
                agent_history_str = str(my_action)
            else:
                agent_history_str = agent_history_str + str(my_action)
        
        print("\n END OF THIS GAME \n")

def main():
    Q = load_table("policies/epsilon_0.2/Tit for Tat.txt")
    #print(Q)
    names = ["Cooperator 0.1", "Tit for Tat 0.2", "Tit for Tat 0.4", "Tit for Two Tats 0.1"]
    dist_names = ["epsilon_0.1/Cooperator", "epsilon_0.2/Tit for Tat", "epsilon_0.4/Tit for Tat", "epsilon_0.1/Tit for Two Tats"]
    dists = []
    for dist_name in dist_names:
        dists.append(load_compressed_table("compressed_obvs/" + dist_name + ".txt"))
    
    io_play(Q, ActionProbability.tit_for_tat, 0.2, 10, 0.85, names, dists)

if __name__ == "__main__":
    main()