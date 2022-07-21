from lib2to3.pgen2.token import NAME
import random 
import math
import numpy as np
import sys
sys.path.append('..')
from strategies.strategy import *
from strategies.action_probability import ActionProbability

from value_iteration.learn import play

PAYOFF = [[(10.0, 10.0), (0.0, 20.0)], [(20.0, 0.0), (5.0, 5.0)]]
STRATS = [Cooperater, Defector, TitForTat, GrimTrigger, TitForTwoTats, TwoTitsForTats, Gradual, 
SoftMajority, HardMajority, RemorsefulProber, SoftGrudger, Prober]
# STRATS = [Cooperater, TitForTat, TwoTitsForTats]
NAMES = ["Cooperator", "Defector", "Tit for Tat", "Grim Trigger", "Tit for Two Tats", 
"Two Tits for Tat", "Gradual", "Soft Majority", "Hard Majority", "Remorseful Prober", "Soft Grudger", "Prober"]
# NAMES = ["Cooperator", "Tit for Tat", "Tit for Two Tats"]
ACTION = [ActionProbability.cooperator, ActionProbability.defector, ActionProbability.tit_for_tat, 
ActionProbability.grim_trigger, ActionProbability.tit_for_two_tats, ActionProbability.two_tits_for_tat, 
ActionProbability.gradual, ActionProbability.soft_majority, ActionProbability.hard_majority, 
ActionProbability.remorseful_prober, ActionProbability.soft_grudger, ActionProbability.prober]
# ACTION = [ActionProbability.cooperator, ActionProbability.tit_for_tat, ActionProbability.tit_for_two_tats]

def load_table(filename):
    Q = {}
    f = open(filename)
    lines = f.readlines()

    for line in lines: 
        full_state = line[:line.index(":")]
        dic = line[line.index(":") + 2:-2]
        states = full_state.split(", ")
        state = (states[0], states[1])
        entries = dic.split(", ")
        zero_entry = entries[0].split(":")
        one_entry = entries[1].split(":")
        actions = {int(zero_entry[0]): float(zero_entry[1]), int(one_entry[0]): float(one_entry[1])}
        Q[state] = actions
    
    return Q

def play_games(Q, opponent, next_act_dist, games, rounds, choice_param, epsilon):
    play_distribution = {}
    compressed_state_distribution = {}
    for game in range(games):
        strat_history = []
        agent_history = []
        strat_history_str = ""
        agent_history_str = "None"
        for round in range(rounds):
            if (strat_history_str, agent_history_str) not in play_distribution:
                play_distribution[(strat_history_str, agent_history_str)] = {}
            if strat_history_str not in compressed_state_distribution:
                compressed_state_distribution[strat_history_str] = {}

            opp_next_act_dist = next_act_dist(strat_history, agent_history, epsilon)
            values = [[Q[(strat_history_str + "0", agent_history_str)][0], 
            Q[(strat_history_str + "1", agent_history_str)][0]], 
            [Q[(strat_history_str + "0", agent_history_str)][1], 
            Q[(strat_history_str + "1", agent_history_str)][1]]]
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
            
            # print("play_distribtuion[(" + strat_history_str + ", " + agent_history_str + ")] = ", play_distribution[(strat_history_str, agent_history_str)])
            if my_action not in play_distribution[(strat_history_str, agent_history_str)]:
                play_distribution[(strat_history_str, agent_history_str)][my_action] = 1
            else:
                play_distribution[(strat_history_str, agent_history_str)][my_action] += 1
            
            if my_action not in compressed_state_distribution[strat_history_str]:
                compressed_state_distribution[strat_history_str][my_action] = 1
            else:
                compressed_state_distribution[strat_history_str][my_action] += 1
            # print("play_distribtuion[(" + strat_history_str + ", " + agent_history_str + ")][" + str(my_action) + "] = ", play_distribution[(strat_history_str, agent_history_str)][my_action])
            
            strat_action = opponent.choose(strat_history, agent_history, round)
            my_reward, strat_reward = PAYOFF[my_action][strat_action]
            # print("my action:", my_action, "strat action:", strat_action)
            # print("my reward:", my_reward, "strat reward:", strat_reward)
            # print("")
            agent_history.append(my_action)
            strat_history.append(strat_action)
            old_strat_history = strat_history_str
            strat_history_str = strat_history_str + str(strat_action)
            old_agent_history = agent_history_str
            if agent_history_str == "None":
                agent_history_str = str(my_action)
            else:
                agent_history_str = agent_history_str + str(my_action)
        
    for state in play_distribution.keys():
        total = sum(play_distribution[state].values())
        for action in play_distribution[state].keys():
            play_distribution[state][action] /= total
            # if play_distribution[state][action] != 1.0:
                # print("state:", state, "and action", action, "has distribution", play_distribution[state][action])
    
    for state in compressed_state_distribution.keys():
        total = sum(compressed_state_distribution[state].values())
        for action in compressed_state_distribution[state].keys():
            compressed_state_distribution[state][action] /= total

    return play_distribution, compressed_state_distribution

def save_dist(obvs_dist, filename):
    f = open(filename + ".txt","w")
    for state in obvs_dist.keys():
        # f.write(state[0] + ", " +  state[1] + ":" + str(obvs_dist[state]) + "\n")
        f.write(state + ":" + str(obvs_dist[state]) + "\n")
    f.close()

def main():
    for i in range(len(STRATS)):
        print("STRAT: " + NAMES[i])
        for e in range(1, 5, 1):
            epsilon = e / 10
            print("epsilon = " + str(epsilon) + "\n")
            Q = load_table("policies/epsilon_" + str(epsilon) + "/" + NAMES[i] + ".txt")
            opponent = STRATS[i](epsilon)
            observed_dist, compressed_dist = play_games(Q, opponent, ACTION[i], 10000, 10, 0.85, epsilon)
            save_dist(compressed_dist, "compressed_obvs/epsilon_" + str(epsilon) + "/" + NAMES[i])

if __name__ == "__main__":
    main()