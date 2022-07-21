from argparse import Action
from collections import defaultdict
import sys
import numpy as np
import math
import random
sys.path.append('..')
from strategies.strategy import * 
from strategies.action_probability import ActionProbability

PAYOFF = [[(10.0, 10.0), (0.0, 20.0)], [(20.0, 0.0), (5.0, 5.0)]]
STRATS = [Cooperater, Defector, TitForTat, GrimTrigger, TitForTwoTats, TwoTitsForTats, Gradual, 
SoftMajority, HardMajority, RemorsefulProber, SoftGrudger, Prober]
NAMES = ["Cooperator", "Defector", "Tit for Tat", "Grim Trigger", "Tit for Two Tats", 
"Two Tits for Tat", "Gradual", "Soft Majority", "Hard Majority", "Remorseful Prober", "Soft Grudger", "Prober"]
ACTION = [ActionProbability.cooperator, ActionProbability.defector, ActionProbability.tit_for_tat, 
ActionProbability.grim_trigger, ActionProbability.tit_for_two_tats, ActionProbability.two_tits_for_tat, 
ActionProbability.gradual, ActionProbability.soft_majority, ActionProbability.hard_majority, 
ActionProbability.remorseful_prober, ActionProbability.soft_grudger, ActionProbability.prober]

def build_transition_probs(Strategy, num_games, num_rounds):
    start_count = {0: 0, 1: 0}
    transition_count = {}
    actions = [0, 1]
    states = [0, 1]
    opponent = Strategy(0)
    for game in range(num_games): 
        # print("game #" + str(game))
        states_to_actions = {}
        for round in range(num_rounds): 
            # print("round #" + str(round))
            new_states_to_actions = {}
            if round == 0:
                # print("\tfirst round")
                initial_state = opponent.choose([], [], 0)
                new_states_to_actions[str(initial_state)] = ["None"]
                start_count[initial_state] += 1
            else: 
                for state in states_to_actions.keys():
                    # print("\tstate: ", state)
                    for prev_actions in states_to_actions[state]:
                        # print("\t\tprevious actions: ", prev_actions)
                        for action in actions: 
                            # print("\t\taction:", action)
                            if prev_actions == "None":
                                prev_actions_list = []
                            else:
                                prev_actions_list = list(map(int, prev_actions))

                            prev_actions_list.append(action)
                            state_list = list(map(int, str(state)))

                            # print("\t\t\tstate list:", state_list)
                            # print("\t\t\tactions list:", prev_actions_list)
                            # print("\t\t\tround:", round)
                            next_state = opponent.choose(state_list, prev_actions_list, round)
                            # print("\t\t\tnext state:", next_state)

                            if prev_actions == "None":
                                new_actions = str(action)
                            else:
                                new_actions = str(prev_actions) + str(action)
                            # print("\t\t\tnew actions:", new_actions)
                            new_state = str(state) + str(next_state)

                            if new_state not in new_states_to_actions:
                                new_states_to_actions[new_state] = [new_actions]
                            else:
                                new_states_to_actions[new_state].append(new_actions)

                            if (state, prev_actions) not in transition_count:
                                # print("(" + str(state) + ", " + str(action) + ") not in transition count")
                                transition_count[(state, prev_actions)] = {action: {(new_state, new_actions): 1}}
                            else: 
                                if action not in transition_count[(state, prev_actions)]:
                                    transition_count[(state, prev_actions)][action] = {(new_state, new_actions): 1}
                                else: 
                                    if (new_state, new_actions) not in transition_count[(state, prev_actions)][action]:
                                        transition_count[(state, prev_actions)][action][(new_state, new_actions)] = 1
                                    else: 
                                        transition_count[(state, prev_actions)][action][(new_state, new_actions)] += 1
                            # print("\t\t\ttransition count: ", transition_count)
            states_to_actions = new_states_to_actions
            # print("states to actions:", states_to_actions)
    for state in transition_count.keys():
        for action in transition_count[state].keys():
            total = sum(transition_count[state][action].values())
            for next_state in transition_count[state][action].keys():
                transition_count[state][action][next_state] /= total
    return transition_count

def include_unexpected_moves(Strategy, transitions, epsilon, rounds):
    opponent = Strategy(0)
    for round in range(1, rounds):
        # print("round #" + str(round))
        new_transitions = {}
        for state in transitions:
            if len(state[0]) == round:
                new_transitions[state] = {}
                if transitions[state] == {}:
                    for a in [0, 1]:
                        new_transitions[state][a] = {}
                        prev_actions_list = list(map(int, state[1]))
                        prev_actions_list = prev_actions_list + [a]
                        states_list = list(map(int, state[0]))
                        new_state = opponent.choose(states_list, prev_actions_list, round)
                        expected_state = state[0] + str(new_state)
                        unexpected_state = state[0] + str((new_state + 1) % 2)
                        actions = state[1] + str(a)
                        new_transitions[state][a][(expected_state, actions)] = 1 - epsilon
                        new_transitions[state][a][(unexpected_state, actions)] = epsilon
                        new_transitions[(expected_state, actions)] = {}
                        new_transitions[(unexpected_state, actions)] = {}
                else:
                    for action in transitions[state]:
                        new_transitions[state][action] = {}
                        for s_prime in transitions[state][action]:
                            new_transitions[state][action][s_prime] = 1 - epsilon
                            last_state = int(s_prime[0][-1])
                            new_last_state = (last_state + 1) % 2
                            new_state = s_prime[0][:-1] + str(new_last_state)
                            # print("state:", state)
                            # print("action:", action)
                            # print("sprime: (" + new_state + ", " + s_prime[1] + ")")
                            new_transitions[state][action][(new_state, s_prime[1])] = epsilon
                            new_transitions[(new_state, s_prime[1])] = {}
                if state[1] == "None":
                    starting_state = int(state[0])
                    unexpected_starting_state = (starting_state + 1) % 2
                    start_state = (str(unexpected_starting_state), "None")
                    new_transitions[start_state] = {}
                    for a in [0, 1]:
                        new_transitions[start_state][a] = {}
                        prev_actions_list = [a]
                        states_list = list(map(int, start_state[0]))
                        new_state = opponent.choose(states_list, prev_actions_list, round)
                        expected_state = start_state[0] + str(new_state)
                        unexpected_state = start_state[0] + str((new_state + 1) % 2)
                        actions = str(a)
                        new_transitions[start_state][a][(expected_state, actions)] = 1 - epsilon
                        # print("new_transitions[", start_state, "][" + str(a) + "][", (expected_state, actions), "] = ", new_transitions[start_state][a][(expected_state, actions)])
                        new_transitions[start_state][a][(unexpected_state, actions)] = epsilon
                        # print("new_transitions[", start_state, "][" + str(a) + "][", (unexpected_state, actions), "] = ", new_transitions[start_state][a][(unexpected_state, actions)])
                        new_transitions[(expected_state, actions)] = {}
                        new_transitions[(unexpected_state, actions)] = {}

            else:
                new_transitions[state] = {}
                for action in transitions[state]:
                    new_transitions[state][action] = {}
                    for s_prime in transitions[state][action]:
                        new_transitions[state][action][s_prime] = transitions[state][action][s_prime]

        transitions = new_transitions 
        # print("transitions:", transitions) 

    return transitions

def learn(transitions, threshold, gamma, rounds):
    terminal = False
    states = set()
    for s in transitions.keys():
        states.add(s)
        for a in transitions[s].keys():
            for s_prime in transitions[s][a].keys():
                states.add(s_prime)
    # print("states", states)
    actions = [0, 1]
    Q = {}
    V = {}
    for s in states:
        if len(s[0]) == rounds:
            Q[s] = {0: PAYOFF[0][int(s[0][-1])][0], 1: PAYOFF[1][int(s[0][-1])][0]}
            V[s] = max(PAYOFF[0][int(s[0][-1])][0], PAYOFF[1][int(s[0][-1])][0])
            # print("V[", s, "] = ", V[s])
        else:
            V[s] = 0
            Q[s] = {0: 0, 1: 0}

    i = 0
    while not terminal: 
        delta = 0
        for s in states: 
            # print("s:", s)
            if len(s[0]) == rounds:
                continue
            for a in actions: 
                # print("\ta:", a)
                Q[s][a] = 0
                for s_prime in actions: 
                    # print("\t\ts prime:", s_prime)
                    new_state = s[0] + str(s_prime)
                    if s[1] == "None":
                        new_actions = str(a)
                    else:
                        new_actions = s[1] + str(a)
                    
                    if (new_state, new_actions) not in transitions[s][a]:
                        # print("\t\t\t(" + new_state + ", " + new_actions + ") not in transitions")
                        continue
                    else:
                        # print("\t\t\tQ[", s, "][" + str(a) + "] = ", Q[s][a])
                        # print("\t\t\tnew state: ", new_state)
                        # print("\t\t\tPAYOFF:", PAYOFF[a][int(s[0][-1])][0])
                        # print("\t\t\tV[(" + new_state + ", " + new_actions + ")] = ", V[(new_state, new_actions)])
                        transition_prob = transitions[s][a][(new_state, new_actions)]
                        Q[s][a] += PAYOFF[a][int(s[0][-1])][0] + (transition_prob * gamma * V[(new_state, new_actions)])
                        # print("\t\t\tQ[s][a] after:", Q[s][a])
                        # print("")
            
            values = []
            for act in Q[s].keys():
                values.append(Q[s][act])
            delta = max(delta, abs(max(values) - V[s]))
            # print("values:", values)
            V[s] = max(values)
            # print("V[", s, "] = ", V[s])
            # print("-----------------------------------------")
        
        i += 1
        # print("delta:", delta)
        if delta <= threshold:
            terminal = True
    
    return Q

def play(Q, opponent, epsilon, next_act_dist, rounds, choice_param):
    # agent_history = []
    # strat_history = []
    # str_agent_hist = "None"
    # str_strat_hist = ""
    # for round in range(rounds):
    #     potential_strat_action = np.argmax(next_act_dist(strat_history, agent_history, 0))
    #     potential_strat_hist = str_strat_hist + str(potential_strat_action)
    #     # print("potential strat hist: "+ potential_strat_hist)
    #     max_val = -1
    #     my_action = -1
    #     for act in Q[(potential_strat_hist, str_agent_hist)].keys():
    #         if Q[(potential_strat_hist, str_agent_hist)][act] > max_val:
    #             max_val = Q[(potential_strat_hist, str_agent_hist)][act]
    #             my_action = act
    #     strat_action = opponent.choose(strat_history, agent_history, round)
    #     my_reward, strat_reward = PAYOFF[my_action][strat_action]
    #     print("my action:", my_action, "strat action:", strat_action)
    #     print("my reward:", my_reward, "strat reward:", strat_reward)
    #     agent_history.append(my_action)
    #     strat_history.append(strat_action)
    #     str_strat_hist = str_strat_hist + str(strat_action)
    #     if str_agent_hist == "None":
    #         str_agent_hist = str(my_action)
    #     else:
    #         str_agent_hist = str_agent_hist + str(my_action)
    strat_history = []
    agent_history = []
    strat_history_str = ""
    agent_history_str = "None"
    for round in range(rounds):
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
        strat_action = opponent.choose(strat_history, agent_history, round)
        my_reward, strat_reward = PAYOFF[my_action][strat_action]
        print("my action:", my_action, "strat action:", strat_action)
        print("my reward:", my_reward, "strat reward:", strat_reward)
        print("")
        agent_history.append(my_action)
        strat_history.append(strat_action)
        strat_history_str = strat_history_str + str(strat_action)
        if agent_history_str == "None":
            agent_history_str = str(my_action)
        else:
            agent_history_str = agent_history_str + str(my_action)

def save(q_table, filename):
    f = open(filename + ".txt","w")
    for state in q_table.keys():
        f.write(state[0] + ", " +  state[1] + ":" + str(q_table[state]) + "\n")
    f.close()

def main():
    rounds = 10
    # Strategy = TitForTat
    # next_action = ActionProbability.tit_for_tat
    # epsilon = 0.2
    # transitions = build_transition_probs(Strategy, 1, rounds)
    # if epsilon != 0.0:
    #     transitions = include_unexpected_moves(Strategy, transitions, epsilon, rounds)
    #     # print("unexpected moves:", transitions)
    # q_table = learn(transitions, 0.001, 1, rounds)
    # play(q_table, Strategy(epsilon), epsilon, next_action, rounds, 0.85)

    for e in range(0, 5, 1):
        epsilon = e / 10
        print("EPSILON = " + str(epsilon))
        for i in range(len(STRATS)):
            print("\tSTRATEGY: " + NAMES[i])
            Strategy = STRATS[i]
            # next_action = ACTION[i]

            transitions = build_transition_probs(Strategy, 1, rounds)
            print("\tcalculated unexpected moves")
            if epsilon != 0.0:
                transitions = include_unexpected_moves(Strategy, transitions, epsilon, rounds)
            print("\tincluded unepected probabilities")
            q_table = learn(transitions, 0.001, 1, rounds)
            save(q_table, "policies/epsilon_" + str(epsilon) + "/" + NAMES[i])
            print("\tlearned and saved policy\n")

if __name__ == "__main__":
    main()