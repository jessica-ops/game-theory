from collections import defaultdict
import sys
import numpy as np
sys.path.append('..')
from strategies.strategy import * 
from strategies.action_probability import ActionProbability

PAYOFF = [[(10.0, 10.0), (0.0, 20.0)], [(20.0, 0.0), (5.0, 5.0)]]

def build_transition_probs(Opponent, num_games, num_rounds):
    start_count = {0: 0, 1: 0}
    transition_count = {}
    actions = [0, 1]
    states = [0, 1]
    strategy = Opponent(0)
    for game in range(num_games): 
        # print("game #" + str(game))
        states_to_actions = {}
        for round in range(num_rounds): 
            # print("round #" + str(round))
            new_states_to_actions = {}
            if round == 0:
                # print("\tfirst round")
                strategy = Opponent(0)
                initial_state = strategy.choose([], [], 0)
                new_states_to_actions[str(initial_state)] = [None]
                start_count[initial_state] += 1
            else: 
                for state in states_to_actions.keys():
                    # print("\tstate: ", state)
                    for prev_actions in states_to_actions[state]:
                        # print("\t\tprevious actions: ", prev_actions)
                        for action in actions: 
                            # print("\t\taction:", action)
                            if prev_actions == None:
                                prev_actions_list = []
                            else:
                                prev_actions_list = list(map(int, prev_actions))

                            prev_actions_list.append(action)
                            state_list = list(map(int, str(state)))

                            # print("\t\t\tstate list:", state_list)
                            # print("\t\t\tactions list:", prev_actions_list)
                            # print("\t\t\tround:", round)
                            next_state = strategy.choose(state_list, prev_actions_list, round)
                            # print("\t\t\tnext state:", next_state)

                            if prev_actions == None:
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
                    if s[1] == None:
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

def play(Q, opponent, next_act_dist, rounds):
    agent_history = []
    strat_history = []
    str_agent_hist = None
    str_strat_hist = ""
    for round in range(rounds):
        potential_strat_action = np.argmax(next_act_dist(strat_history, agent_history, 0))
        potential_strat_hist = str_strat_hist + str(potential_strat_action)
        # print("potential strat hist: "+ potential_strat_hist)
        max_val = -1
        my_action = -1
        for act in Q[(potential_strat_hist, str_agent_hist)].keys():
            if Q[(potential_strat_hist, str_agent_hist)][act] > max_val:
                max_val = Q[(potential_strat_hist, str_agent_hist)][act]
                my_action = act
        strat_action = opponent.choose(strat_history, agent_history, round)
        my_reward, strat_reward = PAYOFF[my_action][strat_action]
        print("my action:", my_action, "strat action:", strat_action)
        print("my reward:", my_reward, "strat reward:", strat_reward)
        agent_history.append(my_action)
        strat_history.append(strat_action)
        str_strat_hist = str_strat_hist + str(strat_action)
        if str_agent_hist == None:
            str_agent_hist = str(my_action)
        else:
            str_agent_hist = str_agent_hist + str(my_action)

def main():
    rounds = 5
    Opponent = Prober
    next_action = ActionProbability.prober

    transitions = build_transition_probs(Opponent, 1, rounds)
    #print("transitions: ", transitions)
    q_table = learn(transitions, 0.001, 1, rounds)
    #print("q table: ", q_table)
    play(q_table, Opponent(0), next_action, rounds)

if __name__ == "__main__":
    main()