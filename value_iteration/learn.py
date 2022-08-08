from argparse import Action
from collections import defaultdict
import sys
import numpy as np
import random
sys.path.append('..')
from strategies.strategy import * 
from strategies.action_probability import ActionProbability

PAYOFF = [[(10.0, 10.0), (0.0, 20.0)], [(20.0, 0.0), (5.0, 5.0)]] # payout matrix of game

# All strategies to learn about. Class, names, and action probability. 
STRATS = [Cooperater, Defector, TitForTat, GrimTrigger, TitForTwoTats, TwoTitsForTats, Gradual, 
SoftMajority, HardMajority, RemorsefulProber, SoftGrudger, Prober]
NAMES = ["Cooperator", "Defector", "Tit for Tat", "Grim Trigger", "Tit for Two Tats", 
"Two Tits for Tat", "Gradual", "Soft Majority", "Hard Majority", "Remorseful Prober", "Soft Grudger", "Prober"]
ACTION = [ActionProbability.cooperator, ActionProbability.defector, ActionProbability.tit_for_tat, 
ActionProbability.grim_trigger, ActionProbability.tit_for_two_tats, ActionProbability.two_tits_for_tat, 
ActionProbability.gradual, ActionProbability.soft_majority, ActionProbability.hard_majority, 
ActionProbability.remorseful_prober, ActionProbability.soft_grudger, ActionProbability.prober]

def build_transition_probs(Strategy, num_rounds):
    """
    Takes a specific strategy class (Strategy) and computes the pure transition probability
    for the given number of rounds (num_rounds). This formats the world as a MDP dictated by
    the specific strategy. 
    """
    opponent = Strategy(0) # initialize the opponent 
    transition_count = {}
    states_to_actions = {}
    actions = [0, 1]

    # Iterate through all rounds
    for round in range(num_rounds): 
        new_states_to_actions = {}

        if round == 0: # Case: First Round
            initial_state = opponent.choose([], [], 0)
            new_states_to_actions[str(initial_state)] = ["None"]
        else: 
            # Iterate through all perviouly computed state history
            for state in states_to_actions.keys(): 
                # Iterate through all action history which yielded those states
                for prev_actions in states_to_actions[state]: 
                    for action in actions: # Next action 
                        if prev_actions == "None":
                            prev_actions_list = []
                        else:
                            prev_actions_list = list(map(int, prev_actions))

                        prev_actions_list.append(action)
                        state_list = list(map(int, str(state)))

                        # Based on state, action history, and new action: what will opponent play 
                        next_state = opponent.choose(state_list, prev_actions_list, round)

                        # Update action and state history
                        if prev_actions == "None":
                            new_actions = str(action)
                        else:
                            new_actions = str(prev_actions) + str(action)
                        
                        new_state = str(state) + str(next_state)

                        # Update new states to actions
                        if new_state not in new_states_to_actions:
                            new_states_to_actions[new_state] = [new_actions]
                        else:
                            new_states_to_actions[new_state].append(new_actions)

                        # Update transition count
                        if (state, prev_actions) not in transition_count:
                            transition_count[(state, prev_actions)] = {action: {(new_state, new_actions): 1}}
                        else: 
                            if action not in transition_count[(state, prev_actions)]:
                                transition_count[(state, prev_actions)][action] = {(new_state, new_actions): 1}
                            else: 
                                if (new_state, new_actions) not in transition_count[(state, prev_actions)][action]:
                                    transition_count[(state, prev_actions)][action][(new_state, new_actions)] = 1
                                else: 
                                    transition_count[(state, prev_actions)][action][(new_state, new_actions)] += 1

        # Update states to actions 
        states_to_actions = new_states_to_actions
    
    # Normalize to compute probabilities
    for state in transition_count.keys():
        for action in transition_count[state].keys():
            total = sum(transition_count[state][action].values())
            for next_state in transition_count[state][action].keys():
                transition_count[state][action][next_state] /= total

    return transition_count

def include_unexpected_moves(Strategy, transitions, epsilon, rounds):
    """
    Given opponent strategy (Strategy) and computed pure transition probabilities (transitions), 
    add in the probability that the opponent will play unexpectedly with a certain probability (epsilon).
    """
    opponent = Strategy(0)
    first_act = opponent.choose([], [], 0)
    start_dist = {first_act: 1 - epsilon, (first_act + 1) % 2: epsilon}
    
    # Iterate through all rounds 
    for round in range(1, rounds):
        new_transitions = {}

        # Iterate through all states 
        for state in transitions:
            # If state matches appropriate round 
            if len(state[0]) == round:
                new_transitions[state] = {}
                # If children of state have not been determined 
                if transitions[state] == {}:
                    for a in [0, 1]: # new acton 
                        new_transitions[state][a] = {}
                        prev_actions_list = list(map(int, state[1]))
                        prev_actions_list = prev_actions_list + [a]
                        states_list = list(map(int, state[0]))

                        # based on state history, action history, and new action: what will opponent play
                        new_state = opponent.choose(states_list, prev_actions_list, round)

                        # add in expected and unexpected state to transitions[state]
                        expected_state = state[0] + str(new_state)
                        unexpected_state = state[0] + str((new_state + 1) % 2)
                        actions = state[1] + str(a)
                        new_transitions[state][a][(expected_state, actions)] = 1 - epsilon
                        new_transitions[state][a][(unexpected_state, actions)] = epsilon

                        # add in expected and unexpected state to transitions (no children)
                        new_transitions[(expected_state, actions)] = {}
                        new_transitions[(unexpected_state, actions)] = {}
                
                # If state already has children (next states)
                else:
                    for action in transitions[state]:
                        new_transitions[state][action] = {}
                        for s_prime in transitions[state][action]:
                            # update expected next state 
                            new_transitions[state][action][s_prime] = 1 - epsilon

                            # add in unexpected next state to children
                            last_state = int(s_prime[0][-1])
                            new_last_state = (last_state + 1) % 2
                            new_state = s_prime[0][:-1] + str(new_last_state)
                            new_transitions[state][action][(new_state, s_prime[1])] = epsilon

                            # add unxpected next state to transitions (no children) 
                            new_transitions[(new_state, s_prime[1])] = {}
                
                if state[1] == "None": # Case: first round
                    starting_state = int(state[0])

                    # add in unexpected stating state to transitions
                    unexpected_starting_state = (starting_state + 1) % 2
                    start_state = (str(unexpected_starting_state), "None")
                    new_transitions[start_state] = {}
                    
                    # determine unexpected starting state's children
                    for a in [0, 1]:
                        new_transitions[start_state][a] = {}
                        prev_actions_list = [a]
                        states_list = list(map(int, start_state[0]))

                        # determine opponents expected play 
                        new_state = opponent.choose(states_list, prev_actions_list, round)

                        expected_state = start_state[0] + str(new_state)
                        unexpected_state = start_state[0] + str((new_state + 1) % 2)
                        actions = str(a)

                        # add in children to start state 
                        new_transitions[start_state][a][(expected_state, actions)] = 1 - epsilon
                        new_transitions[start_state][a][(unexpected_state, actions)] = epsilon

                        # add next states to transitions (no children)
                        new_transitions[(expected_state, actions)] = {}
                        new_transitions[(unexpected_state, actions)] = {}

            else: # If it isnt the right round, copy info over to new transitions 
                new_transitions[state] = {}
                for action in transitions[state]:
                    new_transitions[state][action] = {}
                    for s_prime in transitions[state][action]:
                        new_transitions[state][action][s_prime] = transitions[state][action][s_prime]

        transitions = new_transitions 

    return transitions, start_dist

def learn(transitions, threshold, gamma, rounds):
    """
    Build Q table using Value Iteration Algorithm. 
    """

    # set of all states
    states = set()
    for s in transitions.keys():
        states.add(s)
        for a in transitions[s].keys():
            for s_prime in transitions[s][a].keys():
                states.add(s_prime)

    # Initialize Q and V
    Q = {}
    V = {}
    for s in states:
        if len(s[0]) == rounds: # Compute rewards for the last round
            end_0_val = PAYOFF[0][int(s[0][-1])][0]
            end_1_val = PAYOFF[1][int(s[0][-1])][0] 
            Q[s] = {0: end_0_val, 1: end_1_val} # Q 
            V[s] = max(end_0_val, end_1_val) # V 
        else:
            V[s] = 0
            Q[s] = {0: 0, 1: 0}

    actions = [0, 1]
    terminal = False
    while not terminal: # Iterate until V converges
        delta = 0
        for s in states: # Iterate through all states
            if len(s[0]) == rounds: # Last round has already been computed
                continue
            for a in actions: # Iterate through all actions
                Q[s][a] = PAYOFF[a][int(s[0][-1])][0] # Initialize to payoff of state s and action a 
                for s_prime in actions: # Iterate through potential next state
                    new_state = s[0] + str(s_prime)
                    if s[1] == "None":
                        new_actions = str(a)
                    else:
                        new_actions = s[1] + str(a)
                    
                    # If the new state is not in transitions, don't compute 
                    if (new_state, new_actions) not in transitions[s][a]:
                        continue
                    else:
                        # get transition probability of going from state s to new state with action a 
                        transition_prob = transitions[s][a][(new_state, new_actions)]
                        # add to Q[s][a]
                        Q[s][a] += (transition_prob * gamma * V[(new_state, new_actions)])
            
            values = []
            for act in Q[s].keys():
                values.append(Q[s][act])
            
            # update delta value
            delta = max(delta, abs(max(values) - V[s]))
            # update V[s]
            V[s] = max(values)

        if delta <= threshold: # test whether V has converged
            terminal = True
    
    return Q

def play(Q, opponent, epsilon, next_act_dist, rounds, temperature):
    """
    Policy optimizied against Opponent(epsilon) with certain temperature vs Opponent(epsilon)
    """

    # Initialize strategy and agent history
    strat_history = []
    agent_history = []
    strat_history_str = ""
    agent_history_str = "None"

    # Iterate through all rounds
    for round in range(rounds):
        # Strategy action distribution
        opp_next_act_dist = next_act_dist(strat_history, agent_history, epsilon)

        # Q values of potential states + action
        values = [[Q[(strat_history_str + "0", agent_history_str)][0], 
        Q[(strat_history_str + "1", agent_history_str)][0]], 
        [Q[(strat_history_str + "0", agent_history_str)][1], 
        Q[(strat_history_str + "1", agent_history_str)][1]]]
        
        if round != rounds - 1:
            # Incorporate strategy action distribution when comparing Q values
            values[0][0] *= opp_next_act_dist[0]
            values[1][0] *= opp_next_act_dist[0]
            values[0][1] *= opp_next_act_dist[1]
            values[1][1] *= opp_next_act_dist[1]
        
        # Get value of playing 0 or 1 
        action_value = np.array([sum(values[0]), sum(values[1])])

        # Action distribtuion via Boltzmann Distribtuion
        norm = sum(np.exp(action_value / temperature))
        act_probs = np.exp(action_value / temperature) / norm

        # Choose action using action probabilities
        prob = random.random()
        if prob <= act_probs[0]:
            my_action = 0
        else:
            my_action = 1
        
        # Get strategy action and reward 
        strat_action = opponent.choose(strat_history, agent_history, round)
        my_reward, strat_reward = PAYOFF[my_action][strat_action]

        # Output result of round
        print("my action:", my_action, "strat action:", strat_action)
        print("my reward:", my_reward, "strat reward:", strat_reward)
        print("")

        # Update histories
        agent_history.append(my_action)
        strat_history.append(strat_action)
        strat_history_str = strat_history_str + str(strat_action)
        if agent_history_str == "None":
            agent_history_str = str(my_action)
        else:
            agent_history_str = agent_history_str + str(my_action)

def save(q_table, filename):
    """
    Save the Q table (q_table) to path filename. 
    """
    f = open(filename + ".txt","w")
    for state in q_table.keys():
        f.write(state[0] + ", " +  state[1] + ":" + str(q_table[state]) + "\n")
    f.close()

def main():
    """
    Play against optimized strategy. 
    """
    rounds = 10
    Strategy = Gradual
    next_action = ActionProbability.gradual
    epsilon = 0.2
    transitions = build_transition_probs(Strategy, 1, rounds)
    if epsilon != 0.0:
        transitions, start_dist = include_unexpected_moves(Strategy, transitions, epsilon, rounds)
    else:
        first_move = Strategy(0).choose([], [], 0)
        start_dist = {first_move: 1, (first_move + 1) % 2: 0}
    q_table = learn(transitions, 0.001, 1, rounds)
    play(q_table, Strategy(epsilon), epsilon, next_action, rounds, 2)

    """
    Save Q tables for all strategies (listed at top of file)
    """
    # rounds = 10
    # for e in range(1, 5, 1):
    #     epsilon = e / 10
    #     print("EPSILON = " + str(epsilon))
    #     for i in range(len(STRATS)):
    #         print("\tSTRATEGY: " + NAMES[i])
    #         Strategy = STRATS[i]
    #         # next_action = ACTION[i]

    #         transitions = build_transition_probs(Strategy, 1, rounds)
    #         print("\tcalculated unexpected moves")
    #         if epsilon != 0.0:
    #             transitions, start_dist = include_unexpected_moves(Strategy, transitions, epsilon, rounds)
    #         print("\tincluded unepected probabilities")
    #         q_table = learn(transitions, 0.001, 1, rounds)
    #         save(q_table, "policies/epsilon_" + str(epsilon) + "/" + NAMES[i])
    #         print("\tlearned and saved policy\n")

if __name__ == "__main__":
    main()