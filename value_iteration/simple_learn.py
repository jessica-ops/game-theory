import numpy as np
import sys
sys.path.append('..')
from strategies.fsm import *

PAYOFF = [[(10, 10), (0, 20)], [(20, 0), (5, 5)]]

def q_table(transition_prob, threshold, gamma):
    terminal = False
    states = [0, 1]
    actions = [0, 1]
    Q = np.zeros(shape = (len(states), len(actions)))
    V = np.zeros(shape = (len(states), ))
    while not terminal: 
        delta = 0
        for s in states: 
            for a in actions: 
                Q[s][a] = 0
                for s_prime in states: 
                    Q[s][a] += transition_prob(s, a, s_prime) * (PAYOFF[a][s][0] + gamma * V[s_prime])
            delta = max(delta, abs(np.max((Q[s]) - V[s])))
            V[s] = np.max(Q[s])
        
        print("delta:", delta)
        if delta <= threshold:
            terminal = True
    
    return Q

def main(): 
    strategy = TitForTat()
    print(q_table(strategy.transition_prob, 0.0001, 0.9))

if __name__ == "__main__":
    main()