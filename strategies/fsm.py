class Defector: 
    def __init__(self, epsilon = 0):
        self.epsilon = epsilon
        self.startinng_state = {0: self.epsilon, 1: 1 - self.epsilon}
        self.transitions = {0: {0: {0: self.epsilon, 1: 1 - self.epsilon}, 1: {0: self.epsilon, 1: 1 - self.epsilon}}, 
        1: {0: {0: self.epsilon, 1: 1 - self.epsilon}, 1: {0: self.epsilon, 1: 1 - self.epsilon}}}
    
    def transition_prob(self, curr_state, action, new_state):
        if curr_state == None: 
            return self.starting_state[new_state]
        return self.transitions[curr_state][action][new_state]

class Cooperator: 
    def __init__(self, epsilon = 0):
        self.epsilon = epsilon
        self.startinng_state = {0: 1 - self.epsilon, 1: self.epsilon}
        self.transitions = {0: {0: {0: 1 - self.epsilon, 1: self.epsilon}, 1: {0: 1 - self.epsilon, 1: self.epsilon}}, 
        1: {0: {0: 1 - self.epsilon, 1: self.epsilon}, 1: {0: 1 - self.epsilon, 1: self.epsilon}}}
    
    def transition_prob(self, curr_state, action, new_state):
        if curr_state == None: 
            return self.starting_state[new_state]
        return self.transitions[curr_state][action][new_state]

class Grim: 
    """
    NOTE: Grim currently is only compatible with epsilon = 0
    """
    def __init__(self, epsilon = 0): 
        self.starting_state = {0: 1.0, 1: 0.0}
        self.transitions = {0: {0: {0: 1.0, 1: 0.0}, 1: {0: 0.0, 1: 1.0}}, 
        1: {0: {0: 0.0, 1: 1.0}, 1: {0: 0.0, 1: 1.0}}}
    
    def transition_prob(self, curr_state, action, new_state):
        if curr_state == None: 
            return self.starting_state[new_state]
        return self.transitions[curr_state][action][new_state]

class TitForTat: 
    def __init__(self, epsilon = 0):
        self.epsilon = epsilon
        self.startinng_state = {0: 1 - self.epsilon, 1: self.epsilon}
        self.transitions = {0: {0: {0: 1 - self.epsilon, 1: self.epsilon}, 1: {0: self.epsilon, 1: 1 - self.epsilon}}, 
        1: {0: {0: 1 - self.epsilon, 1: self.epsilon}, 1: {0: self.epsilon, 1: 1 - self.epsilon}}}
    
    def transition_prob(self, curr_state, action, new_state):
        if curr_state == None: 
            return self.starting_state[new_state]
        return self.transitions[curr_state][action][new_state]

class TatForTit:
    def __init__(self, epsilon = 0):
        self.epsilon = epsilon
        self.startinng_state = {0: self.epsilon, 1: 1 - self.epsilon}
        self.transitions = {0: {0: {0: 1 - self.epsilon, 1: self.epsilon}, 1: {0: self.epsilon, 1: 1 - self.epsilon}}, 
        1: {0: {0: 1, 1: 1 - self.epsilon}, 1: {0: 1 - self.epsilon, 1: self.epsilon}}}
    
    def transition_prob(self, curr_state, action, new_state):
        if curr_state == None: 
            return self.starting_state[new_state]
        return self.transitions[curr_state][action][new_state]