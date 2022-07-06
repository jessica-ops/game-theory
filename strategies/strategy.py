import random

class Strategy: 
    """
    Super class which allows all strategies to play action with certain 
    probability of playing unexpected action. 
    """
    def __init__(self, unexpected_prob):
        self.unexpected_prob = unexpected_prob

    def play(self, move):
        prob = random.random()
        if prob <= self.unexpected_prob:
            return (move + 1) % 2
        return move 

class Cooperater(Strategy):
    """
    Cooperates on every move
    """
    def __init__(self, unexpected_prob):
        super().__init__(unexpected_prob)
    
    def choose(self, my_history, other_history, num_rounds):
        return super().play(0)

class Defector(Strategy):
    """
    Defects on every move
    """
    def __init__(self, unexpected_prob):
        super().__init__(unexpected_prob)

    def choose(self, my_history, other_history, num_rounds):
        return super().play(1)

class Random(Strategy):
    """
    Makes a random move built on probability distribution
    """
    def __init__(self, probability):
        self.probability = probability
    
    def choose(self, my_history, other_history, num_rounds):
        prob = random.random()
        if prob <= self.probability[0]:
            return 0
        return 1

class TitForTat(Strategy): 
    """
    Cooperates on first move and then copies oponent's last move
    """
    def __init__(self, unexpected_prob): 
        super().__init__(unexpected_prob)

    def choose(self, my_history, other_history, num_rounds):
        if num_rounds == 0:
            return super().play(0)

        return super().play(other_history[num_rounds - 1])

class GrimTrigger(Strategy):
    """
    Cooperates, until the opponent defects, and thereafter always defects.
    """
    def __init__(self, unexpected_prob):
        super().__init__(unexpected_prob)
        self.opponent_defected = False

    def choose(self, my_history, other_history, num_rounds):
        if num_rounds == 0:
            return super().play(0)
            
        if self.opponent_defected:
            return super().play(1)

        if other_history[num_rounds - 1] == 1:
            self.opponent_defected = True
            return super().play(1)

        return super().play(0)

class TitForTwoTats(Strategy):
    """
    Cooperates on the first two moves, and defects only when the opponent defects two times.
    """
    def __init__(self, unexpected_prob):
        super().__init__(unexpected_prob)

    def choose(self, my_history, other_history, num_rounds):
        if num_rounds < 2:
            return super().play(0)
        
        if other_history[num_rounds - 1] == 1 and other_history[num_rounds - 2] == 1:
            return super().play(1)
        return super().play(0)

class TwoTitsForTats(Strategy):
    """
    Same as Tit for Tat except that it defects twice when the opponent defects.
    """
    def __init__(self, unexpected_prob):
        super().__init__(unexpected_prob)
        self.must_defect = False

    def choose(self, my_history, other_history, num_rounds):
        if num_rounds == 0:
            return super().play(0)
        
        if self.must_defect:
            self.must_defect = False
            return super().play(1)
        
        if other_history[num_rounds - 1] == 1:
            self.must_defect = True
            return super().play(1)
        
        return super().play(0)

class Gradual(Strategy):
    """
    Cooperates on the first move, and cooperates as long as the opponent cooperates. 
    After the first defection of the other player, it defects one time and cooperates 
    two times; … After the nth defection it reacts with n consecutive defections and 
    then calms down its opponent with two cooperations.
    """
    def __init__(self, unexpected_prob):
        super().__init__(unexpected_prob)
        self.num_defections = 0
        self.countdown = -2

    def choose(self, my_history, other_history, num_rounds):
        if num_rounds == 0:
            return super().play(0)
        
        self.num_defections += other_history[num_rounds - 1]

        if self.countdown > 0:
            self.countdown -= 1
            return super().play(1)
        if -1 <= self.countdown <= 0:
            self.countdown -= 1
            return super().play(0)
        
        if other_history[num_rounds - 1] == 1:
            self.countdown = self.num_defections - 1
            return super().play(1)
        
        return super().play(0)

class SoftMajority(Strategy):
    """
    Cooperates on the first move, and cooperates as long as the number of times the 
    opponent has cooperated is greater than or equal to the number of times it has 
    defected, else it defects.
    """
    def __init__(self, unexpected_prob):
        super().__init__(unexpected_prob)
        self.opponent_cooperation = 0
        self.opponent_defection = 0

    def choose(self, my_history, other_history, num_rounds):
        if num_rounds == 0:
            return super().play(0)

        self.opponent_cooperation += (other_history[num_rounds - 1] + 1) % 2
        self.opponent_defection += other_history[num_rounds - 1]
        if self.opponent_cooperation < self.opponent_defection:
            return super().play(1)
        
        return super().play(0)

class HardMajority(Strategy):
    """
    Defects on the first move, and defects if the number of defections of the opponent 
    is greater than or equal to the number of times it has cooperated, else cooperates.
    """
    def __init__(self, unexpected_prob):
        super().__init__(unexpected_prob)
        self.opponent_cooperation = 0
        self.opponent_defection = 0

    def choose(self, my_history, other_history, num_rounds):
        if num_rounds == 0:
            return super().play(1)

        self.opponent_cooperation += (other_history[num_rounds - 1] + 1) % 2
        self.opponent_defection += other_history[num_rounds - 1]

        if self.opponent_cooperation <= self.opponent_defection:
            return super().play(1)
        
        return super().play(0)

class RemorsefulProber(Strategy):
    """
    Like Tit for Tat, but it tries to break the series of mutual 
    defections after defecting 5 times.
    """
    def __init__(self, unexpected_prob):
        super().__init__(unexpected_prob)
        self.max_defections = 5
        self.current_defections = 0

    def choose(self, my_history, other_history, num_rounds):
        if num_rounds == 0:
            return super().play(0)

        if other_history[num_rounds - 1] == 0:
            self.current_defections = 0
            return super().play(0)
        
        if self.current_defections >= self.max_defections:
            self.current_defections = 0
            return super().play(0)
        
        self.current_defections += 1
        return super().play(1)

class SoftGrudger(Strategy):
    """
    Like GRIM except that the opponent is punished with D,D,D,D,C,C.
    """
    def __init__(self, unexpected_prob):
        super().__init__(unexpected_prob)
        self.sequence = -2

    def choose(self, my_history, other_history, num_rounds):
        if num_rounds == 0:
            return super().play(0)
        
        if self.sequence > 0:
            self.sequence -= 1
            return super().play(1)
        if self.sequence > -2:
            self.sequence -= 1
            return super().play(0)

        if other_history[num_rounds - 1] == 1:
            self.sequence = 3
            return super().play(1)
        return super().play(0)

class Prober(Strategy):
    """
    Starts with D,C,C and then defects if the opponent has cooperated 
    in the second and third move; otherwise, it plays TFT.
    """
    def __init__(self, unexpected_prob):
        super().__init__(unexpected_prob)
        self.defector = False

    def choose(self, my_history, other_history, num_rounds):
        if num_rounds == 0: 
            return 1
        if num_rounds == 1:
            return 0
        if num_rounds == 2:
            return 0
        if num_rounds == 3: 
            if other_history[num_rounds - 1] == 0 and other_history[num_rounds - 2] == 0:
                self.defector = True
        if self.defector:
            return super().play(1)
        return super().play(other_history[num_rounds - 1])