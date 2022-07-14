class Likelihood:
    """
    Calculates likelhood of a player being a certain strategy given previous action of 
    player and opponent, the previously calculated likelihood, and number of unexpected 
    actions thus far. 
    """
    def __init__(self, epsilon):
        """
        Unexpected probability = number of unexpected moves * epsilon 
        If epsilon = 0.01, unexpected probability will equal unexpected moves / total moves
        by round 100. 
        """
        self.epsilon = epsilon

    """
    Cooperates on every move
    """
    def cooperator(self, strat_acts, opponent_acts, prev_lh, unexpected):
        if prev_lh == 0:
            return 0, unexpected

        # expected action is always 0 
        if strat_acts[-1] == 0:
            return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
        else:
            unexpected += 1
            return prev_lh * unexpected * self.epsilon, unexpected
    
    """
    Defects on every move
    """
    def defector(self, strat_acts, opponent_acts, prev_lh, unexpected):
        if prev_lh == 0:
            return 0, unexpected

        # expected move is always 1
        if strat_acts[-1] == 1:
            return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
        else:
            unexpected += 1
            return prev_lh * unexpected * self.epsilon, unexpected

    """
    Makes a random move (50/50)
    """
    def random(self, strat_acts, opponent_acts, prev_lh, unexpected):
        if prev_lh == 0:
            return 0, unexpected

        # likelihood of action is always .50 
        return 0.5 * prev_lh, unexpected

    """
    Cooperates on first move and then copies oponent's last move
    """
    def tit_for_tat(self, strat_acts, opponent_acts, prev_lh, unexpected):
        if prev_lh == 0:
            return 0, unexpected

        if len(strat_acts) == 1:
            # first move, expected action is 0
            if strat_acts[0] == 0:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected

        # expected move is to copy opponent's last move 
        if opponent_acts[-2] == strat_acts[-1]:
            return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
        else:
            unexpected += 1
            return prev_lh * unexpected * self.epsilon, unexpected

    """
    Cooperates, until the opponent defects, and thereafter always defects.
    """
    def grim_trigger(self, strat_acts, opponent_acts, prev_lh, unexpected):
        if prev_lh == 0:
            return 0, unexpected

        if len(strat_acts) == 1:
            # first move, expected action is 0
            if strat_acts[0] == 0:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected

        # if opponent has ever played 1, expected action is 1
        if 1 in opponent_acts[:-1] and strat_acts[-1] == 1:
            return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
        # if opponent has never played 1, expected action is 0
        elif 1 not in opponent_acts[:-1] and strat_acts[-1] == 0:
            return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
        else:
            unexpected += 1
            return prev_lh * unexpected * self.epsilon, unexpected

    """
    Cooperates on the first two moves, and defects only when the opponent defects two times.
    """
    def tit_for_two_tats(self, strat_acts, opponent_acts, prev_lh, unexpected):
        if prev_lh == 0:
            return 0, unexpected

        if len(strat_acts) == 1:
            # first move, expected action is 0
            if strat_acts[0] == 0:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected

        if len(strat_acts) == 2:
            # second move, expected action is 0
            if strat_acts[1] == 0:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected
        
        # if opponent has defected twice, expected action is 1
        if opponent_acts[-3] == 1 and opponent_acts[-2] == 1: 
            if strat_acts[-1] == 1:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else: 
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected
        # otherwise, expected action is 0
        else: 
            if strat_acts[-1] == 0:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else: 
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected

    """
    Same as Tit for Tat except that it defects twice when the opponent defects.
    """
    def two_tits_for_tat(self, strat_acts, opponent_acts, prev_lh, unexpected):
        if prev_lh == 0:
            return 0, unexpected

        if len(strat_acts) < 2:
            # first move, expected action is 0
            if strat_acts[0] == 0:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected
        
        # if opponent defected last round, expected move is 1
        if opponent_acts[-2] == 1:
            if strat_acts[-1] == 1:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected
        # if opponent defected round before last, expected move is 1
        elif len(opponent_acts) > 2 and opponent_acts[-3] == 1:
            if strat_acts[-1]:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected
        # otherwise, expected move is 0
        else:
            if strat_acts[-1] == 0:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected

    """
    Cooperates on the first move, and cooperates as long as the opponent cooperates. 
    After the first defection of the other player, it defects one time and cooperates 
    two times; … After the nth defection it reacts with n consecutive defections and 
    then calms down its opponent with two cooperations.
    """
    def gradual(self, strat_acts, opponent_acts, prev_lh, unexpected):
        if prev_lh == 0:
            return 0, unexpected

        unexpected = 0
        if strat_acts[0] != 0:
            unexpected += 1

        if len(strat_acts) == 1:
            # first move, expected action is 0
            if strat_acts[0] == 0:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                return prev_lh * unexpected * self.epsilon, unexpected
        
        num_defects = opponent_acts[0]
        lh = 1
        if num_defects > 0:
            defects_left = num_defects
        else:
            defects_left = -2

        # go through each round to calculate likelihood
        for r in range(1, len(strat_acts)):
            num_defects += opponent_acts[r]
            # expected action is 1 
            if defects_left > 0: 
                defects_left -= 1
                if strat_acts[r] == 1:
                    lh *= (1 - (unexpected * self.epsilon))
                    continue
                else:
                    unexpected += 1
                    lh *= unexpected * self.epsilon
                    continue
            # expected action is 0
            elif defects_left > -2:
                defects_left -= 1
                if strat_acts[r] == 0:
                    lh *= (1 - (unexpected * self.epsilon))
                else: 
                    unexpected += 1
                    lh *= unexpected * self.epsilon
                if defects_left == -2:
                    if opponent_acts[r] == 1:
                        defects_left = num_defects
                continue
            else:
                # expected action is 0
                if opponent_acts[r - 1] == 0:
                    if strat_acts[r] == 0:
                        lh *= (1 - (unexpected * self.epsilon))
                    else:
                        unexpected += 1
                        lh *= unexpected * self.epsilon
                # expected to start defecting next round
                if opponent_acts[r] == 1:
                    defects_left = num_defects
        return lh, unexpected

    """
    Cooperates on the first move, and cooperates as long as the number of times the 
    opponent has cooperated is greater than or equal to the number of times it has 
    defected, else it defects.
    """
    def soft_majority(self, strat_acts, opponent_acts, prev_lh, unexpected):
        if prev_lh == 0:
            return 0, unexpected

        if len(strat_acts) == 1:
            # first move, expected action is 0
            if strat_acts[0] == 0:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected

        # opponent 1s has not exceeded 0s, expected action is 0
        if len(opponent_acts[:-1]) >= 2 * sum(opponent_acts[:-1]):
            if strat_acts[-1] == 0:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected
        # opponent 1s have exceeded 0s, expected action is 1
        else: 
            if strat_acts[-1] == 1:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected

    """
    Defects on the first move, and defects if the number of defections of the opponent 
    is greater than or equal to the number of times it has cooperated, else cooperates.
    """
    def hard_majority(self, strat_acts, opponent_acts, prev_lh, unexpected):
        if prev_lh == 0:
            return 0, unexpected

        if len(strat_acts) == 1:
            # first move, expected action is 1
            if strat_acts[0] == 1:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected

        # opponent 1s < 0s, expected move is 0
        if len(opponent_acts[:-1]) > 2 * sum(opponent_acts[:-1]):
            if strat_acts[-1] == 0:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected
        # opponent 1s >= 0s, expected move is 1 
        else: 
            if strat_acts[-1] == 1:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected

    """
    Like Tit for Tat, but it tries to break the series of mutual 
    defections after defecting 5 times.
    """
    def remorseful_prober(self, strat_acts, opponent_acts, prev_lh, unexpected):
        if prev_lh == 0:
            return 0, unexpected

        if len(strat_acts) == 1:
            # first move, expected action is 0
            if strat_acts[0] == 0:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected
        
        # if player & opponent have defected for the past 5 moves, expected move is 0
        if len(strat_acts) > 5 and sum(opponent_acts[-6:-1]) == 5 and sum(strat_acts[-6:-1]) == 5:
            if strat_acts[-1] == 0:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected
        
        # otherwise, player is expected to repeat opponent's previous move
        if opponent_acts[-2] == strat_acts[-1]:
            return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
        else:
            unexpected += 1
            return prev_lh * unexpected * self.epsilon, unexpected

    """
    Like GRIM except that the opponent is punished with D,D,D,D,C,C.
    """
    def soft_grudger(self, strat_acts, opponent_acts, prev_lh, unexpected):
        if prev_lh == 0:
            return 0, unexpected

        unexpected = 0
        if strat_acts[0] != 0:
            unexpected += 1

        if len(strat_acts) == 1:
            # first move, expected action is 0
            if strat_acts[0] == 0:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                return prev_lh * unexpected * self.epsilon, unexpected

        if opponent_acts[0] == 1:
            sequence = 4
        else:
            sequence = -2
        lh = 1

        # go through all actions to compute likelihood 
        for r in range(1, len(strat_acts)):
            # expected action is 1
            if sequence > 0:
                sequence -= 1 
                if strat_acts[r] == 1:
                    lh *= (1 - (unexpected * self.epsilon))
                    continue
                else:
                    unexpected += 1
                    lh *= unexpected * self.epsilon
                    continue
            # expected action is 0
            elif sequence > -2:
                sequence -= 1 
                if strat_acts[r] == 0:
                    lh *= (1 - (unexpected * self.epsilon))
                else:
                    unexpected += 1
                    lh *= unexpected * self.epsilon
                if sequence == -2:
                    if opponent_acts[r] == 1:
                        sequence = 4
                continue
            else:
                if opponent_acts[r] == 1:
                    sequence = 4

                # expected action is 0
                if opponent_acts[r - 1] == 0:
                    if strat_acts[r] == 0:
                        lh *= (1 - (unexpected * self.epsilon))
                        continue
                    else:
                        unexpected += 1
                        lh *= unexpected * self.epsilon
        return lh, unexpected

    """
    Starts with D,C,C and then defects if the opponent has cooperated 
    in the second and third move; otherwise, it plays TFT.
    """
    def prober(self, strat_acts, opponent_acts, prev_lh, unexpected):
        if prev_lh == 0:
            return 0, unexpected

        # first three moves always have to be D, C, C
        if len(strat_acts) == 1:
            if strat_acts[0] == 1:
                return prev_lh, unexpected
            else:
                return 0, unexpected
        if len(strat_acts) == 2:
            if strat_acts[1] == 0:
                return prev_lh, unexpected
            else:
                return 0, unexpected
        if len(strat_acts) == 3:
            if strat_acts[2] == 0:
                return prev_lh, unexpected
            else:
                return 0, unexpected
        
        # player is defector
        if opponent_acts[1] == 0 and opponent_acts[2] == 0:
            # expected move is 1
            if strat_acts[-1] == 1:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected
        
        # plays tit for tat
        else:
            # expected to copy opponent's mvoe 
            if strat_acts[-1] == opponent_acts[-2]:
                return prev_lh * (1 - (unexpected * self.epsilon)), unexpected
            else:
                unexpected += 1
                return prev_lh * unexpected * self.epsilon, unexpected