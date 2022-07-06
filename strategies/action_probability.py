class ActionProbability:
    def cooperator(strat_acts, opponent_acts, unexpected_prob):
        return [1 - unexpected_prob, unexpected_prob]

    def defector(strat_acts, opponent_acts, unexpected_prob):
        return [unexpected_prob, 1 - unexpected_prob]

    def random(strat_acts, opponent_acts, unexpected_prob):
        return [0.5, 0.5]

    def tit_for_tat(strat_acts, opponent_acts, unexpected_prob):
        if len(strat_acts) == 0:
            return [1 - unexpected_prob, unexpected_prob]

        if opponent_acts[-1] == 0:
            return [1 - unexpected_prob, unexpected_prob]
        
        return [unexpected_prob, 1 - unexpected_prob]

    def grim_trigger(strat_acts, opponent_acts, unexpected_prob):
        if len(strat_acts) == 0:
            return [1 - unexpected_prob, unexpected_prob]
        
        if 1 in opponent_acts:
            return [unexpected_prob, 1 - unexpected_prob]

        return [1 - unexpected_prob, unexpected_prob]

    def tit_for_two_tats(strat_acts, opponent_acts, unexpected_prob):
        if len(strat_acts) < 2:
            return [1 - unexpected_prob, unexpected_prob]
        
        if opponent_acts[-1] == 1 and opponent_acts[-2] == 1:
            return [unexpected_prob, 1 - unexpected_prob]
        
        return [1 - unexpected_prob, unexpected_prob]

    def two_tits_for_tat(strat_acts, opponent_acts, unexpected_prob):
        if len(strat_acts) == 0:
            return [1 - unexpected_prob, unexpected_prob]
        
        if opponent_acts[-1] == 1:
            return [unexpected_prob, 1 - unexpected_prob]

        if len(opponent_acts) >= 2 and opponent_acts[-2] == 1:
            return [unexpected_prob, 1 - unexpected_prob]
        
        return [1 - unexpected_prob, unexpected_prob]

    def gradual(strat_acts, opponent_acts, unexpected_prob):
        if len(strat_acts) == 0:
            return [1 - unexpected_prob, unexpected_prob]
        
        num_defects = opponent_acts[0]
        if num_defects > 0:
            defects_left = num_defects
        else:
            defects_left = -2

        for r in range(1, len(strat_acts)):
            num_defects += opponent_acts[r]
            if defects_left > -1:
                defects_left -= 1
            elif defects_left == -1:
                defects_left -= 1
                if opponent_acts[r] == 1:
                    defects_left = num_defects
            else:
                if opponent_acts[r] == 1:
                    defects_left = num_defects
        
        if defects_left > 0:
            return [unexpected_prob, 1 - unexpected_prob]
        
        if defects_left > -2:
            return [1 - unexpected_prob, unexpected_prob]
        
        return [1 - unexpected_prob, unexpected_prob]

    def soft_majority(strat_acts, opponent_acts, unexpected_prob):
        if len(strat_acts) == 0:
            return [1 - unexpected_prob, unexpected_prob]
        
        if len(opponent_acts) >= 2 * sum(opponent_acts):
            return [1 - unexpected_prob, unexpected_prob]
        
        return [unexpected_prob, 1 - unexpected_prob]

    def hard_majority(strat_acts, opponent_acts, unexpected_prob):
        if len(strat_acts) == 0:
            return [unexpected_prob, 1 - unexpected_prob]
        
        if len(opponent_acts) > 2 * sum(opponent_acts):
            return [1 - unexpected_prob, unexpected_prob]
        
        return [unexpected_prob, 1 - unexpected_prob]

    def remorseful_prober(strat_acts, opponent_acts, unexpected_prob):
        if len(strat_acts) == 0:
            return [unexpected_prob, 1 - unexpected_prob]
        
    
        if len(strat_acts) > 5 and sum(opponent_acts[-5:]) == 5 and sum(strat_acts[-5:]) == 5:
            return [1 - unexpected_prob, unexpected_prob]
        
        if opponent_acts[-1] == 0:
            return [1 - unexpected_prob, unexpected_prob]

        return [unexpected_prob, 1 - unexpected_prob]

    def soft_grudger(strat_acts, opponent_acts, unexpected_prob):
        if len(strat_acts) == 0:
            return [1 - unexpected_prob, unexpected_prob]
        
        if opponent_acts[0] == 1:
            sequence = 4
        else:
            sequence = -2
        
        for r in range(1, len(strat_acts)):
            if sequence > -1:
                sequence -= 1
            elif sequence == -1:
                sequence -= 1
                if opponent_acts[r] == 1:
                    sequence = 4
            else:
                if opponent_acts[r] == 1:
                    sequence = 4
        
        if sequence > 0:
            return [unexpected_prob, 1 - unexpected_prob]
        
        if sequence > -2:
            return [1 - unexpected_prob, unexpected_prob]
        
        return [1 - unexpected_prob, unexpected_prob]

    def prober(strat_acts, opponent_acts, unexpected_prob):
        if len(strat_acts) == 0:
            return [1 - unexpected_prob, unexpected_prob]
        
        if len(strat_acts) == 1:
            return [unexpected_prob, 1 - unexpected_prob]
        
        if len(strat_acts) == 2:
            return [unexpected_prob, 1 - unexpected_prob]
        
        if opponent_acts[1] == 0 and opponent_acts[2] == 0:
            return [unexpected_prob, 1 - unexpected_prob]
        
        if opponent_acts[-1] == 0:
            return [1 - unexpected_prob, unexpected_prob]

        return [unexpected_prob, 1 - unexpected_prob]