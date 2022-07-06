"""
CURRENTLY UNUSED 
"""
class FullLikelihood:
    def cooperator(strat_acts, opponent_acts, unexpected_prob, round_num):
        lh = 1
        for act in strat_acts:
            if act == 0:
                lh *= 1 - unexpected_prob
            else:
                lh *= unexpected_prob
        return lh
    
    def defector(strat_acts, opponent_acts, unexpected_prob, round_num):
        lh = 1
        for act in strat_acts:
            if act == 1:
                lh *= 1 - unexpected_prob
            else:
                lh *= unexpected_prob
        return lh
    
    def random(strat_acts, opponent_acts, unexpected_prob, round_num):
        return 1
    
    def tit_for_tat(strat_acts, opponent_acts, unexpected_prob, round_num):
        lh = 1

        if round_num == 0:
            if strat_acts[0] == 0:
                lh *= 1 - unexpected_prob
            else:
                lh *= unexpected_prob
        
        for i in range(1, len(strat_acts)):
            if strat_acts[i] == opponent_acts[i - 1]:
                lh *= 1 - unexpected_prob
            else:
                lh *= unexpected_prob
        
        return lh
    
    def grim_trigger(strat_acts, opponent_acts, unexpected_prob, round_num):
        lh = 1

        if round_num == 0:
            if strat_acts[0] == 0:
                lh *= 1 - unexpected_prob
            else:
                lh *= unexpected_prob
        
        grim = False
        if opponent_acts[0] == 1:
            grim = True
        
        for i in range(1, len(strat_acts)):
            if grim:
                if strat_acts[i] == 1:
                    lh *= 1 - unexpected_prob
                else:
                    lh *= unexpected_prob
            else:
                if strat_acts[i] == 0:
                    lh *= 1 - unexpected_prob
                else:
                    lh *= unexpected_prob
            
            if opponent_acts[i] == 1:
                grim = True
        
        return lh
    
    def tit_for_two_tats(strat_acts, opponent_acts, unexpected_prob, round_num):
        lh = 1

        if round_num == 0:
            if strat_acts[0] == 0:
                lh *= 1 - unexpected_prob
            else:
                lh *= unexpected_prob
            
            if strat_acts[1] == 0:
                lh *= 1 - unexpected_prob
            else:
                lh *= unexpected_prob
        
        if round_num == 1:
            if strat_acts[0] == 0:
                lh *= 1 - unexpected_prob
            else:
                lh *= unexpected_prob
        
        for i in range(2, len(strat_acts)):
            if opponent_acts[i - 1] == 1 and opponent_acts[i - 2] == 1:
                if strat_acts[i] == 1:
                    lh *= 1 - unexpected_prob
                else: 
                    lh *= unexpected_prob
            else:
                if strat_acts[i] == 0:
                    lh *= 1 - unexpected_prob
                else: 
                    lh *= unexpected_prob

        return lh
    
    def two_tits_for_tat(strat_acts, opponent_acts, unexpected_prob, round_num):
        lh = 1

        if round_num == 0:
            if strat_acts[0] == 0:
                lh *= 1 - unexpected_prob
            else:
                lh *= unexpected_prob

        for i in range(1, len(strat_acts)):
            if opponent_acts[i - 1] == 1:
                if strat_acts[i] == 1:
                    lh *= 1 - unexpected_prob
                else: 
                    lh *= unexpected_prob
            else:
                if i - 2 >= 0 and opponent_acts[i - 2] == 1:
                    if strat_acts[i] == 1:
                        lh *= 1 - unexpected_prob
                    else: 
                        lh *= unexpected_prob
                else:
                    if strat_acts[i] == 0:
                        lh *= 1 - unexpected_prob
                    else: 
                        lh *= unexpected_prob
        
        return lh
    
    def gradual(strat_acts, opponent_acts, unexpected_prob, round_num):
        lh = 1
        
        if round_num == 0:
            if strat_acts[0] == 0:
                lh *= (1 - unexpected_prob)
            else:
                lh *= unexpected_prob

        num_defects = opponent_acts[0]
        defects_left = -2
        if num_defects > 0:
            defects_left = num_defects

        for r in range(1, len(strat_acts)):
            num_defects += opponent_acts[r]
            if defects_left > 0: 
                if strat_acts[r] == 1:
                    defects_left -= 1
                    lh *= (1 - unexpected_prob)
                    continue
                else:
                    # print("UNEXPECTED -- should have defected because defects left = ", defects_left)
                    lh *= unexpected_prob
                    continue
            elif defects_left > -2:
                if strat_acts[r] == 0:
                    defects_left -= 1
                    lh *= (1 - (unexpected_prob))
                else: 
                    # print("UNEXPECTED -- should have cooperated because defects left = ", defects_left)
                    lh *= unexpected_prob
                if defects_left == -2:
                    if opponent_acts[r] == 1:
                        defects_left = num_defects
                continue
            else:
                if opponent_acts[r - 1] == 0:
                    if strat_acts[r] == 0:
                        lh *= (1 - unexpected_prob)
                    else:
                        # print("UNEXPECTED -- should have cooperated")
                        lh *= unexpected_prob
                if opponent_acts[r] == 1:
                    defects_left = num_defects
        return lh
    
    def soft_majority(strat_acts, opponent_acts, unexpected_prob, round_num):
        lh = 1
        
        if round_num == 0:
            if strat_acts[0] == 0:
                lh *= (1 - unexpected_prob)
            else:
                lh *= unexpected_prob
        
        for i in range(1, len(strat_acts)):
            if len(opponent_acts[:-i]) >= 2 * sum(opponent_acts[:-i]): 
                if strat_acts[i] == 0:
                    lh *= 1 - unexpected_prob
                else:
                    lh *= unexpected_prob
            else:
                if strat_acts[i] == 1:
                    lh *= 1 - unexpected_prob
                else:
                    lh *= unexpected_prob
        
        return lh

    def hard_majority(strat_acts, opponent_acts, unexpected_prob, round_num):
        lh = 1
        
        if round_num == 0:
            if strat_acts[0] == 1:
                lh *= (1 - unexpected_prob)
            else:
                lh *= unexpected_prob
        
        for i in range(1, len(strat_acts)):
            if len(opponent_acts[:-i]) > 2 * sum(opponent_acts[:-i]): 
                if strat_acts[i] == 0:
                    lh *= 1 - unexpected_prob
                else:
                    lh *= unexpected_prob
            else:
                if strat_acts[i] == 1:
                    lh *= 1 - unexpected_prob
                else:
                    lh *= unexpected_prob
        
        return lh

    def remorseful_prober(strat_acts, opponent_acts, unexpected_prob, round_num):
        lh = 1
        
        if round_num == 0:
            if strat_acts[0] == 0:
                lh *= (1 - unexpected_prob)
            else:
                lh *= unexpected_prob
        
        for i in range(1, len(strat_acts)):
            if i > 5 and sum(opponent_acts[-(i - 6): -i]):
                if strat_acts[0] == 0:
                    lh *= (1 - unexpected_prob)
                else:
                    lh *= unexpected_prob
            else: 
                if opponent_acts[i - 1] == 0: 
                    if strat_acts[0] == 0:
                        lh *= (1 - unexpected_prob)
                    else:
                        lh *= unexpected_prob
                else: 
                    if strat_acts[0] == 1:
                        lh *= (1 - unexpected_prob)
                    else:
                        lh *= unexpected_prob
        
        return lh

    def soft_grudger(strat_acts, opponent_acts, unexpected_prob, round_num):
        lh = 1
        
        if round_num == 0:
            if strat_acts[0] == 0:
                lh *= (1 - unexpected_prob)
            else:
                lh *= unexpected_prob
        
        if opponent_acts[0] == 1:
            sequence = 4
        else:
            sequence = -2
        
        for r in range(1, len(strat_acts)):
            if sequence > 0:
                if strat_acts[r] == 1:
                    sequence -= 1 
                    lh *= (1 - unexpected_prob)
                    continue
                else:
                    lh *= unexpected_prob
                    continue
            elif sequence > -2:
                if strat_acts[r] == 0:
                    sequence -= 1 
                    lh *= (1 - unexpected_prob)
                else:
                    lh *= unexpected_prob
                if sequence == -2:
                    if opponent_acts[r] == 1:
                        sequence = 4
                continue
            else:
                if opponent_acts[r] == 1:
                    sequence = 4

                if opponent_acts[r - 1] == 0:
                    if strat_acts[r] == 0:
                        lh *= (1 - unexpected_prob)
                        continue
                    else:
                        lh *= unexpected_prob
        return lh

    def prober(strat_acts, opponent_acts, unexpected_prob, round_num):
        defector = False
        tft = False
        start = 0
        if round_num == 0:
            if strat_acts[0] != 1 or strat_acts[1] != 0 or strat_acts[2] != 0:
                return 0
            start = 3
            if opponent_acts[1] == 0 and opponent_acts[2] == 0:
                defector = True
            else:
                tft = True
        elif round_num == 1:
            if strat_acts[0] != 0 or strat_acts[1] != 0:
                return 0
            start = 2
            if opponent_acts[0] == 0 and opponent_acts[1] == 0:
                defector = True
            else:
                tft = True
        elif round_num == 2:
            if strat_acts[0] != 0:
                return 0
            start = 1
        
        defector_lh = 0
        if not tft:
            defector_lh = 1
            for i in range(start, len(strat_acts)):
                if strat_acts[i] == 1:
                    defector_lh *= 1 - unexpected_prob
                else:
                    defector_lh *= unexpected_prob
        tft_lh = 0
        if not defector:
            tft_lh = 1
            for i in range(start, len(strat_acts)):
                if i - 1 > 0 and opponent_acts[i - 1] == 0:
                    if strat_acts[i] == 0:
                        tft_lh *= 1 - unexpected_prob
                    else:
                        tft_lh *= unexpected_prob
                else:
                    if strat_acts[i] == 1:
                        tft_lh *= 1 - unexpected_prob
                    else:
                        tft_lh *= unexpected_prob
        
        return max(defector_lh, tft_lh)