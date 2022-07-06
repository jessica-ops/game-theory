from strategies.strategy import *
import numpy as np
from stable_baselines3 import PPO

models = ["memory_model", "diff_rewards", "strat_no_name", "opponent_agent", "opponent_agent2", "strat_one_name"]
strat_names = ["Cooperater", "Defector", "Random", "TitForTat", "GrimTrigger", "Pavlov", "TitForTwoTats", "TwoTitsForTats", "Gradual", 
"SoftMajority", "HardMajority", "NaiveProber", "RemorsefulProber", "SoftGrudger", "Prober"]
strats = [Cooperater(0), Defector(0), Random([.5, .5]), TitForTat(0), GrimTrigger(0), Pavlov(0), TitForTwoTats(0), TwoTitsForTats(0), 
Gradual(0), SoftMajority(0), HardMajority(0),  RemorsefulProber(0), SoftGrudger(0), Prober(0)]
file = open("/home/jops/game_theory/results/logs/results.txt", "w")
rounds = 100
games = 10
payoff = [[(10, 10), (0, 20)], [(20, 0), (5, 5)]]
model_path = "/home/jops/game_theory/results/models/"

def agents_vs_agents(verbose):
    file.write("-------------------------------------------------------------\n")
    file.write("\nAGENTS VS AGENTS \n")
    results = np.zeros((len(models),))
    for i in range(len(models)):
        for j in range(i + 1, len(models)): 
            model1 = PPO.load(model_path + models[i])
            model2 = PPO.load(model_path + models[j])
            model1_tally = 0
            model2_tally = 0
            file.write("\n" + models[i] + " vs " + models[j] + "\n")
            for x in range(games):
                if verbose:
                    file.write("\nGAME #" + str(x) + "\n")
                model1_total = 0
                model2_total = 0
                model1_acts = np.zeros(shape = (10,), dtype=np.int32)
                model2_acts = np.zeros(shape = (10,), dtype=np.int32)
                for y in range(rounds): 
                    action1, _states = model1.predict({'opponent_action': model2_acts, 'agent_action': model1_acts})
                    if models[j] == "strat_one_name":
                        action2, _states = model2.predict({'opponent_action': model1_acts, 'agent_action': model2_acts, 'opponent_name': len(strats)})
                    else: 
                        action2, _states = model2.predict({'opponent_action': model1_acts, 'agent_action': model2_acts})
                    model1_pay, model2_pay = payoff[action1][action2]
                    model1_total += model1_pay
                    model2_total += model2_pay
                    model1_acts[:-1] = model1_acts[1:]
                    model1_acts[-1] = action1 + 1
                    model2_acts[:-1] = model2_acts[1:]
                    model2_acts[-1] = action2 + 1
                    if verbose:
                        file.write(str(action1) + " : " + str(model1_pay) + "    " + str(action2) + " : " + str(model2_pay) + "\n")
                
                if model1_total > model2_total: 
                    if verbose:
                        file.write(models[i] + " won this game with total " + str(model1_total) + " vs " + str(model2_total) + "\n")
                    model1_tally += 1
                elif model2_total > model1_total: 
                    if verbose: 
                        file.write(models[j] + " won this game with total " + str(model2_total) + " vs " + str(model1_total) + "\n")
                    model2_tally += 1
                else:
                    if verbose: 
                        file.write("there has been a tie! with total " + str(model1_total) + "\n")
            
            if model1_tally > model2_tally: 
                file.write(models[i] + " VICTORIOUS OVER " + models[j] + "\n")
                results[i]+=1
            elif model2_tally > model1_tally: 
                file.write(models[j] + " VICTORIOUS OVER " + models[i] + "\n")
                results[j]+=1
            else: 
                file.write("TIE BETWEEN " + models[i] + " AND " + models[j] + "\n")
                results[i]+=1
                results[j]+=1

    file.write("-------------------------------------------------------------\n")
    file.write("FINAL RESULTS: \n")
    for i in range(len(models)):
        file.write("Model " + models[i] + " won " + str(results[i]) + " times.\n")


def agents_vs_strats(verbose):
    file.write("-------------------------------------------------------------\n")
    file.write("\nAGENTS VS STRATS \n")
    model_results = np.zeros((len(models), ))
    strat_results = np.zeros((len(strat_names), ))
    for i in range(len(models)):
        for j in range(len(strats)): 
            model1 = PPO.load(model_path + models[i])
            model2 = strats[j]
            model1_tally = 0
            model2_tally = 0
            file.write("\n" + models[i] + " vs " + strat_names[j] + "\n")
            for x in range(games):
                if verbose: 
                    file.write("\nGAME #" + str(x) + "\n")
                model1_total = 0
                model2_total = 0
                model1_acts = np.zeros(shape = (10,), dtype=np.int32)
                model2_acts = np.zeros(shape = (10,), dtype=np.int32)
                model1_history = np.empty(shape = (rounds, 2), dtype=np.int32)
                model2_history = np.empty(shape = (rounds, 2), dtype=np.int32)
                for y in range(rounds): 
                    if models[i] == "strat_one_name":
                        action1, _states = model1.predict({'opponent_action': model2_acts, 'agent_action': model1_acts, 'opponent_name': len(strats)})
                    else: 
                        action1, _states = model1.predict({'opponent_action': model2_acts, 'agent_action': model1_acts})
                    action2 = model2.choose(model2_history, model1_history, y)
                    model1_pay, model2_pay = payoff[action1][action2]
                    model1_total += model1_pay
                    model2_total += model2_pay
                    model1_acts[:-1] = model1_acts[1:]
                    model1_acts[-1] = action1 + 1
                    model2_acts[:-1] = model2_acts[1:]
                    model2_acts[-1] = action2 + 1
                    model1_history[y] = [action1, model1_pay]
                    model2_history[y] = [action2, model2_pay]
                    if verbose:
                        file.write(str(action1) + " : " + str(model1_pay) + "    " + str(action2) + " : " + str(model2_pay) + "\n")
                
                if model1_total > model2_total: 
                    if verbose: 
                        file.write(models[i] + " won this game with total " + str(model1_total) + " vs " + str(model2_total) + "\n")
                    model1_tally += 1
                elif model2_total > model1_total:
                    if verbose: 
                        file.write(strat_names[j] + " won this game with total " + str(model2_total) + " vs " + str(model1_total) + "\n")
                    model2_tally += 1
                else:
                    if verbose: 
                        file.write("there has been a tie! with total " + str(model1_total) + "\n")
            
            if model1_tally > model2_tally: 
                file.write(models[i] + " VICTORIOUS OVER " + strat_names[j] + "\n")
                model_results[i]+=1
            elif model2_tally > model1_tally: 
                file.write(strat_names[j] + " VICTORIOUS OVER " + models[i] + "\n")
                strat_results[j]+=1
            else: 
                file.write("TIE BETWEEN " + models[i] + " AND " + strat_names[j] + "\n")
                model_results[i]+=1
                strat_results[j]+=1

    file.write("-------------------------------------------------------------\n")
    file.write("FINAL RESULTS: \n")
    for i in range(len(models)):
        file.write("Model " + models[i] + " won " + str(model_results[i]) + " times.\n")
    for j in range(len(strat_names)):
        file.write("Model " + strat_names[j] + " won " + str(strat_results[j]) + " times.\n")

# STRATEGIES VS STRATEGIES
def strats_vs_strats(verbose):
    file.write("-------------------------------------------------------------\n")
    file.write("\nSTRATS VS STRATS \n")
    results = np.zeros((len(strats), ))
    for i in range(len(strats)):
        for j in range(i + 1, len(strats)): 
            model1 = strats[i]
            model2 = strats[j]
            model1_history = np.empty(shape = (rounds, 2), dtype=np.int32)
            model2_history = np.empty(shape = (rounds, 2), dtype=np.int32)
            model1_tally = 0
            model2_tally = 0
            file.write("\n" + strat_names[i] + " vs " + strat_names[j] + "\n")
            for x in range(games):
                if verbose: 
                    file.write("\nGAME #" + str(x) + "\n")
                model1_total = 0
                model2_total = 0
                for y in range(rounds): 
                    action1 = model1.choose(model1_history, model2_history, y)
                    action2 = model2.choose(model2_history, model1_history, y)
                    model1_pay, model2_pay = payoff[action1][action2]
                    model1_total += model1_pay
                    model2_total += model2_pay
                    model1_history[y] = [action1, model1_pay]
                    model2_history[y] = [action2, model2_pay]
                    if verbose:
                        file.write(str(action1) + " : " + str(model1_pay) + "    " + str(action2) + " : " + str(model2_pay) + "\n")
                
                if model1_total > model2_total: 
                    if verbose: 
                        file.write(strat_names[i] + " won this game with total " + str(model1_total) + " vs " + str(model2_total) + "\n")
                    model1_tally += 1
                elif model2_total > model1_total: 
                    if verbose: 
                        file.write(strat_names[j] + " won this game with total " + str(model2_total) + " vs " + str(model1_total) + "\n")
                    model2_tally += 1
                else:
                    if verbose: 
                        file.write("there has been a tie! with total " + str(model1_total) + "\n")
            
            if model1_tally > model2_tally: 
                file.write(strat_names[i] + " VICTORIOUS OVER " + strat_names[j] + "\n")
                results[i]+=1
            elif model2_tally > model1_tally: 
                file.write(strat_names[j] + " VICTORIOUS OVER " + strat_names[i] + "\n")
                results[j]+=1
            else: 
                file.write("TIE BETWEEN " + strat_names[i] + " AND " + strat_names[j] + "\n")
                results[i]+=1
                results[j]+=1
    
    file.write("-------------------------------------------------------------\n")
    file.write("FINAL RESULTS: \n")
    for j in range(len(strat_names)):
        file.write("Model " + strat_names[j] + " won " + str(results[j]) + " times.\n")

def main():
    agents_vs_agents(False)
    agents_vs_strats(False)
    # strats_vs_strats(False)

if __name__ == "__main__":
    main()