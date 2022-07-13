import numpy as np 
import matplotlib.pyplot as plt 
import statistics

def main():
    strat = "r"
    folder = "round_num"
    imagedir = "/home/jops/game_theory/results/graphs/" + folder + "/" + strat + "/"
    logdir = "/home/jops/game_theory/results/logs/" + folder + "/" + strat + "/"

    average_rewards = np.loadtxt(logdir + "average_rewards.npy")
    actions = np.loadtxt(logdir + "actions.npy")
    strategies = [line.rstrip() for line in open(logdir + "strategies.txt")]

    plt.plot(average_rewards, 'ro', markersize=3)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.savefig(imagedir + "rewards.png")
    plt.close()

    found_strats = []
    strat_rewards = []
    strat_actions = []
    ep = 0
    
    for strat in strategies:
        #print("ep:", ep)
        if strat not in found_strats:
            #print("found a new strategy: " + strat)
            found_strats.append(strat)
            strat_rewards.append(average_rewards[ep])
            for action in actions[ep * 100:(ep * 100) + 100]:
                strat_actions.append(action)
            newep = ep + 1
            for strat2 in strategies[ep + 1:]:
                if strat2 == strat:
                    strat_rewards.append(average_rewards[newep])
                    for action in actions[newep * 100:(newep * 100) + 100]:
                        #print("action:", action)
                        strat_actions.append(action)
                newep+=1

            #print("rewards: ", strat_rewards)
            plt.plot(strat_rewards, 'ro', markersize=3)
            plt.xlabel("instances of strategy")
            plt.ylabel("reward")
            plt.savefig(imagedir + "strat_" + strat + "_rewards.png")
            plt.close()

            num_end_eps = 5
            num_beg_eps = 5
            beg_action_dist = [0, 0]
            # print("strat actions beginning:", strat_actions[:100 * num_beg_eps])
            for action in strat_actions[: 100 * num_beg_eps]:
                beg_action_dist[int(action)]+=1
            end_action_dist = [0, 0]
            for action in strat_actions[-(100 * num_end_eps):]:
                end_action_dist[int(action)]+=1
            end_avg = statistics.mean(strat_rewards[num_end_eps:])
            beg_avg = statistics.mean(strat_rewards[:num_beg_eps])

            fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10,5))
            axs[0].bar([0, 1], beg_action_dist)
            axs[0].title.set_text("First Eps - Average Reward was " + str(beg_avg))
            axs[1].bar([0, 1],end_action_dist)
            axs[1].title.set_text("Last Eps - Average Reward was " + str(end_avg))

            axs[0].set_xlabel("Selected Action")
            axs[1].set_xlabel("Selected Action")
            axs[0].set_ylabel("Number of Picks")

            plt.savefig(imagedir + "strat_" + strat + "_actions.png")
            plt.close()
            strat_rewards = []
            strat_actions = []
        ep+=1

if __name__=="__main__":
    main()