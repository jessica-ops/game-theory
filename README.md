# game-theory
Repo from research in Xaq Lab (https://xaqlab.com/) over summer of 2022. Applies artificial intelligence to a game theory scenario in order to create rational agents whose mental model can be determined by outside observer. 

## Abstract (Explanation of Research) 
A guiding question within the field of Neuroscience is how the brain’s model of the world informs what actions it chooses to make. This internal model can be inferred from an observer’s standpoint by parameterizing different mental models and finding the one with the highest probability of eliciting the observed actions. To test this technique, we can reformulate a “brain” as a rational computer agent and control for its internal model. This has been done in a single-agent context, and this summer we established a foundation to extend this research into a multi-agent setting. We chose the Prisoner’s Dilemma as the setting, where the internal model of an agent playing the game can be characterized by a belief over their opponent’s actions.  
To begin this research, we focused on determining the mental model of a single rational agent. We accomplished this by optimizing agents against select iterated, discrete strategies established for playing the Prisoner’s Dilemma. Each strategy can be represented as a Markov Decision Process (MDP), meaning we can solve for the optimal policy using a value-iteration algorithm. The policies optimized against a single strategy then play against players they were not optimized against. In this setting, we introduce observer, who must determine which strategy the policy thinks it's playing against by only using current and prior observations. Establishing the foundations of this research will allow future research to expand to a truly multi-agent setting, in which both agents are fully rational with respect to their mental model of the world. 

## Tour of Repo
### Strategies 
The code for the discrete strategies is in this directory. It includes classes which represent the strategies themselves as well as classes which describe some of their behaviors (notably: action_probability.py which gives the probability of a certain strategy playing 0 or 1 for the next round). 

### Value Iteration
Most of the relevent code is in this directory. learn.py establishes the MDP for a given strategy and then uses a value iteration algorithm to create a Q table for a policy optimized against that strategy. determine_world.py introduces the outside observer and determines likelihood that an agent is a certain policy + graphs results. 

### RL
This directory encapsulates the first approach to creating rational agents in a game theory scenario: reinforcement learning. It used stable baselines 3 for the algorithm and the framework of an OpenAI Gym environment. There may be aspects of how the environment interacts with the strategies which are deprecated due to changes made after changign approaches. 

### Probabilistic Playing
An attempt to create a bayesian learning rational agent. strategy_inference.py is a practice at using information about the strategies to determine likelihood of which strategy one is playing against. This technique is used in bayesian_agent.py which attempts to use the likelihood of who it is playing against to best select actions. However, it is not sucessfully able to account for future rewards and this avenue was thus abandoned causing aspects of it to be half-finished. 
