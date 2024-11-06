# Gambler RL

An implementation of Q-learning with an explicit Q-table and a neural network for solving the Gambler's problem modeled as a Markov decision process. Both methods converge to an approximately optimal policy after 500,000 simulated episodes. Notably, while the full Q-table contains around 5,000 entries, a neural network with less than 2,000 parameters is sufficient to closely approximate the Q-function and optimal policy.

For an in-depth analysis of the Gambler's problem see https://borundev.medium.com/gamblers-problem-when-inaction-is-infact-optimal-1d8348b69c4f