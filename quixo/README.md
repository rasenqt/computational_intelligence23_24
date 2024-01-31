# Deep Reinforcement Learning implementation of Quixo
We took into account the game.py file with the version related to the commit 0edba49, in order to be row-column (as agreed with teaching assistant)


# Introduction
Quixo is a two-player abstract strategy board game where players take turns shifting rows or columns in an attempt to align five of their pieces in a row. In this project, we explore the implementation of Deep Reinforcement Learning (DRL) techniques to develop an intelligent agent capable of playing Quixo.


# How is the code structured
**hyperparams_selection_genetic.ipynb** => used in order to generate the hyperparameters of the network 

**training_phase.ipynb** => used in order to train an agent 

**testing_phase.ipynb** => We proposed a testing environment where there's the possibility to try our final trained agent both starting as first or second, thanks to the variable 'starts_first'. In case you would like to test the agent against more opponents, just plug them into the vector 'players' and for each game, there will be a different opponent

**trained_models** => here you could find the model parameters to feed into the Training_Complete structure along with a boolean value that specifies wether the model starts as first or second. The structure provides a function that let modify this value in order to start as second 

**game.py** => defines the Game, Player and QuixoNet (the NN used inside the Player) 


# Deep Reinforcement Learning
In our Deep Reinforcement Learning (DRL) approach for Quixo, we rely on the concept of modeling a Q-function through a neural network. .The essence of our methodology lies in the representation of the game state and the subsequent decision-making process orchestrated by this function. Specifically, we employ two separate models: one designated for the agent playing first and the other for the agent playing second. This deliberate separation stems from our observation that attempting to encapsulate all possible game configurations within a single model tends to result in suboptimal generalization.

Our algorithm is structured around the interaction between an agent (a neural network utilized for training) and the environment (which is represented by the game board). At its core, our agent's learning process involves iteratively updating its neural network parameters based on the rewards received, and the states resulting from its actions. These states are characterized by the actions taken by both our training agent and its opponent, with the latter's behavior potentially varying depending on the specific opponent model employed.

During the training phase, our agent plays a certain number of matches against a randomly behaving opponent. Through these matches, it gradually improves its strategy through reinforcement learning. By updating its neural network parameters due to the observed rewards and state transitions, the agent enhances its capabilities of outperforming its adversaries in diverse game scenarios.

Inside our Algorithm we make use of another neural network called TargetNet (some model for making prediction), which is utilized to stabilize the training process by providing a fixed target for computing the loss function. This network is periodically updated to mirror the current parameters of the main neural network, ensuring a more consistent learning signal. Additionally, the replay buffer stores past experiences (state-action-reward transitions) and samples them randomly during training. This technique improves sample efficiency and reduces the correlation between consecutive experiences, leading to more stable and effective learning


# GA to support DRL
In our implementation, we extend the application of the Genetic Algorithm (GA) to address the hyperparameter selection process for both versions of the Quixo model, each specialized for different starting turns. This extension involves running the GA independently for each model's turn version, allowing us to define a sort of baseline for the hyperparameters to start with.

Similar to the previous description, for each turn version of the model, we generate a population of individuals, with each individual representing a unique configuration of hyperparameters. The hyperparameters considered include:

- num_matches: Number of matches
- max_dim_replay_buff: Maximum dimension of the replay buffer
- time_to_update: Time interval for updating the neural network
- gamma: Discount factor
- batch_size: Size of the training batch.

We enforce constraints on the values of these hyperparameters to ensure they fall within predefined ranges, as specified by cell_max_values.

The fitness function for each individual is evaluated based on the accuracy achieved by training the corresponding model with the specified hyperparameters. By optimizing for accuracy, the GA aims to identify hyperparameter configurations that result in superior performance for each turn version of the Quixo model.

Given the computational demands of this solution, we maintain the limitation of 50 generations per turn version. However, this parameter can be adjusted based on available computational resources, allowing for more extensive exploration of the hyperparameter space if desired. This approach ensures that each model's turn version starts its training with a set of hyperparameters that let it reach a decent accuracy at least at the beginning (by modifing them, we observed huge gaps in training, therefore it's really important to start with a good base)

Note: For each best solution (one for the models that starts first and one for the one that starts second), we then try different learning rates (in the GA we used 0.0001 as default one)


## Random Opponent vs QuixoNet / QuixoNet vs QuixoNet
In our experiments with opponent training strategies, we initially let our Quixo agent play against a random opponent. This served as a foundational training phase, allowing the agent to learn fundamental strategies and adapt to diverse game states. Once we achieved satisfactory performance with the trained models for both starting turns, we explored further training methodologies to enhance the agent's capabilities.

In order to refine the agent's strategies, we conducted additional training sessions where the trained agent for one starting turn (e.g., the first turn) competed against the agent specialized for the opposite starting turn (e.g., the second turn). During this training process, we kept the parameters of the opponent agent frozen. This iterative training approach aimed to leverage the strengths and weaknesses of each agent to foster mutual improvement.

Following this training iteration, we optimize the performance of the trained models by having them play against each other. Despite the expectation of observing improvements resulting from the iterative training process, our analysis revealed no remarkable signs of improvement in gameplay quality or by means of new strategies.

Consequently, we concluded that the training against a random opponent was enough in providing the necessary learning experiences for our Quixo agent. As such, we retained the model trained solely against the random opponent, as it demonstrated comparable or superior performance without the added complexity and computational burden associated with further training iterations against specialized opponents.



## Final Observations
We present a Deep Q-Learning model by leveraging GA for hyperparameters initialization, trained only against a random opponent. In this way, we highlight the importance of carefully evaluating training strategies and considering the trade-offs between complexity, computational resources, and performance gains in reinforcement learning tasks.
In our final assessment, we achieved significant success with our Quixo agent, reaching a more than acceptable accuracy (87-92%) by using a light neural networks (only 17k parameters per model turn version)

## Have fun
Made with ❤️ by [Michelangelo Caretto](https://github.com/rasenqt/computational_intelligence23_24) and [Silvio Chito](https://github.com/SilvioChito/computational_intelligence)
