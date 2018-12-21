# evolution_strategies
Idea: Trying out evolution strategies in some simple gym environment.

## What are evolution strategies?
Evolution strategies (ES) are an optimization technique that has been known for a long time. When developing an agent for a given environment, we might approach it by trying to find the best set of parameters to map the environment state to agent's actions (or distribution over actions) in order to maximize the reward. In policy optimization algorithms we often use backpropagation to improve the parameters little by little. However in ES the learning process is much simpler - we treat the environment and the agent together as a black box - the only things that we can see are the policy parameters that we input and the reward on the output of this black box. Our task is to change the parameters in order to increase the total reward. After initializing the weights, we add some noise to them a number of times and this way create a batch of noisy parameters. Then we evaluate each set of parameters and save its corresponding reward. Lastly, each set of parameters gets multiplied by their normalized reward and we add them all up together. The process is repeated multiple times.

You can read more about ES in [OpenAI's blog post](https://blog.openai.com/evolution-strategies/).

## Why do ES work?
When you think about what are actually ES doing, you'll likely come to a conclusion that it actually is taking small steps uphill, along with the gradient of some reward function. Parameters that score a higher reward, will receive larger weights. Making a step towards larger reward values is exactly the definition of making a step uphill with the gradient.

## Results
I used the [Mountain Car gym environment](https://gym.openai.com/envs/MountainCar-v0/) for evaluating the algorithm. It is a simple but not a trivial environment. The reward is between -200 and 0, and it depends on how fast does the player manage to reach the flag. If the number of time-steps is greater than 200, a reward of -200 is returned and the episode gets terminated.

I decided to take three different approaches to the amplitude of the noise:
* Constant - fixed at 100
* Constant decay by 1 point each 10 episodes, then fixed at 1
* Exponential decay - every 10 episodes the noise_scaling parameters gets multiplied by 0.9

Worth noting:
* Noise was uniformly generated (why? check [here](https://github.com/szymonWojdat/Cartpole))
* Because the environment returns only negative rewards, I decided to add 200 to each one of them because negative rewards are more difficult to normalize - smaller rewards would receive bigger weights without applying any tricks.
* Batch size = 100

Below are presented graphs of mean reward progression over time.

Constant noise scaling:

![](https://github.com/szymonWojdat/evolution_strategies/blob/master/graphs/avg_score_noise_100_const.png)

Constantly decreasing noise scaling:
![](https://github.com/szymonWojdat/evolution_strategies/blob/master/graphs/avg_score_noise_1_per_10_linear_decay.png)

Exponentially decreasing noise scaling:
![](https://github.com/szymonWojdat/evolution_strategies/blob/master/graphs/avg_score_noise_90_per_10t_decay.png)

## Interpretation
Decreasing noise scaling parameter dramatically decreases agents' variance and speeds up the learning process. The best performance is achieved when the noise gets linearly decreased, however its variance is higher and peak performance is achieved later than in case of exponentially decreasing the noise. I believe this is because the noise scaling factor stops at an arbitrary value of 1 instead of decreasing even further. However, this somewhat large (at least compared to the third approach) final value of the noise weight is probably still big enough to push the agent out of the local optima, that it gets stuck at when having the noise factor decreased by 0.9 every 10 episodes.

One might try to optimize the third approach by checking whether the agent has stopped improving and, if this is the case, increasing the noise ratio parameter by a bit to try pushing the agent out of the potential local optimum.
