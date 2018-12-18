import gym
import numpy as np
import matplotlib.pyplot as plt


def softmax(v):
	return np.exp(v)/np.sum(np.exp(v))


# let's sample both params uniformly from [-1, 1], takes state and params, outputs action probabilities
def policy_fn(state, params):
	y = np.matmul(state, params)
	return softmax(y)


def inject_noise(params, scaling):
	return params + (np.random.random(params.shape) * 2 - 1) * scaling


def sample(probs):
	return np.random.multinomial(1, probs).argmax()


def run_episode(env, get_action, render=False):
	state = env.reset()
	done = False
	total_reward = 0
	while not done:
		if render:
			env.render()
		action = get_action(state)
		state, reward, done, info = env.step(action)
		total_reward += reward
	return total_reward


def main():
	epochs = 2000
	batch_size = 100
	noise_scaling = 100

	env = gym.make('MountainCar-v0')
	memo = []

	w = np.zeros([2, 3], dtype=np.float64)  # weights initialization
	for i in range(epochs):
		parameters = []
		rewards = []
		for _ in range(batch_size):
			w_noisy = inject_noise(w, noise_scaling)
			parameters.append(w_noisy)
			rewards.append(run_episode(env, lambda state: sample(policy_fn(state, w_noisy))))

		rewards = np.array(rewards, dtype=float) + 200.
		rewards_norm = (rewards/sum(rewards))[:, np.newaxis, np.newaxis]
		w = np.sum(parameters * rewards_norm, axis=0)

		avg_reward = sum(rewards)/len(rewards)
		memo.append(avg_reward)
		if i % 100 == 0:
			print('Timestep #{}:\t avg = {} \tmax = {}'.format(i, avg_reward, max(rewards)))

	env.close()
	plt.plot(memo)
	plt.show()


if __name__ == '__main__':
	main()
