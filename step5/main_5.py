import numpy as np
import matplotlib.pyplot as plt
import swucb_learner as swucb
import cusum_ucb_learner as cu_ucb
import ucb_learner as ucb
import environment_5 as env


T = 365
phase_size = int(T / 3)
window_size = int(T ** 0.5)

class_id = 2
env = env.Environment(class_id, T)
opt = np.array([env.get_opt(t) for t in range(T)])
n_arms = env.n_arms

n_experiments = 500
ucb_rewards_per_experiment = []
swucb_rewards_per_experiment = []
cusumucb_rewards_per_experiment = []

cumregret_ucb = []
cumregret_swucb = []
cumregret_cusumucb = []

cumreward_ucb = []
cumreward_swucb = []
cumreward_cusumucb = []

for e in range (0,n_experiments):
    # Create environment and learners
    ucb_learner = ucb.UCBLearner(n_arms=n_arms)
    swucb_learner = swucb.SWUCBLearner(n_arms=n_arms, window_size=window_size)
    cusum_ucb_learner = cu_ucb.CusumUCBLearner(n_arms)

    if e % 10 == 0:
        print(f"Experiment {e}")

    for t in range (0,T):
        # Pull arms and update learners
        # UCB
        pulled_arm = ucb_learner.pull_arm()
        reward = env.round(pulled_arm, t)
        ucb_learner.update(pulled_arm, reward)

        # SW-UCB
        pulled_arm = swucb_learner.pull_arm()
        reward = env.round(pulled_arm, t)
        swucb_learner.update(pulled_arm, reward)

        # Cusum-UCB
        pulled_arm = cusum_ucb_learner.pull_arm()
        reward = env.round(pulled_arm, t)
        cusum_ucb_learner.update(pulled_arm, reward)
    # Store collected rewards
    ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)
    swucb_rewards_per_experiment.append(swucb_learner.collected_rewards)
    cusumucb_rewards_per_experiment.append(cusum_ucb_learner.collected_rewards)

    cumregret_ucb.append(np.cumsum(opt - ucb_rewards_per_experiment[e]))
    cumregret_swucb.append(np.cumsum(opt - swucb_rewards_per_experiment[e]))
    cumregret_cusumucb.append(np.cumsum(opt - cusumucb_rewards_per_experiment[e]))

    cumreward_ucb.append(np.cumsum(ucb_rewards_per_experiment[e]))
    cumreward_swucb.append(np.cumsum(swucb_rewards_per_experiment[e]))
    cumreward_cusumucb.append(np.cumsum(cusumucb_rewards_per_experiment[e]))


plt.figure(0)
plt.title(f"Step1 - Class {class_id}")
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
plt.plot(np.mean(cumregret_ucb, axis=0), 'b')
plt.plot(np.mean(cumregret_swucb, axis=0), 'r')
plt.plot(np.mean(cumregret_cusumucb, axis=0), 'g')
plt.fill_between(range(T), np.mean(cumregret_ucb, axis=0) - np.std(cumregret_ucb, axis=0), np.mean(cumregret_ucb, axis=0) + np.std(cumregret_ucb, axis=0), color = "blue", alpha = 0.2)
plt.fill_between(range(T), np.mean(cumregret_swucb, axis=0) - np.std(cumregret_swucb, axis=0), np.mean(cumregret_swucb, axis=0) + np.std(cumregret_swucb, axis=0), color ="red", alpha = 0.2)
plt.fill_between(range(T), np.mean(cumregret_swucb, axis=0) - np.std(cumregret_cusumucb, axis=0), np.mean(cumregret_cusumucb, axis=0) + np.std(cumregret_cusumucb, axis=0), color ="green", alpha = 0.2)
plt.legend(["UCB", "SW-UCB", "CUSUM-UCB"])
plt.show()

plt.figure(1)
plt.title(f"Step1 - Class {class_id}")
plt.xlabel("t")
plt.ylabel("Instantaneous Regret")
plt.plot(np.mean(opt - ucb_rewards_per_experiment, axis=0), 'b')
plt.plot(np.mean(opt - swucb_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(opt - cusumucb_rewards_per_experiment, axis=0), 'g')
plt.fill_between(range(T), np.mean(opt - ucb_rewards_per_experiment, axis=0) - np.std(opt - ucb_rewards_per_experiment, axis=0), np.mean(opt - ucb_rewards_per_experiment, axis=0) + np.std(opt - ucb_rewards_per_experiment, axis=0), color = "blue", alpha = 0.2)
plt.fill_between(range(T), np.mean(opt - swucb_rewards_per_experiment, axis=0) - np.std(opt - swucb_rewards_per_experiment, axis=0), np.mean(opt - swucb_rewards_per_experiment, axis=0) + np.std(opt - swucb_rewards_per_experiment, axis=0), color ="red", alpha = 0.2)
plt.fill_between(range(T), np.mean(opt - cusumucb_rewards_per_experiment, axis=0) - np.std(opt - cusumucb_rewards_per_experiment, axis=0), np.mean(opt - cusumucb_rewards_per_experiment, axis=0) + np.std(opt - cusumucb_rewards_per_experiment, axis=0), color ="green", alpha = 0.2)
plt.legend(["UCB", "SW-UCB", "CUSUM-UCB"])
plt.show()

plt.figure(2)
plt.title(f"Step1 - Class {class_id}")
plt.xlabel("t")
plt.ylabel("Cumulative Reward")
plt.plot(np.mean(cumreward_ucb, axis=0), 'b')
plt.plot(np.mean(cumreward_swucb, axis=0), 'r')
plt.plot(np.mean(cumreward_cusumucb, axis=0), 'g')
plt.fill_between(range(T), np.mean(cumreward_ucb, axis=0) - np.std(cumreward_ucb, axis=0), np.mean(cumreward_ucb, axis=0) + np.std(cumreward_ucb, axis=0), color = "blue", alpha = 0.2)
plt.fill_between(range(T), np.mean(cumreward_swucb, axis=0) - np.std(cumreward_swucb, axis=0), np.mean(cumreward_swucb, axis=0) + np.std(cumreward_swucb, axis=0), color ="red", alpha = 0.2)
plt.fill_between(range(T), np.mean(cumreward_cusumucb, axis=0) - np.std(cumreward_cusumucb, axis=0), np.mean(cumreward_cusumucb, axis=0) + np.std(cumreward_cusumucb, axis=0), color ="green", alpha = 0.2)
plt.legend(["UCB", "SW-UCB", "CUSUM-UCB"])
plt.show()

plt.figure(3)
plt.title(f"Step1 - Class {class_id}")
plt.xlabel("t")
plt.ylabel("Instantaneous Reward")
plt.plot(opt, 'k--')
plt.plot(np.mean(ucb_rewards_per_experiment, axis=0), 'b')
plt.plot(np.mean(swucb_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(cusumucb_rewards_per_experiment, axis=0), 'g')
plt.fill_between(range(T), np.mean(ucb_rewards_per_experiment, axis=0) - np.std(ucb_rewards_per_experiment, axis=0), np.mean(ucb_rewards_per_experiment, axis=0) + np.std(ucb_rewards_per_experiment, axis=0), color = "blue", alpha = 0.2)
plt.fill_between(range(T), np.mean(swucb_rewards_per_experiment, axis=0) - np.std(swucb_rewards_per_experiment, axis=0), np.mean(swucb_rewards_per_experiment, axis=0) + np.std(swucb_rewards_per_experiment, axis=0), color ="red", alpha = 0.2)
plt.fill_between(range(T), np.mean(cusumucb_rewards_per_experiment, axis=0) - np.std(cusumucb_rewards_per_experiment, axis=0), np.mean(cusumucb_rewards_per_experiment, axis=0) + np.std(cusumucb_rewards_per_experiment, axis=0), color ="green", alpha = 0.2)
plt.legend(["Optimal", "UCB", "SW-UCB", "CUSUM-UCB"])
plt.show()
