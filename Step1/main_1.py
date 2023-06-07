import numpy as np
import matplotlib.pyplot as plt
import ts_learner as ts
import ucb_learner as ucb
import environment_1 as env


T = 365

class_id = 1
env = env.Environment(class_id)
opt = env.optimal
n_arms = env.n_arms

n_experiments = 100
ts_rewards_per_experiment = []
ucb_rewards_per_experiment = []

cumregret_ts = []
cumregret_ucb = []

cumreward_ts = []
cumreward_ucb = []

for e in range (0,n_experiments):
    # Create environment and learners
    ts_learner = ts.TSLearner(n_arms=n_arms)
    ucb_learner = ucb.UCBLearner(n_arms=n_arms)

    for t in range (0,T):
        # Pull arms and update learners
        # Thompson sampling
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        # UCB
        pulled_arm = ucb_learner.pull_arm()
        reward = env.round(pulled_arm)
        ucb_learner.update(pulled_arm, reward)
    # Store collected rewards
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)

    cumregret_ts.append(np.cumsum(opt - ts_rewards_per_experiment[e]))
    cumregret_ucb.append(np.cumsum(opt - ucb_rewards_per_experiment[e]))

    cumreward_ts.append(np.cumsum(ts_rewards_per_experiment[e]))
    cumreward_ucb.append(np.cumsum(ucb_rewards_per_experiment[e]))


plt.figure(0)
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
plt.plot(np.mean(cumregret_ts, axis=0), 'r')
plt.plot(np.mean(cumregret_ucb, axis=0), 'b')
plt.fill_between(range(T), np.mean(cumregret_ts, axis=0) - np.std(cumregret_ts, axis=0), np.mean(cumregret_ts, axis=0) + np.std(cumregret_ts, axis=0), color = "red", alpha = 0.2)
plt.fill_between(range(T), np.mean(cumregret_ucb, axis=0) - np.std(cumregret_ucb, axis=0), np.mean(cumregret_ucb, axis=0) + np.std(cumregret_ucb, axis=0), color = "blue", alpha = 0.2)
plt.legend(["TS","UCB"])
plt.show()

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Instantaneous Regret")
plt.plot(np.mean(opt - ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(opt - ucb_rewards_per_experiment, axis=0), 'b')
plt.fill_between(range(T), np.mean(opt - ts_rewards_per_experiment, axis=0) - np.std(opt - ts_rewards_per_experiment, axis=0), np.mean(opt - ts_rewards_per_experiment, axis=0) + np.std(opt - ts_rewards_per_experiment, axis=0), color = "red", alpha = 0.2)
plt.fill_between(range(T), np.mean(opt - ucb_rewards_per_experiment, axis=0) - np.std(opt - ucb_rewards_per_experiment, axis=0), np.mean(opt - ucb_rewards_per_experiment, axis=0) + np.std(opt - ucb_rewards_per_experiment, axis=0), color = "blue", alpha = 0.2)
plt.legend(["TS","UCB"])
plt.show()

plt.figure(2)
plt.xlabel("t")
plt.ylabel("Cumulative Reward")
plt.plot(np.mean(cumreward_ts, axis=0), 'r')
plt.plot(np.mean(cumreward_ucb, axis=0), 'b')
plt.fill_between(range(T), np.mean(cumreward_ts, axis=0) - np.std(cumreward_ts, axis=0), np.mean(cumreward_ts, axis=0) + np.std(cumreward_ts, axis=0), color = "red", alpha = 0.2)
plt.fill_between(range(T), np.mean(cumreward_ucb, axis=0) - np.std(cumreward_ucb, axis=0), np.mean(cumreward_ucb, axis=0) + np.std(cumreward_ucb, axis=0), color = "blue", alpha = 0.2)
plt.legend(["TS","UCB"])
plt.show()

plt.figure(3)
plt.xlabel("t")
plt.ylabel("Instantaneous Reward")
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(ucb_rewards_per_experiment, axis=0), 'b')
plt.fill_between(range(T), np.mean(ts_rewards_per_experiment, axis=0) - np.std(ts_rewards_per_experiment, axis=0), np.mean(ts_rewards_per_experiment, axis=0) + np.std(ts_rewards_per_experiment, axis=0), color = "red", alpha = 0.2)
plt.fill_between(range(T), np.mean(ucb_rewards_per_experiment, axis=0) - np.std(ucb_rewards_per_experiment, axis=0), np.mean(ucb_rewards_per_experiment, axis=0) + np.std(ucb_rewards_per_experiment, axis=0), color = "blue", alpha = 0.2)
plt.legend(["TS","UCB"])
plt.show()