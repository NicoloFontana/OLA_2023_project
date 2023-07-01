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

pulled_arms_ucb = []
pulled_arms_ts = []

for e in range(0, n_experiments):
    # Create environment and learners
    ts_learner = ts.TSLearner(n_arms=n_arms)
    ucb_learner = ucb.UCBLearner(n_arms=n_arms)
    pulled_arms_ucb_exp = []
    pulled_arms_ts_exp = []

    for t in range(0, T):
        # Pull arms and update learners
        # Thompson sampling
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)
        pulled_arms_ts_exp.append(pulled_arm + 1)

        # UCB
        pulled_arm = ucb_learner.pull_arm()
        reward = env.round(pulled_arm)
        ucb_learner.update(pulled_arm, reward)
        pulled_arms_ucb_exp.append(pulled_arm + 1)
    # Store collected rewards
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)

    cumregret_ts.append(np.cumsum(opt - ts_rewards_per_experiment[e]))
    cumregret_ucb.append(np.cumsum(opt - ucb_rewards_per_experiment[e]))

    cumreward_ts.append(np.cumsum(ts_rewards_per_experiment[e]))
    cumreward_ucb.append(np.cumsum(ucb_rewards_per_experiment[e]))
    pulled_arms_ucb.append(pulled_arms_ucb_exp)
    pulled_arms_ts.append(pulled_arms_ts_exp)

plt.figure(0)
plt.title(f"Step1 - Class {class_id}")
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
plt.plot(np.mean(cumregret_ts, axis=0), 'r')
plt.plot(np.mean(cumregret_ucb, axis=0), 'b')
plt.fill_between(range(T), np.mean(cumregret_ts, axis=0) - np.std(cumregret_ts, axis=0),
                 np.mean(cumregret_ts, axis=0) + np.std(cumregret_ts, axis=0), color="red", alpha=0.2)
plt.fill_between(range(T), np.mean(cumregret_ucb, axis=0) - np.std(cumregret_ucb, axis=0),
                 np.mean(cumregret_ucb, axis=0) + np.std(cumregret_ucb, axis=0), color="blue", alpha=0.2)
plt.legend(["TS", "UCB"])
plt.show()

plt.figure(1)
plt.title(f"Step1 - Class {class_id}")
plt.xlabel("t")
plt.ylabel("Instantaneous Regret")
plt.plot(np.mean(opt - ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(opt - ucb_rewards_per_experiment, axis=0), 'b')
plt.fill_between(range(T),
                 np.mean(opt - ts_rewards_per_experiment, axis=0) - np.std(opt - ts_rewards_per_experiment, axis=0),
                 np.mean(opt - ts_rewards_per_experiment, axis=0) + np.std(opt - ts_rewards_per_experiment, axis=0),
                 color="red", alpha=0.2)
plt.fill_between(range(T),
                 np.mean(opt - ucb_rewards_per_experiment, axis=0) - np.std(opt - ucb_rewards_per_experiment, axis=0),
                 np.mean(opt - ucb_rewards_per_experiment, axis=0) + np.std(opt - ucb_rewards_per_experiment, axis=0),
                 color="blue", alpha=0.2)
plt.legend(["TS", "UCB"])
plt.show()

plt.figure(2)
plt.title(f"Step1 - Class {class_id}")
plt.xlabel("t")
plt.ylabel("Cumulative Reward")
plt.plot(np.mean(cumreward_ts, axis=0), 'r')
plt.plot(np.mean(cumreward_ucb, axis=0), 'b')
plt.fill_between(range(T), np.mean(cumreward_ts, axis=0) - np.std(cumreward_ts, axis=0),
                 np.mean(cumreward_ts, axis=0) + np.std(cumreward_ts, axis=0), color="red", alpha=0.2)
plt.fill_between(range(T), np.mean(cumreward_ucb, axis=0) - np.std(cumreward_ucb, axis=0),
                 np.mean(cumreward_ucb, axis=0) + np.std(cumreward_ucb, axis=0), color="blue", alpha=0.2)
plt.legend(["TS", "UCB"])
plt.show()

plt.figure(3)
plt.title(f"Step1 - Class {class_id}")
plt.xlabel("t")
plt.ylabel("Instantaneous Reward")
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(ucb_rewards_per_experiment, axis=0), 'b')
plt.fill_between(range(T), np.mean(ts_rewards_per_experiment, axis=0) - np.std(ts_rewards_per_experiment, axis=0),
                 np.mean(ts_rewards_per_experiment, axis=0) + np.std(ts_rewards_per_experiment, axis=0), color="red",
                 alpha=0.2)
plt.fill_between(range(T), np.mean(ucb_rewards_per_experiment, axis=0) - np.std(ucb_rewards_per_experiment, axis=0),
                 np.mean(ucb_rewards_per_experiment, axis=0) + np.std(ucb_rewards_per_experiment, axis=0), color="blue",
                 alpha=0.2)
plt.legend(["TS", "UCB"])
plt.show()

plt.figure(4)
plt.title("Pulled arms UCB")
plt.xlabel("t")
plt.ylabel("Arm")
plt.plot(pulled_arms_ucb[-1], marker='o', markerfacecolor='None', markeredgecolor='b', markersize=4, linestyle='None')
plt.show()

plt.figure(5)
plt.title("Pulled arms TS")
plt.xlabel("t")
plt.ylabel("Arm")
plt.plot(pulled_arms_ts[-1], marker='o', markerfacecolor='None', markeredgecolor='r', markersize=4, linestyle='None')
plt.show()

mean_pulls_per_arm_ts = np.zeros(n_arms)
mean_pulls_per_arm_ucb = np.zeros(n_arms)
for e in range(n_experiments):
    for arm in range(n_arms):
        mean_pulls_per_arm_ts[arm] += np.sum(np.array(pulled_arms_ts[e]) == arm + 1) / n_experiments
        mean_pulls_per_arm_ucb[arm] += np.sum(np.array(pulled_arms_ucb[e]) == arm + 1) / n_experiments

barWidth = 0.4
br1 = np.arange(n_arms)
br2 = [x + barWidth for x in br1]
plt.figure(6)
plt.title("Pulled arms: UCB vs TS")
plt.bar(br1, mean_pulls_per_arm_ts, color='r', width=barWidth)
plt.bar(br2, mean_pulls_per_arm_ucb, color='b', width=barWidth)
plt.ylabel("Mean number of pulls")
plt.xlabel("Arm")
plt.xticks([r + barWidth for r in range(n_arms)], np.array(range(n_arms)) + 1)
plt.legend(["TS", "UCB"])
plt.show()
