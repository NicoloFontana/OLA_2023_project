import numpy as np
import matplotlib.pyplot as plt
import environment_4 as env
from Step3.gpts_and_price_optimizer import *
from Step3.gpucb_and_price_optimizer import *
from context_optimizer import *

np.random.seed(param.seed)
T = 100
n_features = len(param.feature_combos)

envs = {feature: env.Environment(feature) for feature in param.feature_combos}
opts = {feature: envs[feature].optimal for feature in param.feature_combos}
opt = sum(opts.values())

n_experiments = 3
ts_rewards_per_experiment = []
ucb_rewards_per_experiment = []

cumregret_ts = []
cumregret_ucb = []

cumreward_ts = []
cumreward_ucb = []

for e in range(0, n_experiments):
    # Create environment and learners
    ts_context_optimizer = ContextOptimizer(GPTSAndPriceOptimizer)
    ucb_context_optimizer = ContextOptimizer(GPUCBAndPriceOptimizer)

    for t in range(0, T):
        # Pull arms and update learners
        if t % 10 == 0:
            print(f"{t} of experiment {e}")

        # TS
        pulled_arms_per_feature = ts_context_optimizer.pull_arms()
        optimizer_update_input = {}
        for feature in pulled_arms_per_feature.keys():
            optimizer_update_input[feature] = pulled_arms_per_feature[feature] + envs[feature].round(*pulled_arms_per_feature[feature])
        ts_context_optimizer.update(optimizer_update_input)

        # UCB
        # pulled_arms_per_feature = ucb_context_optimizer.pull_arms()
        # optimizer_update_input = {}
        # for feature in pulled_arms_per_feature.keys():
        #     optimizer_update_input[feature] = pulled_arms_per_feature[feature] + envs[feature].round(*pulled_arms_per_feature[feature])
        # ucb_context_optimizer.update(optimizer_update_input)

    # Store collected rewards
    ts_rewards_per_experiment.append(ts_context_optimizer.collected_rewards)
    # ucb_rewards_per_experiment.append(ucb_context_optimizer.collected_rewards)

    cumregret_ts.append(np.cumsum(opt - ts_rewards_per_experiment[e]))
    # cumregret_ucb.append(np.cumsum(opt - ucb_rewards_per_experiment[e]))

    cumreward_ts.append(np.cumsum(ts_rewards_per_experiment[e]))
    # cumreward_ucb.append(np.cumsum(ucb_rewards_per_experiment[e]))

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
plt.plot(np.mean(cumregret_ts, axis=0), 'r')
plt.plot(np.mean(cumregret_ucb, axis=0), 'b')
plt.fill_between(range(T), np.mean(cumregret_ts, axis=0) - np.std(cumregret_ts, axis=0),
                 np.mean(cumregret_ts, axis=0) + np.std(cumregret_ts, axis=0), color="red", alpha=0.2)
# plt.fill_between(range(T), np.mean(cumregret_ucb, axis=0) - np.std(cumregret_ucb, axis=0),
#                  np.mean(cumregret_ucb, axis=0) + np.std(cumregret_ucb, axis=0), color="blue", alpha=0.2)
plt.legend(["TS", "UCB"])
plt.show()

plt.figure(1)
plt.xlabel("t")
plt.ylabel("Instantaneous Regret")
plt.plot(np.mean(opt - ts_rewards_per_experiment, axis=0), 'r')
# plt.plot(np.mean(opt - ucb_rewards_per_experiment, axis=0), 'b')
plt.fill_between(range(T),
                 np.mean(opt - ts_rewards_per_experiment, axis=0) - np.std(opt - ts_rewards_per_experiment, axis=0),
                 np.mean(opt - ts_rewards_per_experiment, axis=0) + np.std(opt - ts_rewards_per_experiment, axis=0),
                 color="red", alpha=0.2)
# plt.fill_between(range(T),
#                  np.mean(opt - ucb_rewards_per_experiment, axis=0) - np.std(opt - ucb_rewards_per_experiment, axis=0),
#                  np.mean(opt - ucb_rewards_per_experiment, axis=0) + np.std(opt - ucb_rewards_per_experiment, axis=0),
#                  color="blue", alpha=0.2)
plt.legend(["TS", "UCB"])
plt.show()

plt.figure(2)
plt.xlabel("t")
plt.ylabel("Cumulative Reward")
plt.plot(np.mean(cumreward_ts, axis=0), 'r')
plt.plot(np.mean(cumreward_ucb, axis=0), 'b')
plt.fill_between(range(T), np.mean(cumreward_ts, axis=0) - np.std(cumreward_ts, axis=0),
                 np.mean(cumreward_ts, axis=0) + np.std(cumreward_ts, axis=0), color="red", alpha=0.2)
# plt.fill_between(range(T), np.mean(cumreward_ucb, axis=0) - np.std(cumreward_ucb, axis=0),
#                  np.mean(cumreward_ucb, axis=0) + np.std(cumreward_ucb, axis=0), color="blue", alpha=0.2)
plt.legend(["TS", "UCB"])
plt.show()

plt.figure(3)
plt.xlabel("t")
plt.ylabel("Instantaneous Reward")
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(ucb_rewards_per_experiment, axis=0), 'b')
plt.fill_between(range(T), np.mean(ts_rewards_per_experiment, axis=0) - np.std(ts_rewards_per_experiment, axis=0),
                 np.mean(ts_rewards_per_experiment, axis=0) + np.std(ts_rewards_per_experiment, axis=0), color="red",
                 alpha=0.2)
# plt.fill_between(range(T), np.mean(ucb_rewards_per_experiment, axis=0) - np.std(ucb_rewards_per_experiment, axis=0),
#                  np.mean(ucb_rewards_per_experiment, axis=0) + np.std(ucb_rewards_per_experiment, axis=0), color="blue",
#                  alpha=0.2)
plt.legend(["TS", "UCB"])
plt.show()
