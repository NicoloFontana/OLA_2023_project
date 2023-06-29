import numpy as np
import matplotlib.pyplot as plt
from Step3 import environment_3 as env
from Step3.gpts_and_price_optimizer import *
from Step3.gpucb_and_price_optimizer import *
from Step3.optimizer_learner import *

np.random.seed(param.seed)
T = 365
n_classes = 3

envs = [env.Environment(class_id) for class_id in range(1, n_classes+1)]
opts = [env.optimal for env in envs]
opt = sum(opts)
print(opts)
print(opt)

n_experiments = 10
ts_rewards_per_experiment = []
ucb_rewards_per_experiment = []

cumregret_ts = []
cumregret_ucb = []

cumreward_ts = []
cumreward_ucb = []

for e in range (0,n_experiments):
    # Create environment and learners
    gpts_and_price_optimizers = [GPTSAndPriceOptimizer(param.bids, param.prices) for class_id in range(1,n_classes+1)]
    gpucb_and_price_optimizers = [GPUCBAndPriceOptimizer(param.bids, param.prices) for class_id in range(1,n_classes+1)]

    for t in range (0,T):
        # Pull arms and update learners
        # Thompson sampling
        if t % 10 == 0:
            print(f"{t} of experiment {e}")
        for class_id in range(n_classes):
            pulled_arms = gpts_and_price_optimizers[class_id].pull_arms()
            pulled_bids_arm = pulled_arms[0]
            pulled_prices_arm = pulled_arms[1]
            round_reward = envs[class_id].round(pulled_bids_arm, pulled_prices_arm)
            gpts_and_price_optimizers[class_id].update(pulled_bids_arm, pulled_prices_arm, *round_reward)

            # UCB
            pulled_arms = gpucb_and_price_optimizers[class_id].pull_arms()
            pulled_bids_arm = pulled_arms[0]
            pulled_prices_arm = pulled_arms[1]
            round_reward = envs[class_id].round(pulled_bids_arm, pulled_prices_arm)
            gpucb_and_price_optimizers[class_id].update(pulled_bids_arm, pulled_prices_arm, *round_reward)
    # Store collected rewards
    ts_rewards_per_experiment.append(sum(gpts_and_price_optimizers[class_id].collected_rewards for class_id in range(n_classes)))
    ucb_rewards_per_experiment.append(sum(gpucb_and_price_optimizers[class_id].collected_rewards for class_id in range(n_classes)))

    cumregret_ts.append(np.cumsum(opt - ts_rewards_per_experiment[e]))
    cumregret_ucb.append(np.cumsum(opt - ucb_rewards_per_experiment[e]))

    cumreward_ts.append(np.cumsum(ts_rewards_per_experiment[e]))
    cumreward_ucb.append(np.cumsum(ucb_rewards_per_experiment[e]))


plt.figure(0)
plt.title(f"Step4 - Single known context")
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
plt.plot(np.mean(cumregret_ts, axis=0), 'r')
plt.plot(np.mean(cumregret_ucb, axis=0), 'b')
plt.fill_between(range(T), np.mean(cumregret_ts, axis=0) - np.std(cumregret_ts, axis=0), np.mean(cumregret_ts, axis=0) + np.std(cumregret_ts, axis=0), color = "red", alpha = 0.2)
plt.fill_between(range(T), np.mean(cumregret_ucb, axis=0) - np.std(cumregret_ucb, axis=0), np.mean(cumregret_ucb, axis=0) + np.std(cumregret_ucb, axis=0), color = "blue", alpha = 0.2)
plt.legend(["TS","UCB"])
plt.show()

plt.figure(1)
plt.title(f"Step4 - Single known context")
plt.xlabel("t")
plt.ylabel("Instantaneous Regret")
plt.plot(np.mean(opt - ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(opt - ucb_rewards_per_experiment, axis=0), 'b')
plt.fill_between(range(T), np.mean(opt - ts_rewards_per_experiment, axis=0) - np.std(opt - ts_rewards_per_experiment, axis=0), np.mean(opt - ts_rewards_per_experiment, axis=0) + np.std(opt - ts_rewards_per_experiment, axis=0), color = "red", alpha = 0.2)
plt.fill_between(range(T), np.mean(opt - ucb_rewards_per_experiment, axis=0) - np.std(opt - ucb_rewards_per_experiment, axis=0), np.mean(opt - ucb_rewards_per_experiment, axis=0) + np.std(opt - ucb_rewards_per_experiment, axis=0), color = "blue", alpha = 0.2)
plt.legend(["TS","UCB"])
plt.show()

plt.figure(2)
plt.title(f"Step4 - Single known context")
plt.xlabel("t")
plt.ylabel("Cumulative Reward")
plt.plot(np.mean(cumreward_ts, axis=0), 'r')
plt.plot(np.mean(cumreward_ucb, axis=0), 'b')
plt.fill_between(range(T), np.mean(cumreward_ts, axis=0) - np.std(cumreward_ts, axis=0), np.mean(cumreward_ts, axis=0) + np.std(cumreward_ts, axis=0), color = "red", alpha = 0.2)
plt.fill_between(range(T), np.mean(cumreward_ucb, axis=0) - np.std(cumreward_ucb, axis=0), np.mean(cumreward_ucb, axis=0) + np.std(cumreward_ucb, axis=0), color = "blue", alpha = 0.2)
plt.legend(["TS","UCB"])
plt.show()

plt.figure(3)
plt.title(f"Step4 - Single known context")
plt.xlabel("t")
plt.ylabel("Instantaneous Reward")
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(ucb_rewards_per_experiment, axis=0), 'b')
plt.fill_between(range(T), np.mean(ts_rewards_per_experiment, axis=0) - np.std(ts_rewards_per_experiment, axis=0), np.mean(ts_rewards_per_experiment, axis=0) + np.std(ts_rewards_per_experiment, axis=0), color = "red", alpha = 0.2)
plt.fill_between(range(T), np.mean(ucb_rewards_per_experiment, axis=0) - np.std(ucb_rewards_per_experiment, axis=0), np.mean(ucb_rewards_per_experiment, axis=0) + np.std(ucb_rewards_per_experiment, axis=0), color = "blue", alpha = 0.2)
plt.legend(["TS","UCB"])
plt.show()
