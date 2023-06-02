import numpy as np
import matplotlib.pyplot as plt
import parameters as param
import tslearner as ts
import ucb_learner as ucb
import greedy_learner as gr
import environment_1 as env


T = 365



# copied
n_arms = 5
class_id = 1
opt = p[3]

n_experiments = 1000
ts_rewards_per_experiment = []
gr_rewards_per_experiment = []

for e in range (0,n_experiments):
    # Create environment and learners
    env = env.Environment(class_id)
    ts_learner = ts.TSLearner(n_arms=n_arms)
    gr_learner = Greedy_Learner(n_arms=n_arms)

    for t in range (0,T):
        # Pull arms and update learners
        # Thompson sampling
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)
        # Greedy
        pulled_arm = gr_learner.pull_arm()
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm, reward)
    # Store collected rewards
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    gr_rewards_per_experiment.append(gr_learner.collected_rewards)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment, axis=0)), 'b')
plt.legend(["TS","Greedy"])
plt.show()