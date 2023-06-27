import numpy as np
import matplotlib.pyplot as plt

import parameters
import swucb_learner as swucb
import cusum_ucb_learner as cu_ucb
import environment_5 as env
import ucb_optimizer as ucb_opt

np.random.seed(parameters.seed)

T = 365
class_id = 1
env = env.Environment(class_id, T)
opt = np.array([env.get_opt(t) for t in range(T)])
n_arms = env.n_arms
window_size = int(2 * (T ** 0.5))
M = 70
eps = 0.15
h = np.log(T)
alpha = np.sqrt(0.5 * np.log(T) / T)

Ms = [1, 2, 3, 10, 50, 200, 1000000000000]
epss = [0.03, 0.1, 0.2]
hs = np.array([0.1, 1, 10]) * np.log(T)
alphas = np.sqrt(np.array([0.1, 1, 10])) * np.sqrt(np.log(T) / T)
window_sizes = np.array([int(1 * (T ** 0.5)), int(2 * (T ** 0.5)), int(4 * (T ** 0.5)), int(8 * (T ** 0.5))])

n_experiments = 100

plt.figure(0)
plt.title(f"Step5 - Cusum, h={'{:.2f}'.format(h)}, eps={'{:.2f}'.format(eps)}, alpha={'{:.2f}'.format(alpha)}")
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
for M_ in Ms:
    cusum_ucb_rewards_per_experiment_per_parameters = []
    cumregret_cusum_ucb_per_parameters = []
    for e in range(0, n_experiments):
        cusum_ucb_learner = ucb_opt.UCBOptimizer(cu_ucb.CusumUCBLearner, class_id,
                                                 (n_arms, M_, eps, h, alpha))
        if e % 10 == 0:
            print(f"Experiment {e}")

        for t in range(0, T):
            pulled_arm_price, pulled_arm_bid = cusum_ucb_learner.pull_arm()
            reward = env.round(pulled_arm_price, pulled_arm_bid, t)
            cusum_ucb_learner.update(pulled_arm_price, reward)
        cusum_ucb_rewards_per_experiment_per_parameters.append(cusum_ucb_learner.collected_rewards)
        cumregret_cusum_ucb_per_parameters.append(np.cumsum(opt - cusum_ucb_rewards_per_experiment_per_parameters[e]))
        # print(f"Cusum detections: {cusum_ucb_learner.learner.detections}")
    plt.plot(np.mean(cumregret_cusum_ucb_per_parameters, axis=0), label=f'M={M_}')
    plt.fill_between(range(T), np.mean(cumregret_cusum_ucb_per_parameters, axis=0) - np.std(cumregret_cusum_ucb_per_parameters, axis=0),
                     np.mean(cumregret_cusum_ucb_per_parameters, axis=0) + np.std(cumregret_cusum_ucb_per_parameters, axis=0), alpha=0.2)
plt.legend()
plt.show()

plt.figure(1)
plt.title(f"Step5 - Cusum, M={M}, eps={'{:.2f}'.format(eps)}, alpha={'{:.2f}'.format(alpha)}")
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
for h_ in hs:
    cusum_ucb_rewards_per_experiment_per_parameters = []
    cumregret_cusum_ucb_per_parameters = []
    for e in range(0, n_experiments):
        cusum_ucb_learner = ucb_opt.UCBOptimizer(cu_ucb.CusumUCBLearner, class_id,
                                                 (n_arms, M, eps, h_, alpha))
        if e % 10 == 0:
            print(f"Experiment {e}")

        for t in range(0, T):
            pulled_arm_price, pulled_arm_bid = cusum_ucb_learner.pull_arm()
            reward = env.round(pulled_arm_price, pulled_arm_bid, t)
            cusum_ucb_learner.update(pulled_arm_price, reward)
        cusum_ucb_rewards_per_experiment_per_parameters.append(cusum_ucb_learner.collected_rewards)
        cumregret_cusum_ucb_per_parameters.append(np.cumsum(opt - cusum_ucb_rewards_per_experiment_per_parameters[e]))
    plt.plot(np.mean(cumregret_cusum_ucb_per_parameters, axis=0), label='h={:.2f}'.format(h_))
    plt.fill_between(range(T), np.mean(cumregret_cusum_ucb_per_parameters, axis=0) - np.std(cumregret_cusum_ucb_per_parameters, axis=0),
                     np.mean(cumregret_cusum_ucb_per_parameters, axis=0) + np.std(cumregret_cusum_ucb_per_parameters, axis=0), alpha=0.2)
plt.legend()
plt.show()

plt.figure(2)
plt.title(f"Step5 - Cusum, M={M}, h={'{:.2f}'.format(h)}, alpha={'{:.2f}'.format(alpha)}")
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
for eps_ in epss:
    cusum_ucb_rewards_per_experiment_per_parameters = []
    cumregret_cusum_ucb_per_parameters = []
    for e in range(0, n_experiments):
        cusum_ucb_learner = ucb_opt.UCBOptimizer(cu_ucb.CusumUCBLearner, class_id,
                                                 (n_arms, M, eps_, h, alpha))
        if e % 10 == 0:
            print(f"Experiment {e}")

        for t in range(0, T):
            pulled_arm_price, pulled_arm_bid = cusum_ucb_learner.pull_arm()
            reward = env.round(pulled_arm_price, pulled_arm_bid, t)
            cusum_ucb_learner.update(pulled_arm_price, reward)
        cusum_ucb_rewards_per_experiment_per_parameters.append(cusum_ucb_learner.collected_rewards)
        cumregret_cusum_ucb_per_parameters.append(np.cumsum(opt - cusum_ucb_rewards_per_experiment_per_parameters[e]))
    plt.plot(np.mean(cumregret_cusum_ucb_per_parameters, axis=0), label='eps={:.2f}'.format(eps_))
    plt.fill_between(range(T), np.mean(cumregret_cusum_ucb_per_parameters, axis=0) - np.std(cumregret_cusum_ucb_per_parameters, axis=0),
                     np.mean(cumregret_cusum_ucb_per_parameters, axis=0) + np.std(cumregret_cusum_ucb_per_parameters, axis=0), alpha=0.2)
plt.legend()
plt.show()

plt.figure(3)
plt.title(f"Step5 - Cusum, M={M}, h={'{:.2f}'.format(h)}, eps={'{:.2f}'.format(eps)}")
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
for alpha_ in alphas:
    cusum_ucb_rewards_per_experiment_per_parameters = []
    cumregret_cusum_ucb_per_parameters = []
    for e in range(0, n_experiments):
        cusum_ucb_learner = ucb_opt.UCBOptimizer(cu_ucb.CusumUCBLearner, class_id,
                                                 (n_arms, M, eps, h, alpha_))
        if e % 10 == 0:
            print(f"Experiment {e}")

        for t in range(0, T):
            pulled_arm_price, pulled_arm_bid = cusum_ucb_learner.pull_arm()
            reward = env.round(pulled_arm_price, pulled_arm_bid, t)
            cusum_ucb_learner.update(pulled_arm_price, reward)
        cusum_ucb_rewards_per_experiment_per_parameters.append(cusum_ucb_learner.collected_rewards)
        cumregret_cusum_ucb_per_parameters.append(np.cumsum(opt - cusum_ucb_rewards_per_experiment_per_parameters[e]))
    plt.plot(np.mean(cumregret_cusum_ucb_per_parameters, axis=0), label='alpha={:.2f}'.format(alpha_))
    plt.fill_between(range(T), np.mean(cumregret_cusum_ucb_per_parameters, axis=0) - np.std(cumregret_cusum_ucb_per_parameters, axis=0),
                     np.mean(cumregret_cusum_ucb_per_parameters, axis=0) + np.std(cumregret_cusum_ucb_per_parameters, axis=0), alpha=0.2)
plt.legend()
plt.show()

plt.figure(4)
plt.title("Step5 - Sliding Window")
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
for window_size in window_sizes:
    swucb_rewards_per_experiment_per_parameters = []
    cumregret_swucb_per_parameters = []
    for e in range(0, n_experiments):
        swucb_learner = ucb_opt.UCBOptimizer(swucb.SWUCBLearner, class_id,
                                             (n_arms, window_size))
        if e % 10 == 0:
            print(f"Experiment {e}")

        for t in range(0, T):
            pulled_arm_price, pulled_arm_bid = swucb_learner.pull_arm()
            reward = env.round(pulled_arm_price, pulled_arm_bid, t)
            swucb_learner.update(pulled_arm_price, reward)
        swucb_rewards_per_experiment_per_parameters.append(swucb_learner.collected_rewards)
        cumregret_swucb_per_parameters.append(np.cumsum(opt - swucb_rewards_per_experiment_per_parameters[e]))
    plt.plot(np.mean(cumregret_swucb_per_parameters, axis=0), label=f'SW={window_size}')
    plt.fill_between(range(T), np.mean(cumregret_swucb_per_parameters, axis=0) - np.std(cumregret_swucb_per_parameters, axis=0),
                     np.mean(cumregret_swucb_per_parameters, axis=0) + np.std(cumregret_swucb_per_parameters, axis=0), alpha=0.2)
plt.legend()
plt.show()
