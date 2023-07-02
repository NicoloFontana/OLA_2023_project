import numpy as np
import matplotlib.pyplot as plt
import environment_3 as env
import parameters
from gpts_and_price_optimizer import *
from gpucb_and_price_optimizer import *
from scipy.stats import beta

np.random.seed(param.seed)
T = 100

class_id = 1
env = env.Environment(class_id)
opt = env.optimal

n_experiments = 1
ts_rewards_per_experiment = []
ucb_rewards_per_experiment = []

cumregret_ts = []
cumregret_ucb = []

cumreward_ts = []
cumreward_ucb = []

for e in range (0,n_experiments):
    # Create environment and learners
    gpts_and_price_optimizer = GPTSAndPriceOptimizer(param.bids, param.prices)
    gpucb_and_price_optimizer = GPUCBAndPriceOptimizer(param.bids, param.prices)

    for t in range (0,T):
        # Pull arms and update learners
        # Thompson sampling
        if t % 10 == 0:
            print(f"{t} of experiment {e}")
        pulled_arms = gpts_and_price_optimizer.pull_arms()
        pulled_bids_arm = pulled_arms[0]
        pulled_prices_arm = pulled_arms[1]
        round_reward = env.round(pulled_bids_arm, pulled_prices_arm)
        gpts_and_price_optimizer.update(pulled_bids_arm, pulled_prices_arm, *round_reward)


        # UCB
        pulled_arms = gpucb_and_price_optimizer.pull_arms()
        pulled_bids_arm = pulled_arms[0]
        pulled_prices_arm = pulled_arms[1]
        round_reward = env.round(pulled_bids_arm, pulled_prices_arm)
        gpucb_and_price_optimizer.update(pulled_bids_arm, pulled_prices_arm, *round_reward)

        if t == T-1:
            ts_one_round_betas = gpts_and_price_optimizer.price_learner.beta_parameters

            ts_one_round_nclick_gp_mean = gpts_and_price_optimizer.n_click_learner.means
            ts_one_round_nclick_gp_std = gpts_and_price_optimizer.n_click_learner.sigmas
            ts_one_round_nclick_gp_samples_x = gpts_and_price_optimizer.n_click_learner.pulled_arms
            ts_one_round_nclick_gp_samples_y = gpts_and_price_optimizer.n_click_learner.collected_rewards
            ts_one_round_cumulative_gp_mean = gpts_and_price_optimizer.cum_cost_learner.means
            ts_one_round_cumulative_gp_std = gpts_and_price_optimizer.cum_cost_learner.sigmas
            ts_one_round_cumulative_gp_samples_x = gpts_and_price_optimizer.cum_cost_learner.pulled_arms
            ts_one_round_cumulative_gp_samples_y = gpts_and_price_optimizer.cum_cost_learner.collected_rewards

            ucb_one_round_nclick_gp_mean = gpucb_and_price_optimizer.n_click_learner.empirical_means
            ucb_one_round_nclick_gp_std = gpucb_and_price_optimizer.n_click_learner.sigmas
            ucb_one_round_nclick_gp_samples_x = gpucb_and_price_optimizer.n_click_learner.pulled_arms
            ucb_one_round_nclick_gp_samples_y = gpucb_and_price_optimizer.n_click_learner.collected_rewards
            ucb_one_round_cumulative_gp_mean = gpucb_and_price_optimizer.cum_cost_learner.empirical_means
            ucb_one_round_cumulative_gp_std = gpucb_and_price_optimizer.cum_cost_learner.sigmas
            ucb_one_round_cumulative_gp_samples_x = gpucb_and_price_optimizer.cum_cost_learner.pulled_arms
            ucb_one_round_cumulative_gp_samples_y = gpucb_and_price_optimizer.cum_cost_learner.collected_rewards

    # Store collected rewards
    ts_rewards_per_experiment.append(gpts_and_price_optimizer.collected_rewards)
    ucb_rewards_per_experiment.append(gpucb_and_price_optimizer.collected_rewards)

    cumregret_ts.append(np.cumsum(opt - ts_rewards_per_experiment[e]))
    cumregret_ucb.append(np.cumsum(opt - ucb_rewards_per_experiment[e]))

    cumreward_ts.append(np.cumsum(ts_rewards_per_experiment[e]))
    cumreward_ucb.append(np.cumsum(ucb_rewards_per_experiment[e]))


plt.figure(0)
plt.title(f"Step3 - Class {class_id}")
plt.xlabel("t")
plt.ylabel("Cumulative Regret")
plt.plot(np.mean(cumregret_ts, axis=0), 'r')
plt.plot(np.mean(cumregret_ucb, axis=0), 'b')
plt.fill_between(range(T), np.mean(cumregret_ts, axis=0) - np.std(cumregret_ts, axis=0), np.mean(cumregret_ts, axis=0) + np.std(cumregret_ts, axis=0), color = "red", alpha = 0.2)
plt.fill_between(range(T), np.mean(cumregret_ucb, axis=0) - np.std(cumregret_ucb, axis=0), np.mean(cumregret_ucb, axis=0) + np.std(cumregret_ucb, axis=0), color = "blue", alpha = 0.2)
plt.legend(["TS","UCB"])
plt.show()

plt.figure(1)
plt.title(f"Step3 - Class {class_id}")
plt.xlabel("t")
plt.ylabel("Instantaneous Regret")
plt.plot(np.mean(opt - ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(opt - ucb_rewards_per_experiment, axis=0), 'b')
plt.fill_between(range(T), np.mean(opt - ts_rewards_per_experiment, axis=0) - np.std(opt - ts_rewards_per_experiment, axis=0), np.mean(opt - ts_rewards_per_experiment, axis=0) + np.std(opt - ts_rewards_per_experiment, axis=0), color = "red", alpha = 0.2)
plt.fill_between(range(T), np.mean(opt - ucb_rewards_per_experiment, axis=0) - np.std(opt - ucb_rewards_per_experiment, axis=0), np.mean(opt - ucb_rewards_per_experiment, axis=0) + np.std(opt - ucb_rewards_per_experiment, axis=0), color = "blue", alpha = 0.2)
plt.legend(["TS","UCB"])
plt.show()

plt.figure(2)
plt.title(f"Step3 - Class {class_id}")
plt.xlabel("t")
plt.ylabel("Cumulative Reward")
plt.plot(np.mean(cumreward_ts, axis=0), 'r')
plt.plot(np.mean(cumreward_ucb, axis=0), 'b')
plt.fill_between(range(T), np.mean(cumreward_ts, axis=0) - np.std(cumreward_ts, axis=0), np.mean(cumreward_ts, axis=0) + np.std(cumreward_ts, axis=0), color = "red", alpha = 0.2)
plt.fill_between(range(T), np.mean(cumreward_ucb, axis=0) - np.std(cumreward_ucb, axis=0), np.mean(cumreward_ucb, axis=0) + np.std(cumreward_ucb, axis=0), color = "blue", alpha = 0.2)
plt.legend(["TS","UCB"])
plt.show()

plt.figure(3)
plt.title(f"Step3 - Class {class_id}")
plt.xlabel("t")
plt.ylabel("Instantaneous Reward")
plt.plot(np.mean(ts_rewards_per_experiment, axis=0), 'r')
plt.plot(np.mean(ucb_rewards_per_experiment, axis=0), 'b')
plt.fill_between(range(T), np.mean(ts_rewards_per_experiment, axis=0) - np.std(ts_rewards_per_experiment, axis=0), np.mean(ts_rewards_per_experiment, axis=0) + np.std(ts_rewards_per_experiment, axis=0), color = "red", alpha = 0.2)
plt.fill_between(range(T), np.mean(ucb_rewards_per_experiment, axis=0) - np.std(ucb_rewards_per_experiment, axis=0), np.mean(ucb_rewards_per_experiment, axis=0) + np.std(ucb_rewards_per_experiment, axis=0), color = "blue", alpha = 0.2)
plt.legend(["TS","UCB"])
plt.show()

plt.figure(4)
plt.title(f"n_clicks learned by GPs at round {t+1}")
plt.xlabel("bids")
plt.ylabel("n_clicks")
plt.plot(param.bids,parameters.n_clicks_per_bid_by_class[1](param.bids), 'k')
plt.plot(param.bids,ts_one_round_nclick_gp_mean, 'r')
plt.plot(param.bids,ucb_one_round_nclick_gp_mean, 'b')
plt.plot(ts_one_round_nclick_gp_samples_x,ts_one_round_nclick_gp_samples_y, 'ro',markersize=2)
plt.plot(ucb_one_round_nclick_gp_samples_x,ucb_one_round_nclick_gp_samples_y, 'ob',markersize=2)
plt.fill_between(param.bids, ts_one_round_nclick_gp_mean - ts_one_round_nclick_gp_std, ts_one_round_nclick_gp_mean + ts_one_round_nclick_gp_std, color = "red", alpha = 0.2)
plt.fill_between(param.bids, ucb_one_round_nclick_gp_mean - ucb_one_round_nclick_gp_std, ucb_one_round_nclick_gp_mean + ucb_one_round_nclick_gp_std, color = "blue", alpha = 0.2)
plt.legend(["original","GPTS","GPUCB"])
plt.show()

plt.figure(5)
plt.title(f"cumulative cost learned by GPs at round {t+1}")
plt.xlabel("bids")
plt.ylabel("cumulative cost")
plt.plot(param.bids,parameters.cum_cost_per_bid_by_class[1](param.bids), 'k')
plt.plot(param.bids,ts_one_round_cumulative_gp_mean, 'r')
plt.plot(param.bids,ucb_one_round_cumulative_gp_mean, 'b')
plt.plot(ts_one_round_cumulative_gp_samples_x,ts_one_round_cumulative_gp_samples_y, 'or',markersize=2)
plt.plot(ucb_one_round_cumulative_gp_samples_x,ucb_one_round_cumulative_gp_samples_y, 'bo',markersize=2)
plt.fill_between(param.bids, ts_one_round_cumulative_gp_mean - ts_one_round_cumulative_gp_std, ts_one_round_cumulative_gp_mean + ts_one_round_cumulative_gp_std, color = "red", alpha = 0.2)
plt.fill_between(param.bids, ucb_one_round_cumulative_gp_mean - ucb_one_round_cumulative_gp_std, ucb_one_round_cumulative_gp_mean + ucb_one_round_cumulative_gp_std, color = "blue", alpha = 0.2)
plt.legend(["original","GPTS","GPUCB"])
plt.show()

# plt.figure(6)
# plt.title(f"cumulative cost learned by GPs at round {t+1}")
# plt.xlabel("bids")
# plt.ylabel("cumulative cost")
# plt.plot(param.prices,param.pricing_probabilities, 'k')
# plt.plot(param.bids,ts_one_round_cumulative_gp_mean, 'r')
# plt.fill_between(param.bids, ts_one_round_cumulative_gp_mean - ts_one_round_cumulative_gp_std, ts_one_round_cumulative_gp_mean + ts_one_round_cumulative_gp_std, color = "red", alpha = 0.2)
# plt.legend(["original","GPTS","GPUCB"])
# plt.show()


plt.figure(6)
a = ts_one_round_betas[0][0]
b = ts_one_round_betas[0][1]
a1 = ts_one_round_betas[1][0]
b1 = ts_one_round_betas[1][1]
a2 = ts_one_round_betas[2][0]
b2 = ts_one_round_betas[2][1]
a3 = ts_one_round_betas[3][0]
b3 = ts_one_round_betas[3][1]
a4 = ts_one_round_betas[4][0]
b4 = ts_one_round_betas[4][1]
vec = np.linspace(0,1, 100)
scale = 100

plt.plot([1, 2, 3, 4, 5],param.pricing_probabilities[1], 'ok')
plt.plot(beta.pdf(vec, a, b)/scale+1,vec,
       'r-', lw=3, alpha=1, label='beta pdf')
plt.plot(beta.pdf(vec, a1, b1)/scale+2,vec,
       'r-', lw=3, alpha=1, label='beta pdf')
plt.plot(beta.pdf(vec, a2, b2)/scale+3,vec,
       'r-', lw=3, alpha=1, label='beta pdf')
plt.plot(beta.pdf(vec, a3, b3)/scale+4,vec,
       'r-', lw=3, alpha=1, label='beta pdf')
plt.plot(beta.pdf(vec, a4, b4)/scale+5,vec,
       'r-', lw=3, alpha=1, label='beta pdf')
plt.legend(["original","TS"])
plt.title(f"conversion rates learned by TS at round {t+1}")
plt.xlabel("arms")
plt.ylabel("probability distribution")
plt.show()
