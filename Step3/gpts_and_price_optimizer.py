import numpy as np
from Step3.optimizer_learner import *
from gpts_learner import *
import parameters as param
from ts_learner import *


class GPTSAndPriceOptimizer(OptimizerLearner):
    def __init__(self, bids_arms, prices_arms, class_id):
        super().__init__(bids_arms, prices_arms)
        self.n_click_learner = GPTSLearner(bids_arms)
        self.cum_cost_learner = GPTSLearner(bids_arms)
        self.price_learner = TSLearner(len(prices_arms))
        self.class_id = class_id

    def update(self, pulled_bids_arm, pulled_prices_arm, n_conversions, n_clicks, cum_cost, reward):
        self.update_observations(reward)
        self.n_click_learner.update(pulled_bids_arm, n_clicks)
        self.cum_cost_learner.update(pulled_bids_arm, cum_cost)
        self.price_learner.update(pulled_prices_arm, [n_conversions, n_clicks - n_conversions, reward])

    def pull_arms(self):
        sampled_price_idx = self.price_learner.pull_arm()
        sampled_n_clicks = np.random.normal(self.n_click_learner.means, self.n_click_learner.sigmas)
        sampled_cum_cost = np.random.normal(self.cum_cost_learner.means, self.cum_cost_learner.sigmas)
        sampled_conversion_rate = np.random.beta(self.price_learner.beta_parameters[sampled_price_idx, 0],
                                                 self.price_learner.beta_parameters[sampled_price_idx, 1])
        sampled_reward = sampled_conversion_rate * sampled_n_clicks * (
                param.prices[sampled_price_idx] - param.cost) - sampled_cum_cost
        return np.argmax(sampled_reward), sampled_price_idx
