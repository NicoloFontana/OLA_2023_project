import numpy as np
from Step3.optimizer_learner import *
from gpucb_learner import *
import parameters as param
from ts_learner import *


class GPUCBAndPriceOptimizer(OptimizerLearner):
    def __init__(self, bids_arms, prices_arms, class_id):
        super().__init__(bids_arms, prices_arms)
        self.n_click_learner = GPUCBLearner(bids_arms)
        self.cum_cost_learner = GPUCBLearner(bids_arms)
        self.price_learner = TSLearner(len(prices_arms))
        self.class_id = class_id

    def update(self, pulled_bids_arm, pulled_prices_arm, n_conversions, n_clicks, cum_cost, reward):
        self.update_observations(reward)
        self.n_click_learner.update(pulled_bids_arm, n_clicks)
        self.cum_cost_learner.update(pulled_bids_arm, cum_cost)
        self.price_learner.update(pulled_prices_arm, [n_conversions, n_clicks - n_conversions, reward])

    def pull_arms(self):
        sampled_price_idx = self.price_learner.pull_arm()
        n_clicks_upper_conf = self.n_click_learner.empirical_means + self.n_click_learner.confidence
        cum_cost_lower_conf = self.cum_cost_learner.empirical_means - self.cum_cost_learner.confidence
        sampled_conversion_rate = np.random.beta(self.price_learner.beta_parameters[sampled_price_idx, 0],
                                                 self.price_learner.beta_parameters[sampled_price_idx, 1])
        sampled_reward = sampled_conversion_rate * n_clicks_upper_conf * (
                param.prices[sampled_price_idx] - param.cost) - cum_cost_lower_conf
        return np.argmax(sampled_reward), sampled_price_idx
