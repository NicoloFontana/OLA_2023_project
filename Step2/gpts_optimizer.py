import numpy as np
from learner import *
from gpts_learner import *
import parameters as param


class GPTSOptimizer(Learner):
    def __init__(self, arms, class_id):
        super().__init__(arms.shape[0])
        self.n_click_learner = GPTSLearner(arms)
        self.cumcost_learner = GPTSLearner(arms)
        self.class_id = class_id

    def update(self, pulled_arm, n_clicks, cum_cost, reward):
        self.update_observations(pulled_arm, reward)
        self.n_click_learner.update(pulled_arm, n_clicks)
        self.cumcost_learner.update(pulled_arm, cum_cost)

    def pull_arm(self):
        sampled_n_clicks = np.random.normal(self.n_click_learner.means, self.n_click_learner.sigmas)
        sampled_cum_cost = np.random.normal(self.cumcost_learner.means, self.cumcost_learner.sigmas)
        optimal_price_idx = np.argmax(param.pricing_probabilities[self.class_id] * (param.prices - param.cost))
        sampled_reward = param.pricing_probabilities[self.class_id][optimal_price_idx] * sampled_n_clicks * (
                    param.prices[optimal_price_idx] - param.cost) - sampled_cum_cost
        return np.argmax(sampled_reward)
