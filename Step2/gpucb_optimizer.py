import numpy as np
from learner import *
from gpucb_learner import *
import parameters as param


class GPUCBOptimizer(Learner):
    def __init__(self, arms, class_id):
        super().__init__(arms.shape[0])
        self.n_click_learner = GPUCBLearner(arms)
        self.cumcost_learner = GPUCBLearner(arms)
        self.class_id = class_id

    def update(self, pulled_arm, n_clicks, cum_cost, reward):
        self.update_observations(pulled_arm, reward)
        self.n_click_learner.update(pulled_arm, n_clicks)
        self.cumcost_learner.update(pulled_arm, cum_cost)

    def pull_arm(self):
        n_clicks_upper_conf = self.n_click_learner.empirical_means + self.n_click_learner.confidence
        cum_cost_lower_conf = self.cumcost_learner.empirical_means - self.cumcost_learner.confidence
        optimal_price_idx = np.argmax(param.pricing_probabilities[self.class_id] * (param.prices - param.cost))
        sampled_reward = param.pricing_probabilities[self.class_id][optimal_price_idx] * n_clicks_upper_conf * (
                    param.prices[optimal_price_idx] - param.cost) - cum_cost_lower_conf
        return np.random.choice(np.where(sampled_reward == sampled_reward.max())[0])
