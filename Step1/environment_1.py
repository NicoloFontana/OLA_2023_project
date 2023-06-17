import parameters as param
import numpy as np


class Environment:
    def __init__(self, class_id):
        self.class_id = class_id
        self.probabilities = param.pricing_probabilities[self.class_id]
        self.optimal_price_idx = np.argmax(self.probabilities*(param.prices-param.cost))
        self.n_clicks = np.round(param.n_clicks_per_bid_by_class[self.class_id](param.bids)).astype(np.int32)
        self.cum_costs = param.cum_cost_per_bid_by_class[self.class_id](param.bids)
        self.optimal = np.max(self.probabilities[self.optimal_price_idx] * self.n_clicks * (param.prices[self.optimal_price_idx] - param.cost) - self.cum_costs)
        self.optimal_bid_idx = np.argmax(self.probabilities[self.optimal_price_idx] * self.n_clicks * (param.prices[self.optimal_price_idx] - param.cost) - self.cum_costs)
        self.n_arms = len(self.probabilities)

    def round(self, pulled_arm):
        result = np.random.binomial(1, self.probabilities[pulled_arm], self.n_clicks[self.optimal_bid_idx])
        reward = np.sum(result) * (param.prices[pulled_arm] - param.cost) - self.cum_costs[self.optimal_bid_idx]
        return np.sum(result), self.n_clicks[self.optimal_bid_idx] - np.sum(result), reward
