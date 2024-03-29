import parameters as param
import numpy as np


class Environment:
    def __init__(self, feature):
        self.feature = feature
        self.pricing_probabilities = param.pricing_probabilities[param.class_from_feature[self.feature]]
        self.optimal_price_idx = np.argmax(self.pricing_probabilities * (param.prices - param.cost))
        self.n_clicks = np.round(param.n_clicks_per_bid_by_feature[self.feature](param.bids)).astype(np.int32)
        self.cum_costs = param.cum_cost_per_bid_by_feature[self.feature](param.bids)
        self.optimal = np.max(self.pricing_probabilities[self.optimal_price_idx] * self.n_clicks * (
                param.prices[self.optimal_price_idx] - param.cost) - self.cum_costs)
        self.optimal_bid_idx = np.argmax(self.pricing_probabilities[self.optimal_price_idx] * self.n_clicks * (
                param.prices[self.optimal_price_idx] - param.cost) - self.cum_costs)
        self.n_bids_arms = len(param.bids)
        self.n_prices_arms = len(param.prices)

    def round(self, pulled_bids_arm, pulled_prices_arm):
        bid = param.bids[pulled_bids_arm]
        price = param.prices[pulled_prices_arm]
        n_clicks = np.round(max(0, param.n_clicks_per_bid_by_feature[self.feature](bid) + np.random.normal(0,
                                                                                                           param.n_clicks_per_bid_sigma))).astype(
            np.int32)
        cum_cost = max(0, param.cum_cost_per_bid_by_feature[self.feature](bid) + np.random.normal(0,
                                                                                                  param.cum_cost_per_bid_sigma))
        result = np.random.binomial(1, self.pricing_probabilities[pulled_prices_arm], n_clicks)
        reward = np.sum(result) * (price - param.cost) - cum_cost
        return np.sum(result), n_clicks, cum_cost, reward
