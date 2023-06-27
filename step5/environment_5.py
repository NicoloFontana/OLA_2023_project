import parameters as param
import numpy as np
import math


class Environment:
    def __init__(self, class_id, T):
        self.class_id = class_id
        self.probabilities = param.pricing_probabilities_by_phase
        self.T = T
        self.n_phases = len(self.probabilities.keys())
        self.phase_size = math.ceil(T / self.n_phases)
        self.n_clicks = np.round(param.n_clicks_per_bid_by_class[self.class_id](param.bids)).astype(np.int32)
        self.cum_costs = param.cum_cost_per_bid_by_class[self.class_id](param.bids)
        self.optimal_price_idx = {}
        self.optimal = {}
        self.optimal_bid_idx = {}
        self.n_arms = len(param.prices)
        for phase in range(1, self.n_phases + 1):
            self.optimal_price_idx[phase] = np.argmax(self.probabilities[phase]*(param.prices-param.cost))
            self.optimal[phase] = np.max(self.probabilities[phase][self.optimal_price_idx[phase]] * self.n_clicks * (param.prices[self.optimal_price_idx[phase]] - param.cost) - self.cum_costs)
            self.optimal_bid_idx[phase] = np.argmax(self.probabilities[phase][self.optimal_price_idx[phase]] * self.n_clicks * (param.prices[self.optimal_price_idx[phase]] - param.cost) - self.cum_costs)

    def round(self, pulled_arm, t):
        phase = min(math.floor(t / self.phase_size) + 1, self.n_phases)
        result = np.random.binomial(1, self.probabilities[phase][pulled_arm], self.n_clicks[self.optimal_bid_idx[phase]])
        reward = np.sum(result) * (param.prices[pulled_arm] - param.cost) - self.cum_costs[self.optimal_bid_idx[phase]]
        return np.sum(result), self.n_clicks[self.optimal_bid_idx[phase]] - np.sum(result), reward, result

    def get_opt(self, t):
        phase = min(math.floor(t / self.phase_size) + 1, self.n_phases)
        return self.optimal[phase]
