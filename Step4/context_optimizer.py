from context_generator import *
import pandas as pd
from Step3.gpts_and_price_optimizer import *
import parameters as param


class ContextOptimizer:
    def __init__(self, optimizer_type):
        self.context_generator = ContextGenerator(param.features_names, param.feature_combos)
        self.optimizer_type = optimizer_type
        self.rm_per_context = {
            tuple(param.feature_combos): self.optimizer_type(param.bids, param.prices)
        }
        self.T = 0
        self.collected_rewards = []

    def pull_arms(self):
        bids_and_prices = {}
        for context in self.rm_per_context.keys():
            arms = self.rm_per_context[context].pull_arms()
            for feature in context:
                bids_and_prices[feature] = arms
        return bids_and_prices

    def update(self, input_per_feature):
        self.T += 1
        # update
        for context in self.rm_per_context.keys():
            pulled_bids_arm = input_per_feature[context[0]][0]
            pulled_prices_arm = input_per_feature[context[0]][1]
            n_conversion_per_context = sum(input_per_feature[feature][2] for feature in context)
            n_clicks_per_context = sum(input_per_feature[feature][3] for feature in context)
            cum_cost_per_context = sum(input_per_feature[feature][4] for feature in context)
            reward_per_context = sum(input_per_feature[feature][5] for feature in context)
            self.rm_per_context[context].update(pulled_bids_arm, pulled_prices_arm, n_conversion_per_context,
                                                n_clicks_per_context, cum_cost_per_context, reward_per_context)
        self.collected_rewards.append(sum(input_per_feature[feature][5] for feature in param.feature_combos))
        self.context_generator.update(
            [tuple(feature) + input_per_feature[feature] for feature in input_per_feature.keys()])
        # generate contexts
        if self.T % 14 == 0:
            context_structure = self.context_generator.get_context()
            keys = list(self.rm_per_context.keys()).copy()
            for context in keys:
                if context not in context_structure:
                    del self.rm_per_context[context]
            for context in context_structure:
                if context not in keys:
                    self.rm_per_context[tuple(context)] = self.optimizer_type(param.bids, param.prices)
                    # retrieve samples
                    samples = self.context_generator.get_samples(context)
                    self.context_generator.samples = pd.DataFrame(
                        columns=[*param.features_names, 'bid', 'price', 'n_conversions', 'n_clicks', 'cum_costs',
                                 'reward'])
                    # bulk update learner
                    # self.rm_per_context[tuple(context)].update_bulk(*samples)
