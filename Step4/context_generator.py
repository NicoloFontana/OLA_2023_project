import pandas as pd

class ContextGenerator:
    def __init__(self, features):
        self.features = features
        self.samples = pd.DataFrame(columns=[*self.features, 'bid', 'price', 'n_clicks', 'cum_costs', 'n_conversions'])

    def update(self, new_samples):
        new_samples_df = pd.DataFrame(new_samples, columns=[*self.features, 'bid', 'price', 'n_clicks', 'cum_costs', 'n_conversions'])
        self.samples.append(new_samples_df, ignore_index=True)

    # def get_context(self):
    #     curr_features = self.features.copy()
    #     while len(curr_features) > 0:
    #         set_feature = # extract df where drop features not in curr_features
    #         # find best (bid,price) pair
    #         # filter set_feature wrt best (bid,price) pair
    #         # compute bound over filtered set_feature
    #         for feature in curr_features:
    #             # compute values per split
    #             set0 = #extract from df where feature == 0
    #             # compute p = #entries in set0 / #entries in set_feature
    #             # find best (bid,price) pair
    #             # filter set0 wrt best (bid,price) pair
    #             # compute bound over filtered set0
    #             set1 = #extract from df
    #             # repeat for set1
    #
    #             # compare set0+set1 vs set_feature and store if better (and how much)
    #         # if exists split better than not => choose best split