import pandas as pd
import numpy as np
import parameters as param


class ContextGenerator:
    def __init__(self, features_names, features):
        self.features_names = features_names
        self.features = features
        self.samples = pd.DataFrame(
            columns=[*self.features_names, 'bid', 'price', 'n_conversions', 'n_clicks', 'cum_costs', 'reward'])

    def update(self, new_samples):
        new_samples_df = pd.DataFrame(new_samples,
                                      columns=[*self.features_names, 'bid', 'price', 'n_conversions', 'n_clicks',
                                               'cum_costs',
                                               'reward'])
        self.samples = pd.concat([self.samples, new_samples_df])

    def get_context(self): # convert list of dictionaries into list of lists
        context_dict = self.get_context_recursive()
        context_list = []
        if len(context_dict) == 0:
            return [param.feature_combos]
        for dictionary in context_dict:
            class_list = []
            for i in range(len(param.features_names)):
                if param.features_names[i] not in dictionary.keys():
                    class_list_copy = class_list.copy()
                    if len(class_list_copy) == 0:
                        class_list = [['0'], ['1']]
                    else:
                        for class_name_list in class_list_copy:
                            class_name_list_0 = class_name_list + ['0']
                            class_name_list_1 = class_name_list + ['1']
                            class_list.append(class_name_list_0)
                            class_list.append(class_name_list_1)
                            class_list.remove(class_name_list)
                            print(class_list)
                else:
                    if len(class_list) == 0:
                        class_list = [[str(dictionary[param.features_names[i]])]]
                    else:
                        for class_name_list in class_list:
                            class_name_list.append(str(dictionary[param.features_names[i]]))
                            print(class_name_list)
            context_list.append([''.join(class_name_list) for class_name_list in class_list])
        print(f"dict {context_dict}")
        print(f"list {context_list}")
        return context_list

    def get_context_recursive(self, features=param.features_names, samples=None, lower_bound_mean=None):
        if len(features) == 0:
            return []
        features_names_to_be_split = features.copy()
        if samples is None:
            samples = self.samples
            empirical_mean_per_arm = samples.groupby(['bid', 'price'])[
                'reward'].mean().to_numpy()
            empirical_std_per_arm = samples.groupby(['bid', 'price'])[
                'reward'].std().fillna(0).to_numpy()
            n_samples_per_arm = samples.groupby(['bid', 'price'])['reward'].count().to_numpy()
            lower_bound_mean = np.max(empirical_mean_per_arm - 1.96 * empirical_std_per_arm / np.sqrt(n_samples_per_arm))
        max_split_value = -np.infty
        max_feature = None
        max_filtered_samples_0 = None
        max_lower_bound_0 = None
        max_filtered_samples_1 = None
        max_lower_bound_1 = None
        for feature in features:
            # filter df
            filtered_samples_per_feature_0 = samples.copy().loc[samples[feature] == '0']
            lower_bounds_0 = self.get_lower_bounds(filtered_samples_per_feature_0, samples)
            filtered_samples_per_feature_1 = samples.copy().loc[samples[feature] == '1']
            lower_bounds_1 = self.get_lower_bounds(filtered_samples_per_feature_1, samples)
            # print(lower_bounds_0, lower_bounds_1)
            split_value = lower_bounds_0[0]*lower_bounds_0[1] + lower_bounds_1[0]*lower_bounds_1[1]
            print(f"Feature {feature}: \n"
                  f"p1={lower_bounds_0[0]}, mu1={lower_bounds_0[1]}, p1*mu1={lower_bounds_0[0]*lower_bounds_0[1]}\n"
                  f"p2={lower_bounds_1[0]}, mu2={lower_bounds_1[1]}, p2*mu2={lower_bounds_1[0]*lower_bounds_1[1]}\n"
                  f"mu0={lower_bound_mean}")
            if split_value >= lower_bound_mean and split_value > max_split_value:
                max_split_value = split_value
                max_filtered_samples_0 = filtered_samples_per_feature_0.copy()
                max_lower_bound_0 = lower_bounds_0[1]
                max_filtered_samples_1 = filtered_samples_per_feature_1.copy()
                max_lower_bound_1 = lower_bounds_1[1]
                max_feature = feature
        if max_feature is not None:
            features_names_to_be_split.remove(max_feature)
            context_0 = self.get_context_recursive(features_names_to_be_split, max_filtered_samples_0, max_lower_bound_0)
            if len(context_0) == 0:
                context_0.append({max_feature: 0})
            else:
                for sub_context in context_0:
                    sub_context[max_feature] = 0
            context_1 = self.get_context_recursive(features_names_to_be_split, max_filtered_samples_1, max_lower_bound_1)
            if len(context_1) == 0:
                context_1.append({max_feature: 1})
            else:
                for sub_context in context_1:
                    sub_context[max_feature] = 1
            return context_0 + context_1
        return []

    def get_lower_bounds(self, filtered_samples, total_samples):
        lb_probability = len(filtered_samples.index) / len(total_samples.index) - np.sqrt(
            -np.log(param.confidence) / (2 * len(filtered_samples.index)))
        grouped_rewards = filtered_samples.groupby(['bid', 'price'])['reward']
        empirical_mean_per_arm = grouped_rewards.mean().to_numpy()
        empirical_std_per_arm = grouped_rewards.std().fillna(0).to_numpy()
        n_samples_per_arm = filtered_samples.groupby(['bid', 'price']).size().to_numpy()
        print(f"get_lower_bounds(): n_samples_per_arm={n_samples_per_arm}")
        # print(f"mean {empirical_mean_per_arm}, std {empirical_std_per_arm}, n_samples {n_samples_per_arm}")
        lb_reward = empirical_mean_per_arm - 1.96 * empirical_std_per_arm / np.sqrt(n_samples_per_arm)
        return lb_probability, np.max(lb_reward)


    def get_samples(self, context):
        filtered_samples = pd.DataFrame(
            columns=[*self.features_names, 'bid', 'price', 'n_conversions', 'n_clicks', 'cum_costs', 'reward'])
        for feature_combo in context:
            filtered_samples_per_feature = self.samples
            features = list(feature_combo)
            for i in range(len(param.features_names)):
                filtered_samples_per_feature = filtered_samples_per_feature.loc[
                    filtered_samples_per_feature[param.features_names[i]] == features[i]]
            filtered_samples = pd.concat([filtered_samples, filtered_samples_per_feature])
        return [filtered_samples[column].to_numpy() for column in filtered_samples.columns[2:]]

        # return  pulled_bids_arms, pulled_prices_arms, n_conversions_per_arm, n_clicks_per_arm, cum_cost_per_arm, reward_per_arm
