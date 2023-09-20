import random
import numpy as np
import pandas as pd
import json

import itertools as it

from tqdm import tqdm

import sklearn.linear_model
import sklearn.ensemble
import sklearn.model_selection

import matplotlib.pyplot as plt

class Util():
    
    def __init__(self, ds_mf, algo_mf, train_curves, valid_curves, test_curves):
        self.ds_mf = ds_mf
        self.algo_mf = algo_mf
        self.train_curves = train_curves
        self.valid_curves = valid_curves
        self.test_curves = test_curves
        self.datasets = datasets = sorted(ds_mf.keys())
        self.algos = algos = sorted(algo_mf.keys())
        self.budgets = budgets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # compute times
        self.TIMES = np.zeros((len(datasets), len(algos), 10))
        for i, ds_id in enumerate(datasets):
            for j, algo_id in enumerate(algos):
                induction_times = np.array(train_curves[ds_id][algo_id].times)
                for k in range(10):
                    if len(induction_times) > k:
                        self.TIMES[i,j,k] = induction_times[k]
                    else:
                        self.TIMES[i,j,k] = np.nan
        
        # compute scores of actions
        self.SCORES = np.zeros((len(datasets), len(algos), 10))
        for i, ds_id in enumerate(datasets):
            for j, algo_id in enumerate(algos):
                curve = test_curves[ds_id][algo_id].scores
                for k in range(10):
                    if len(curve) > k:
                        self.SCORES[i,j,k] = curve[k]
                    else:
                        self.SCORES[i,j,k] = np.nan
                        
    def get_cumulative_time(self, dataset, t):
        t_0 = 60.0 # this is hard-coded in the scoring.py
        time_budget = int(self.ds_mf[dataset]["time_budget"])
        if t > time_budget:
            raise ValueError(f"Cannot compute cumulative time for t = {t} if time budget is only {time_budget}")
        return np.log(1+t/t_0)/np.log(1+time_budget/t_0)
    
    def get_lalc_of_schedule(self, dataset, schedule):
        v_last = 0
        i = self.datasets.index(dataset)
        lalc = 0
        elapsed_time = 0
        
        for algo, budget in schedule:
            j = self.algos.index(algo)
            k = self.budgets.index(budget)
            v = self.SCORES[i,j,k]
            elapsed_time_now = self.TIMES[i,j,k]

            # ignore deteriorations
            if v < v_last:
                v = v_last

            if elapsed_time > 0:
                t_last = self.get_cumulative_time(dataset, elapsed_time)
                t_now = self.get_cumulative_time(dataset, elapsed_time + elapsed_time_now)
                lalc += (t_now - t_last) * v_last

            elapsed_time += elapsed_time_now
            v_last = v

        # at the end put the last value to what is remaining
        lalc += (1 - self.get_cumulative_time(dataset, elapsed_time)) * v_last

        return lalc

    def get_lalcs_of_schedule_on_datasets(self, schedule, datasets = None):
        scores = {}
        for i, ds in enumerate(datasets if datasets is not None else self.datasets):
            time_budget = int(self.ds_mf[ds]["time_budget"])
            elapsed_time = 0
            sub_schedule = []
            for a, b in schedule:
                consumed_time = self.TIMES[i, self.algos.index(a), self.budgets.index(b)]
                if np.isnan(consumed_time) or elapsed_time + consumed_time > time_budget:
                    break
                elapsed_time += consumed_time
                sub_schedule.append((a,b))
            score = self.get_lalc_of_schedule(ds, sub_schedule)
            scores[ds] = score
        return scores
    
    '''
        get single best schedule on a set of datasets based on dynamic programming
    '''
    def get_single_best_schedule(self, datasets = None):
        best_schedules = []
        best_scores = []

        highest_budget = max([int(self.ds_mf[ds]["time_budget"]) for ds in self.datasets])

        relevant_actions = set(it.product(self.algos, self.budgets))

        for d in tqdm(range(1, highest_budget + 1)):
            
            # find the maximum d' value among all schedules so that the action can be executed within d for all datasets
            best_score_for_new_schedule = best_scores[-1] if best_scores else 0
            best_new_schedule = best_schedules[-1].copy() if best_schedules else []
            obsolete_actions = []
            for action in relevant_actions:
                algo, budget = action
                j = self.algos.index(algo)
                k = self.budgets.index(budget)
                d_prime = min([d - int(np.ceil(self.TIMES[i,j,k] if not np.isnan(self.TIMES[i,j,k]) else 10**6)) for i, ds in enumerate(self.datasets)])

                # if the action is feasible, determine the new schedule and compute its performance
                if not np.isnan(d_prime) and d_prime > 0:
                    base_schedule = best_schedules[d_prime]
                    new_schedule = base_schedule + [action]
                    score_for_schedule = np.mean(list(self.get_lalcs_of_schedule_on_datasets(new_schedule, datasets).values()))
                    if score_for_schedule > best_score_for_new_schedule:
                        best_score_for_new_schedule = score_for_schedule
                        best_new_schedule = new_schedule
                    elif score_for_schedule <= min(best_scores[d_prime:]):
                        obsolete_actions.append(action)

            # set the best schedule
            best_schedules.append(best_new_schedule)
            best_scores.append(best_score_for_new_schedule)
            for a in obsolete_actions:
                relevant_actions.remove(a)

        return best_schedules[-1], best_scores[-1]
    
    def reset(self, ds_mf):
        self.observations = pd.DataFrame([], columns=["algorithm", "anchor", "time", "score_train", "score_valid"], dtype=object)
        self.remaining_time = int(ds_mf["time_budget"])
        self.num_train = int(ds_mf["train_num"])
        self.num_feat = int(ds_mf["feat_num"])
        self.ds_mf_cur = ds_mf
        self.saturation_point = None
    
    def add_observation(self, observation):
        if observation is not None:
            obs = list(observation)
            obs[0] = int(obs[0])
            self.observations.loc[len(self.observations)] = obs
            self.remaining_time -= observation[2]
            
            
    def get_instance_for_runtime_prediction(self, num_instances, num_features):
        return [num_instances, num_instances ** 2, num_features, num_features ** 2]
    
    def learn_runtimes(self):
        
        # learn runtime models
        Xs = {}
        ys = {}
        for ds, curves_on_ds in self.train_curves.items():
            num_instances = int(self.ds_mf[ds]["train_num"])
            num_features = int(self.ds_mf[ds]["feat_num"])
            for algo, curve in curves_on_ds.items():
                for b, t in zip(curve.training_data_sizes, curve.times):
                    if not algo in Xs:
                        Xs[algo] = []
                        ys[algo] = []
                    num_instances_here = b * num_instances
                    Xs[algo].append(self.get_instance_for_runtime_prediction(num_instances_here, num_features))
                    ys[algo].append(t)
        
        self.runtime_predictors = {}
        print("Building runtime models for all the algorithms.")
        for algo, X in tqdm(Xs.items()):
            X = np.array(X)
            y = np.array(ys[algo])
        
            rf = sklearn.ensemble.RandomForestRegressor()
            rf.fit(X, y)
            self.runtime_predictors[algo] = rf
    
    def get_runtimes_for_all_actions_on_current_dataset(self):
        fig, ax = plt.subplots()
                
        for algo, runtime_predictor in self.runtime_predictors.items():
            X = []
            for budget in self.budgets:
                X.append(self.get_instance_for_runtime_prediction(self.num_train * budget, self.num_feat))
            y_hat = runtime_predictor.predict(X)
            ax.plot(range(len(y_hat)), y_hat)
        plt.show()
    
    def get_algo_to_determine_saturation_point(self):
        return 39
    
    def get_missing_actions_to_compute_saturation_point_on_current_dataset(self):
        algo = self.get_algo_to_determine_saturation_point()
        seen_anchors = list(self.observations[self.observations["algorithm"] == algo]["anchor"])
        return [b for b in self.budgets if not b in seen_anchors]
        
    def get_saturation_budget_on_current_dataset(self):
        algo = self.get_algo_to_determine_saturation_point()
        df = self.observations[self.observations["algorithm"] == algo]
        scores = np.array(list(df["score_valid"]))
        best_score = max(scores)
        smallest_anchor_with_best_score = min(np.where(scores >= best_score)[0])
        return np.round(smallest_anchor_with_best_score * 0.1, 1)
    
    def set_saturation_point_if_possible(self):
        if self.saturation_point is not None:
            return
        if self.get_missing_actions_to_compute_saturation_point_on_current_dataset():
            return
        self.saturation_point = self.get_saturation_budget_on_current_dataset()

    def get_algorithms_not_evaluated_on_budget(self, budget):
        evaluated_algos = [str(int(a)) for a in pd.unique(self.observations[self.observations["anchor"] == budget]["algorithm"])]
        return [a for a in self.algos if not a in evaluated_algos]
    
    def get_algorithms_not_evaluated_on_saturation_budget(self):
        return self.get_algorithms_not_evaluated_on_budget(self.saturation_point)
    
class Agent():
    def __init__(self, number_of_algorithms):
        """
        Initialize the agent

        Parameters
        ----------
        number_of_algorithms : int
            The number of algorithms

        """
        ### TO BE IMPLEMENTED ###
        pass

    def reset(self, dataset_meta_features, algorithms_meta_features):
        """
        Reset the agents' memory for a new dataset

        Parameters
        ----------
        dataset_meta_features : dict of {str : str}
            The meta-features of the dataset at hand, including:
                usage = 'AutoML challenge 2014'
                name = name of the dataset
                task = 'binary.classification', 'multiclass.classification', 'multilabel.classification', 'regression'
                target_type = 'Binary', 'Categorical', 'Numerical'
                feat_type = 'Binary', 'Categorical', 'Numerical', 'Mixed'
                metric = 'bac_metric', 'auc_metric', 'f1_metric', 'pac_metric', 'a_metric', 'r2_metric'
                time_budget = total time budget for running algorithms on the dataset
                feat_num = number of features
                target_num = number of columns of target file (one, except for multi-label problems)
                label_num = number of labels (number of unique values of the targets)
                train_num = number of training examples
                valid_num = number of validation examples
                test_num = number of test examples
                has_categorical = whether there are categorical variable (yes=1, no=0)
                has_missing = whether there are missing values (yes=1, no=0)
                is_sparse = whether this is a sparse dataset (yes=1, no=0)

        algorithms_meta_features : dict of dict of {str : str}
            The meta_features of each algorithm, for example:
                meta_feature_0 = 0, 1, 2, …
                meta_feature_1 = 0, 1, 2, …
                meta_feature_2 = 0.000001, 0.0001 …


        Examples
        ----------
        >>> dataset_meta_features
        {'usage': 'AutoML challenge 2014', 'name': 'dataset01', 'task': 'regression',
        'target_type': 'Binary', 'feat_type': 'Binary', 'metric': 'f1_metric',
        'time_budget': '600', 'feat_num': '9', 'target_num': '6', 'label_num': '10',
        'train_num': '17', 'valid_num': '87', 'test_num': '72', 'has_categorical': '1',
        'has_missing': '0', 'is_sparse': '1'}
        >>> algorithms_meta_features
        {'0': {'meta_feature_0': '0', 'meta_feature_1': '0', meta_feature_2 : '0.000001'},
         '1': {'meta_feature_0': '1', 'meta_feature_1': '1', meta_feature_2 : '0.0001'},
         ...
         '39': {'meta_feature_0': '2', 'meta_feature_1': '2', meta_feature_2 : '0.01'},
         }
        """

        self.ds_mf = dataset_meta_features
        self.util.reset(dataset_meta_features)
        
        #self.util.get_runtimes_for_all_actions_on_current_dataset()
        
        ## SH-related stuff ##
        self.round = 1
        self.algos_alive = list(range(40))
        
    def meta_train(self, datasets_meta_features, algorithms_meta_features, train_learning_curves, validation_learning_curves, test_learning_curves):
        """
        Start meta-training the agent

        Parameters
        ----------
        datasets_meta_features : dict of {str : dict of {str : str}}
            Meta-features of meta-training datasets

        algorithms_meta_features : dict of {str : dict of {str : str}}
            The meta_features of all algorithms

        train_learning_curves : dict of {str : dict of {str : Learning_Curve}}
            TRAINING learning curves of meta-training datasets

        validation_learning_curves : dict of {str : dict of {str : Learning_Curve}}
            VALIDATION learning curves of meta-training datasets

        test_learning_curves : dict of {str : dict of {str : Learning_Curve}}
            TEST learning curves of meta-training datasets

        Examples:
        To access the meta-features of a specific dataset:
        >>> datasets_meta_features['dataset01']
        {'name':'dataset01', 'time_budget':'1200', ...}

        To access the validation learning curve of Algorithm 0 on the dataset 'dataset01' :

        >>> validation_learning_curves['dataset01']['0']
        <learning_curve.Learning_Curve object at 0x9kwq10eb49a0>

        >>> validation_learning_curves['dataset01']['0'].timestamps
        [196, 319, 334, 374, 409]

        >>> validation_learning_curves['dataset01']['0'].scores
        [0.6465293662860659, 0.6465293748988077, 0.6465293748988145, 0.6465293748988159, 0.6465293748988159]
        """
        self.algos = sorted(algorithms_meta_features.keys())
        
        self.util = Util(datasets_meta_features, algorithms_meta_features, train_learning_curves, validation_learning_curves, test_learning_curves)
        self.best_schedule, self.best_score = self.util.get_single_best_schedule()
        print(self.best_schedule)
        print(self.best_score)
        
        #self.util.learn_runtimes()
    
       
    def suggest(self, observation):
        self.util.add_observation(observation)
        return self.suggest_hard_coded_schedule(observation)
    
    def suggest_for_getting_saturation_point(self, observation):
        algo_to_determine_saturation_point = self.util.get_algo_to_determine_saturation_point()
        missing_budgets = self.util.get_missing_actions_to_compute_saturation_point_on_current_dataset()
        self.util.set_saturation_point_if_possible()
        if missing_budgets:
            return algo_to_determine_saturation_point, missing_budgets[0]
        
        budget = self.util.saturation_point - 0.1
        missing_algos = self.util.get_algorithms_not_evaluated_on_saturation_budget()
        while not missing_algos:
            budget += 0.1
            missing_algos = self.util.get_algorithms_not_evaluated_on_budget(budget)
        return missing_algos[0], budget
        
        
    def suggest_sh(self, observation): # run successive halving
        
        # get counts and current budget
        cnts = [np.count_nonzero(self.observations["algorithm"] == a) for a in self.algos_alive]
        
        # check whether last phase has been finished
        last_phase_finished = min(cnts) == self.round
        if last_phase_finished:
            self.round += 1
        
        # budget
        budget = min(1.0, np.round(0.1 * 2**(self.round - 1), 1))
        least_explored = np.argmin(cnts)
        return self.algos_alive[least_explored], budget

    def suggest_previously_computed_single_best_schedule(self, observation):
        schedule = self.best_schedule
        index = len(self.util.observations)
        if index < len(schedule):
            return schedule[index]
        print(f"{100 * self.util.remaining_time / int(self.ds_mf['time_budget'])}% of the budget unused")
        return 0, 1.0 # do nothing anymore
    
    def suggest_hard_coded_schedule(self, observation):
        #schedule = [('9', 0.2), ('39', 0.1),  ('9', 0.6),  ('5', 0.1),  ('39', 1.0),  ('5', 1.0),  ('35', 0.8),  ('23', 0.2)]
        schedule = [('9', 1.0), ('39', 1.0), ('32', 1.0), ('37', 1.0), ('38', 1.0), ('5', 1.0), ('23', 1.0)]
        index = len(self.util.observations)
        if index < len(schedule):
            return schedule[index]
        print(f"{100 * self.util.remaining_time / int(self.ds_mf['time_budget'])}% of the budget unused")
        return 0, 1.0 # do nothing anymore

    def suggest_hard_coded_ratios(self, observation):
        
        worst_case_fastest = [9, 4]
        best_ratios = [i for i in [35, 39, 3, 4, 9, 38, 36, 37, 1, 5, 33, 6, 34, 0, 8, 32, 2, 7, 30, 31, 17, 21, 20, 22, 29, 10, 11, 12, 13, 14, 15, 16, 23, 18, 19, 24, 25, 27, 28, 26] if not i in worst_case_fastest]
        
        if observation is None or len(self.observations) < len(worst_case_fastest):
            index = 0 if observation is None else len(self.observations)
            return worst_case_fastest[index], 1.0
        
        if len(self.observations) < len(worst_case_fastest) + len(best_ratios):
            return best_ratios[len(self.observations) - len(worst_case_fastest)], 1.0
        
        return 0, 1.0 # no nothing anymore