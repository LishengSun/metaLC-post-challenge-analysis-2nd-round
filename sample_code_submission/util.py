import numpy as np
import pandas as pd
import itertools as it
from tqdm import tqdm
import sklearn
import matplotlib.pyplot as plt


class Util:
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
                        self.TIMES[i, j, k] = induction_times[k]
                    else:
                        self.TIMES[i, j, k] = np.nan

        # compute scores of actions
        self.SCORES = np.zeros((len(datasets), len(algos), 10))
        for i, ds_id in enumerate(datasets):
            for j, algo_id in enumerate(algos):
                curve = test_curves[ds_id][algo_id].scores
                for k in range(10):
                    if len(curve) > k:
                        self.SCORES[i, j, k] = curve[k]
                    else:
                        self.SCORES[i, j, k] = np.nan

    def get_cumulative_time(self, dataset, t):
        t_0 = 60.0  # this is hard-coded in the scoring.py
        time_budget = int(self.ds_mf[dataset]["time_budget"])
        if t > time_budget:
            raise ValueError(
                f"Cannot compute cumulative time for t = {t} if time budget is only {time_budget}"
            )
        return np.log(1 + t / t_0) / np.log(1 + time_budget / t_0)

    def get_lalc_of_schedule(self, dataset, schedule):
        v_last = 0
        i = self.datasets.index(dataset)
        lalc = 0
        elapsed_time = 0

        for algo, budget in schedule:
            j = self.algos.index(algo)
            k = self.budgets.index(budget)
            v = self.SCORES[i, j, k]
            elapsed_time_now = self.TIMES[i, j, k]

            # ignore deteriorations
            if v < v_last:
                v = v_last

            if elapsed_time > 0:
                t_last = self.get_cumulative_time(dataset, elapsed_time)
                t_now = self.get_cumulative_time(
                    dataset, elapsed_time + elapsed_time_now
                )
                lalc += (t_now - t_last) * v_last

            elapsed_time += elapsed_time_now
            v_last = v

        # at the end put the last value to what is remaining
        lalc += (1 - self.get_cumulative_time(dataset, elapsed_time)) * v_last

        return lalc

    def get_lalcs_of_schedule_on_datasets(self, schedule, datasets=None):
        scores = {}
        for i, ds in enumerate(datasets if datasets is not None else self.datasets):
            time_budget = int(self.ds_mf[ds]["time_budget"])
            elapsed_time = 0
            sub_schedule = []
            for a, b in schedule:
                consumed_time = self.TIMES[
                    i, self.algos.index(a), self.budgets.index(b)
                ]
                if (
                    np.isnan(consumed_time)
                    or elapsed_time + consumed_time > time_budget
                ):
                    break
                elapsed_time += consumed_time
                sub_schedule.append((a, b))
            score = self.get_lalc_of_schedule(ds, sub_schedule)
            scores[ds] = score
        return scores

    """
        get single best schedule on a set of datasets based on dynamic programming
    """

    def get_single_best_schedule(self, datasets=None):
        best_schedules = []
        best_scores = []

        highest_budget = max(
            [int(self.ds_mf[ds]["time_budget"]) for ds in self.datasets]
        )

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
                d_prime = min(
                    [
                        d
                        - int(
                            np.ceil(
                                self.TIMES[i, j, k]
                                if not np.isnan(self.TIMES[i, j, k])
                                else 10 ** 6
                            )
                        )
                        for i, ds in enumerate(self.datasets)
                    ]
                )

                # if the action is feasible, determine the new schedule and compute its performance
                if not np.isnan(d_prime) and d_prime > 0:
                    base_schedule = best_schedules[d_prime]
                    new_schedule = base_schedule + [action]
                    score_for_schedule = np.mean(
                        list(
                            self.get_lalcs_of_schedule_on_datasets(
                                new_schedule, datasets
                            ).values()
                        )
                    )
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
        self.observations = pd.DataFrame(
            [], columns=["algorithm", "anchor", "time", "score_train", "score_valid"],dtype=object
        )
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

    def get_advanced_instance_for_runtime_prediction(
        self, num_instances, num_features, first_anchor
    ):
        return [
            num_instances,
            num_instances ** 2,
            num_features,
            num_features ** 2,
            first_anchor,
            first_anchor ** 2,
        ]

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
                    Xs[algo].append(
                        self.get_instance_for_runtime_prediction(
                            num_instances_here, num_features
                        )
                    )
                    ys[algo].append(t)

        self.runtime_predictors = {}
        print("Building runtime models for all the algorithms.")
        for algo, X in tqdm(Xs.items()):
            X = np.array(X)
            y = np.array(ys[algo])

            rf = sklearn.ensemble.RandomForestRegressor()
            rf.fit(X, y)
            self.runtime_predictors[algo] = rf

    def learn_advanced_runtimes(self):
        """learns runtime models with first anchor point as additional feature"""

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
                    first_anchor = curve.times[0]
                    Xs[algo].append(
                        self.get_advanced_instance_for_runtime_prediction(
                            num_instances_here, num_features, first_anchor
                        )
                    )
                    ys[algo].append(t)

        self.advanced_runtime_predictors = {}
        print("Building runtime models for all the algorithms.")
        for algo, X in tqdm(Xs.items()):
            X = np.array(X)
            y = np.array(ys[algo])

            rf = sklearn.ensemble.RandomForestRegressor()
            rf.fit(X, y)
            self.advanced_runtime_predictors[algo] = rf

    def get_runtimes_for_all_actions_on_current_dataset(self):
        fig, ax = plt.subplots()

        for algo, runtime_predictor in self.runtime_predictors.items():
            X = []
            for budget in self.budgets:
                X.append(
                    self.get_instance_for_runtime_prediction(
                        self.num_train * budget, self.num_feat
                    )
                )
            y_hat = runtime_predictor.predict(X)
            ax.plot(range(len(y_hat)), y_hat)
        plt.show()

    def get_advanced_runtimes_for_all_actions_on_current_dataset_where_available(self):
        # returns advanced runtime predictions for each algorithm where at least one anchor has been observed already
        fig, ax = plt.subplots()

        for algo in self.observations["algorithm"].unique():
            runtime_predictor = self.advanced_runtime_predictors[str(int(algo))]
            print(self.observations)
            seen_anchors = list(
                self.observations[self.observations["algorithm"] == algo]["anchor"]
            )
            print("seen anchors", seen_anchors)
            if len(seen_anchors) < 1:
                continue

            first_anchor = seen_anchors[0]
            print("first anchor", first_anchor)
            X = []
            for budget in self.budgets:
                X.append(
                    self.get_advanced_instance_for_runtime_prediction(
                        self.num_train * budget, self.num_feat, first_anchor
                    )
                )
            y_hat = runtime_predictor.predict(X)
            print(len(y_hat), y_hat)
            ax.plot(range(len(y_hat)), y_hat, label=algo)
        ax.legend()
        plt.show()

    def get_algo_to_determine_saturation_point(self):
        return 39

    def get_missing_actions_to_compute_saturation_point_on_current_dataset(self):
        algo = self.get_algo_to_determine_saturation_point()
        seen_anchors = list(
            self.observations[self.observations["algorithm"] == algo]["anchor"]
        )
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
        evaluated_algos = [
            str(int(a))
            for a in pd.unique(
                self.observations[self.observations["anchor"] == budget]["algorithm"]
            )
        ]
        return [a for a in self.algos if not a in evaluated_algos]

    def get_algorithms_not_evaluated_on_saturation_budget(self):
        return self.get_algorithms_not_evaluated_on_budget(self.saturation_point)
