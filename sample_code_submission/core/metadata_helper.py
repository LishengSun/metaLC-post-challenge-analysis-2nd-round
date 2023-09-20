import pandas as pd
import numpy as np

from sklearn.preprocessing import minmax_scale


def transform_metadata_to_dataframe(datasets_meta_features):
    if (isinstance(list(datasets_meta_features.values())[0], dict)):
        datasets_data = [{**value, 'dataset_id': key}
                         for key, value in datasets_meta_features.items()]
    else:
        datasets_meta_features["dataset_id"] = 0
        datasets_data = [datasets_meta_features]

    datasets_df = pd.DataFrame(datasets_data)

    # Generate a complete DataFrame of all the available data.
    return datasets_df


""" def transform_metadata_to_dataframe(datasets_meta_features):
    if (isinstance(list(datasets_meta_features.values())[0], dict)):
        datasets_data = [{**value, 'dataset_id': key}
                         for key, value in datasets_meta_features.items()]
    else:
        datasets_meta_features["dataset_id"] = 0
        datasets_data = [datasets_meta_features]

    algorithms_data = [{**value, 'algorithm_id': key}
                       for key, value in algorithms_meta_features.items()]

    datasets_df = pd.DataFrame(datasets_data)
    algorithms_df = pd.DataFrame(algorithms_data)

    # Generate a complete DataFrame of all the available data.
    return datasets_df.assign(t=1).merge(algorithms_df.assign(t=1), on='t').drop(columns='t') """


def get_learning_score(learning_curves, datasets_meta_features):
    data_list = []

    for dataset_id, value in learning_curves.items():
        time_budget = int(
            datasets_meta_features[dataset_id]['time_budget'])

        for algorithm_id, algorithm_value in value.items():
            scores = np.array(algorithm_value.scores)
            times = np.array(algorithm_value.times)

            if (len(scores) == 0):
                scores = np.zeros((10,))

            if (len(times) == 0):
                times = np.full((10,), time_budget)

            data_list.append({
                'dataset_id': dataset_id,
                'algorithm_id': algorithm_id,
                'score': times[0]
            })

    return pd.DataFrame(data_list)


def compute_alc(df, total_time_budget, normalize_t=True):
    alc = 0.0

    for i in range(len(df)):
        if df.iloc[i]['test_score_of_best_algorithm_so_far'] == 'None':
            continue
        if i == 0:
            if normalize_t:
                alc += df.iloc[i]['test_score_of_best_algorithm_so_far'] * \
                    (1-df.iloc[i]['normalized_cumulative_t'])
            else:
                alc += df.iloc[i]['test_score_of_best_algorithm_so_far'] * \
                    (total_time_budget-df.iloc[i]['cumulative_t'])
        elif i > 0:
            if normalize_t:
                alc += (df.iloc[i]['test_score_of_best_algorithm_so_far']-df.iloc[i-1]
                        ['test_score_of_best_algorithm_so_far']) * (1-df.iloc[i]['normalized_cumulative_t'])
            else:
                alc += (df.iloc[i]['test_score_of_best_algorithm_so_far']-df.iloc[i-1]
                        ['test_score_of_best_algorithm_so_far']) * (total_time_budget-df.iloc[i]['cumulative_t'])
    return round(alc, 2)
