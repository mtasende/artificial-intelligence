import re
import pandas as pd
import numpy as np


def extract_df(results):
    columns = [
        'problem',
        'actions',
        'expansions',
        'goal_tests',
        'new_nodes',
        'path_length',
        'time_s'
    ]
    pattern = 'Solving Air Cargo Problem (\d) using (.*)\.\.\.\s*'
    pattern = pattern + '# Actions   Expansions   Goal Tests   New Nodes\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)'
    pattern = pattern + '\s*Plan length:\s*(\d+)\s*Time elapsed in seconds:\s*(\d+\.\d+)'

    results_list = re.findall(pattern, results)
    results_t = tuple(zip(*results_list))
    return pd.DataFrame(np.array((results_t[0],) + results_t[2:]).T,
                        index=results_t[1],
                        columns=columns).transform(pd.to_numeric)


def process_results_file(filepath):
    with open(filepath, 'r') as file:
        res = file.read()
    return extract_df(res)


def get_required_df(res_df):
    required_columns = ['problem', 'actions', 'expansions', 'time_s', 'path_length']
    resr_df = res_df.copy()
    resr_df.time_s = np.round(resr_df.time_s, 3)
    required_res = resr_df[required_columns].set_index('problem', append=True).unstack()
    return required_res.reorder_levels([1, 0], axis=1)
