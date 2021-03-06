

import utils
from detection_model import Detection_Model
from unique_transitions import unique_transitions

import pandas as pd
import numpy as np

df_a_ground_truth = pd.DataFrame(
    [ 
        # 1 column = 1 program
        # 1 cell   = label for 1 program counter (not 1 window)
        [False, True,  True], 
        [True,  True,  False],
        [False, False, True],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, True,  False],
        [False, True,  np.NaN]
    ],
    columns=['test.pc', 'test2.pc', 'test3.pc'])

results_ut = np.array([
        [False, False, False], # some of the starting rows will be deleted
        [False, False, False], # depending on the chosen sequence size
        [False, False, False], # e.g. if seq_size is 3 then 2 rows will be deleted 
        [False, False, True],
        [False, False, False],
        [False, False, False],
        [True, False, False],
        [False, False]
    ]).T

seq_size = 5

results_ut = results_ut[seq_size-1:]


# df_a_ground_truth_windowized = utils.windowize_ground_truth_labels(
#         df_a_ground_truth,
#         seq_size # window/sequence size
#         )

df_a_ground_truth_windowized = utils.windowize_ground_truth_labels_2(
        df_a_ground_truth,
        seq_size # window/sequence size
        )


# df_id_mask = utils.get_anomaly_identifier_mask(df_a_ground_truth)

print('\n\nOriginal df_a_ground_truth:')
print(df_a_ground_truth)

print('\n\ndf_a_ground_truth_windowized:')
print(df_a_ground_truth_windowized)

dm = Detection_Model()
# not_detected, em = dm.evaluate_all(results_ut, df_a_ground_truth_windowized)
not_detected, em = dm.evaluate_all_2(results_ut, df_a_ground_truth_windowized)

print('\n\nRESULTS:')
print(dm.format_evaluation_metrics(em))

print('\nNot detected:')
print(not_detected)
