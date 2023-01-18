# class containing common methods for different detection system classes,
# as of 03/03/2022 there are 2 classes that inherit from it:
# - Unique_Transitions
# - LSTM_Autoencoder

import pandas as pd
import numpy as np
import logging
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from functools import reduce
from abc import ABC, abstractmethod

class Detection_Model(ABC):
    evaluation_metrics = [
        'anomaly_recall', 'false_positives_ratio', 'anomaly_count', 
        'detected_anomaly_count', 'non_anomaly_count', 'false_positives'
            ]
    
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_consecutive_index_groups(series, index_list=[]):
        ''' This function turns pd.Series like this:
            101     False
            102     False
            652     False
            653     False
            654     False
            1203    False
            1204    False
            1205    False
            1206    False

        Into this:
            return = [
                [101 102] ,
                [652 653 654] ,
                [1203 1204 1205 1206]
            ] 

        Index list allows to substitute indices.
        '''
        i_series = pd.Series(series.index.values)
        groups = []
        for k, g in i_series.groupby(i_series.diff().ne(1).cumsum()):
            values = g.values
            if index_list:
                values = [index_list.index(v) for v in values]
            groups.append(values)
        return groups

    @staticmethod
    def get_consecutive_index_groups_2(series, index_list=[]):
        ''' This function turns pd.Series like this:
            101     [0,2]
            102     [2]
            652     [1,9]
            653     [1]
            654     [1,9]

        Into this:
            return = [
                (2,[101 102]),
                (1,[652 653 654])
            ] 

        Index list allows to substitute indices.
        '''
        i_series = pd.Series(series.index.values)
        groups = []
        for k, g in i_series.groupby(i_series.diff().ne(1).cumsum()):
            values = g.values
            if index_list:
                values = [index_list.index(v) for v in values]
            groups.append(values)
        return groups

    def evaluate_all(self, results_all, df_a_ground_truth_windowized):
        ''' results_all = return of predict_all function 
            This function returns a dictionary of evaluation metrics:
            - anomaly_recall
            - false_positives_ratio

            - anomaly_count
            - detected_anomaly_count
            - non_anomaly_count
            - false_positives

            The returned dictionary can be supplied directly to 
            "format_evaluation_metrics" method, to make it suitable
            for printing/logging. '''
        # concatinate detection results and ground truth labels from 
        # all test examples (in other words, flatten nested list)
        all_detection_results = [val for results in results_all for val in results]

        x = df_a_ground_truth_windowized
        # get indices of all consecutive anomaly duplicates from all runs and merge them together into one pd.Series
        x = x[ x[ x.shift(1) == x ] == True ].melt().drop('variable', axis=1)['value'].dropna()

        # all_ground_truth = df_a_ground_truth_windowized.melt(value_name='melted').drop('variable', axis=1).dropna()['melted']

        melted_ground_truth = pd.melt(df_a_ground_truth_windowized.reset_index(), id_vars=['index']).dropna()
        # all_ground_truth = melted_ground_truth['value'].dropna()
        # consecutive_index_groups are used to avoid treating a single anomalous program counter as multiple anomalies
        # just because of the window/sequence size being larger than 1 
        consecutive_index_groups = __class__.get_consecutive_index_groups_2(x, index_list=melted_ground_truth.index.tolist())
        for group in consecutive_index_groups:
            # pi = preserved index (of all_detection_results)
            pi = group[0] - 1
            try:
                if not all_detection_results[pi]:
                    # set the predicted value at the first index of truly anomalous consecutive window sequence to True
                    # if any of the windows (within sequence) was predicted anomalous
                    all_detection_results[pi] = any(all_detection_results[pi:group[-1]+1])
            except Exception as e:
                logging.error(e)
                import pdb; pdb.set_trace()

            # set all consecutive anomalous windows to be normal (except the first, preserved index)
            all_detection_results[group[0]:group[-1]+1] = [False] * len(group)

            # ugly because of mixing positional and label based indexing (which requires index slicing since
            # depreciation of "ix" method)
            melted_ground_truth.loc[ melted_ground_truth.index[ group[0]:group[-1]+1 ]  , 'value'] = False
            
        # all_detection_results[1] = True # TODO: DELETE (it allowed verifying correctness of evaluation metrics)
        all_ground_truth = melted_ground_truth['value'].values.reshape(-1).tolist()
        # import pdb; pdb.set_trace()

        #precision, recall, fscore, support = precision_recall_fscore_support(all_ground_truth, all_detection_results, zero_division=0)
        #        # all_ground_truth.values.reshape(-1).tolis(), 
        #        # all_detection_results
        #        # )

        ## anomaly_recall and false_positives_ratio are sufficient for evaluation of the anomaly
        ## detection system in our case. However, it may be good idea to output the total vs detected
        ## anomalies, and total non-anomalies vs false positives (just for the sake of verifying 
        ## that evaluation metrics are calculated appropriately, it also gives more insight to us)
        #tn, fp, fn, tp = confusion_matrix(all_ground_truth, all_detection_results).ravel()

        #evaluation_metrics = {
        #    ####################################################
        #    # 2 MAIN evaluation metrics

        #    # what percent of anomalies will get detected
        #    'anomaly_recall' : recall[1],
        #    # what percent of normal program behaviour will be classified as anomalous
        #    'false_positives_ratio' : 1 - recall[0],

        #    #####################################################
        #    # Some additional metrics for verification purposes
        #    'anomaly_count' : fn + tp,
        #    'detected_anomaly_count' : tp,
        #    'non_anomaly_count' : tn + fp,
        #    'false_positives' : fp
        #        }
        evaluation_metrics = utils.labels_to_evaluation_metrics(all_ground_truth, all_detection_results)
        mgt = melted_ground_truth['value'].reset_index(drop=True)
        not_detected_anomalies = melted_ground_truth.iloc[ mgt[ (mgt == True) & (pd.Series(all_detection_results) == False) ].index ]
        # if not not_detected_anomalies.empty:
        #     logging.info('Not detected anomalies:')
        #     logging.info(not_detected_anomalies)
        return not_detected_anomalies, evaluation_metrics

    def evaluate_all_2(self, results_all, df_a_ground_truth_windowized, windows_counts=None):
        ''' results_all = return of predict_all function 
            This function returns a dictionary of evaluation metrics:
            - anomaly_recall
            - false_positives_ratio

            - anomaly_count
            - detected_anomaly_count
            - non_anomaly_count
            - false_positives

            The returned dictionary can be supplied directly to 
            "format_evaluation_metrics" method, to make it suitable
            for printing/logging. 
            
            windows_counts is used for performance. results_all now contains results for unique windows,
            and windows_counts contains how many times each window was repeated in the test/abnormal dataset.
            We can then multiply both and get evaluation metrics for all windows without the need to run prediction
            for duplicate windows, this should speed up evaluation process. '''
        # concatinate detection results and ground truth labels from 
        # all test examples (in other words, flatten nested list)
        all_detection_results = [val for results in results_all for val in results]

        # x = df_a_ground_truth_windowized.copy()
        # # get indices of all consecutive anomaly duplicates from all runs and merge them together into one pd.Series
        # # x = x[ x[ x.shift(1) == x ] == True ].melt().drop('variable', axis=1)['value'].dropna()
        # x[:] = x.shift(1, fill_value=set()).values & x.values# ].melt().drop('variable', axis=1)['value'].dropna()

        # all_ground_truth = df_a_ground_truth_windowized.melt(value_name='melted').drop('variable', axis=1).dropna()['melted']

        melted_ground_truth = pd.melt(df_a_ground_truth_windowized.reset_index(), id_vars=['index']).dropna()
        melted_windows_counts = pd.melt(pd.DataFrame(windows_counts).T.reset_index(), id_vars=['index']).dropna()
        false_positives = melted_ground_truth[ np.where(melted_ground_truth.value.values, False, all_detection_results) ]
        false_positives_windows_counts = melted_windows_counts.loc[ false_positives.index ]
        non_anomalous = melted_ground_truth[ melted_ground_truth.value == set() ]
        non_anomalous_windows_counts = melted_windows_counts.loc[ non_anomalous.index ]
        # non_anomaly_count = non_anomalous.shape[0]
        # import pdb; pdb.set_trace()
        non_anomaly_count = non_anomalous_windows_counts.value.sum() #non_anomalous * non_anomalous_windows_counts # TODO: check if it works well
        # false_positives_count = false_positives.shape[0]
        false_positives_count = false_positives_windows_counts.value.sum() # false_positives * false_positives_windows_counts # TODO: check if it works well

        # get all anomaly ids 
        all_anomalies = reduce(lambda s, s2: s|s2, melted_ground_truth.value.values)

        # import pdb; pdb.set_trace()

        # get all detected anomaly ids (even if they were detected in 1 window despite being able to be detected in many)
        try:
            detected_anomalies = reduce(lambda s, s2: s|s2, melted_ground_truth[all_detection_results].value.values) # TODO: melted_ground_truth[ all_detection_results ].value.drop_duplicates().values (will be faster I guess)
        except TypeError:
            # TypeError happens when iterable supplied to "reduce" is empty 
            detected_anomalies = set()
        
        # set below is mainly for printing not detected anomaly counts
        # not_detected_anomalies_set = all_anomalies - detected_anomalies
        anomaly_recall = len(detected_anomalies) / len(all_anomalies)

        # Series below is mainly for plotting not detected regions
        x = melted_ground_truth.value.apply(lambda x: x - detected_anomalies)
        not_detected_anomalies = melted_ground_truth[x!=set()]
        # not_detected_anomalies.drop_duplicates(subset=['value'], inplace=True)
        not_detected_anomalies = not_detected_anomalies[ (not_detected_anomalies.value - not_detected_anomalies.value.shift(1)) != set() ]


        evaluation_metrics = {
            ####################################################
            # 2 MAIN evaluation metrics

            # what percent of anomalies will get detected
            'anomaly_recall' : anomaly_recall,
            # what percent of normal program behaviour will be classified as anomalous
            'false_positives_ratio' : false_positives_count / non_anomaly_count,

            #####################################################
            # Some additional metrics for verification purposes
            'anomaly_count' : len(all_anomalies),
            'detected_anomaly_count' : len(detected_anomalies),
            'non_anomaly_count' : non_anomaly_count,
            'false_positives' : false_positives_count
            }
        # if not not_detected_anomalies.empty:
        #     logging.info('Not detected anomalies:')
        #     logging.info(not_detected_anomalies)
        return not_detected_anomalies, evaluation_metrics



    def format_evaluation_metrics(self, em):
        ''' em = evaluation metrics dict.
            This function converts a dict returned from "evaluate_all" to a meaningful string.
            Which then can be printed/logged. '''
        anomaly_recall = em['anomaly_recall']
        false_positives_ratio = em['false_positives_ratio']
        anomaly_count = em['anomaly_count']
        detected_anomaly_count = em['detected_anomaly_count']
        non_anomaly_count = em['non_anomaly_count']
        false_positives = em['false_positives']
        return f'anomaly_recall={anomaly_recall:.2f} ({detected_anomaly_count}/{anomaly_count}) false_positives_ratio={false_positives_ratio} ({false_positives}/{non_anomaly_count})'

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, df_a):
        pass

    def predict_all(self, abnormal_windows_all_files):
        return [self.predict(windows.values) for windows in abnormal_windows_all_files]



