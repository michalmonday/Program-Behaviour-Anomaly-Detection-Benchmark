# class containing common methods for different detection system classes,
# as of 03/03/2022 there are 2 classes that inherit from it:
# - Unique_Transitions
# - LSTM_Autoencoder

import pandas as pd
import numpy as np
import logging
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

class Detection_Model:
    evaluation_metrics = [
        'anomaly_recall', 'false_positives_ratio', 'anomaly_count', 
        'detected_anomaly_count', 'non_anomaly_count', 'false_positives'
            ]

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

        melted_ground_truth = df_a_ground_truth_windowized.melt().dropna()
        # all_ground_truth = melted_ground_truth['value'].dropna()
        # consecutive_index_groups are used to avoid treating a single anomalous program counter as multiple anomalies
        # just because of the window/sequence size being larger than 1 
        consecutive_index_groups = __class__.get_consecutive_index_groups(x, index_list=melted_ground_truth.index.tolist())
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
        precision, recall, fscore, support = precision_recall_fscore_support(all_ground_truth, all_detection_results)
                # all_ground_truth.values.reshape(-1).tolis(), 
                # all_detection_results
                # )

        # anomaly_recall and false_positives_ratio are sufficient for evaluation of the anomaly
        # detection system in our case. However, it may be good idea to output the total vs detected
        # anomalies, and total non-anomalies vs false positives (just for the sake of verifying 
        # that evaluation metrics are calculated appropriately, it also gives more insight to us)
        tn, fp, fn, tp = confusion_matrix(all_ground_truth, all_detection_results).ravel()

        evaluation_metrics = {
            ####################################################
            # 2 MAIN evaluation metrics

            # what percent of anomalies will get detected
            'anomaly_recall' : recall[1],
            # what percent of normal program behaviour will be classified as anomalous
            'false_positives_ratio' : 1 - recall[0],

            #####################################################
            # Some additional metrics for verification purposes
            'anomaly_count' : fn + tp,
            'detected_anomaly_count' : tp,
            'non_anomaly_count' : tn + fp,
            'false_positives' : fp
                }
        return evaluation_metrics

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

        



