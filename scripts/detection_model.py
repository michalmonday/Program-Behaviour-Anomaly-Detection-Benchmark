# class containing common methods for different detection system classes,
# as of 03/03/2022 there are 2 classes that inherit from it:
# - Unique_Transitions
# - LSTM_Autoencoder

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

class Detection_Model:
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
            This function returns 2 evaluation metrics that really matter
            for anomaly detection systems:
            - anomaly recall
            - false anomalies (referred to as "false positives" in some papers)
        '''
        # results = []
        # for col_a, col_a_name in zip(results_all, df_a_ground_truth):
        #     result = self.evaluate(col_a, df_a_ground_truth[col_a_name])
        #     results.append(result)

        
        # windowize the ground truth labels (so they determine if the whole window/sequence was anomalous)
        # 

        # concatinate detection results and ground truth labels from 
        # all test examples (in other words, flatten nested list)
        all_detection_results = [val for results in results_all for val in results]
        # all_ground_truth = df_a_ground_truth_windowized.melt(value_name='melted').drop('variable', axis=1).dropna()[['melted']].values.reshape(-1).tolist()
        
        
        

        x = df_a_ground_truth_windowized
        # get indices of all consecutive anomaly duplicates from all runs and merge them together into one pd.Series
        x = x[ x[ x.shift(1) == x ] == True ].melt(value_name='melted').drop('variable', axis=1)['melted'].dropna()
        #.reset_index(drop=True)
        


        # df_a_ground_truth_windowized = df_a_ground_truth_windowized[non_consecutive_anomalies]
        # consecutive_duplicates_indices = s[s==True].index.values
        # s = (x[ x.shift(1) == x ] == True).iloc[:,0]
        # consecutive_duplicates_indices = s[s==True].index.values

        all_ground_truth = df_a_ground_truth_windowized.melt(value_name='melted').drop('variable', axis=1).dropna()['melted']#.values.reshape(-1).tolist()
        consecutive_index_groups = __class__.get_consecutive_index_groups(x, index_list=all_ground_truth.index.tolist())
        for group in consecutive_index_groups:
            # pi = preserved index (of all_detection_results)
            pi = group[0] - 1
            if not all_detection_results[pi]:
                # set the predicted value at the first index of truly anomalous consecutive window sequence to True
                # if any of the windows (within sequence) was predicted anomalous
                all_detection_results[pi] = any(all_detection_results[pi:group[-1]+1])

            # set all consecutive anomalous windows to be normal (except the first, preserved index)
            all_detection_results[group[0]:group[-1]+1] = [False] * len(group)
            all_ground_truth.iloc[group[0]:group[-1]+1] = False
            


        # all_detection_results[0] = True # TODO: DELETE (it allowed verifying correctness of evaluation metrics)

        all_ground_truth = all_ground_truth.values.reshape(-1).tolist()

        precision, recall, fscore, support = precision_recall_fscore_support(all_ground_truth, all_detection_results)

        # what percent of anomalies will get detected
        anomaly_recall = recall[1]
        # what percent of normal program behaviour will be classified as anomalous
        inverse_normal_recall = 1 - recall[0]

        return anomaly_recall, inverse_normal_recall
