import pandas as pd
import numpy as np

default_offset = 100
class Artificial_Anomalies:
    # all_methods = [
    #         __class__.randomize_section,
    #         __class__.slightly_randomize_section,
    #         __class__.minimal
    #         ]

    # TODO: Ensure that random values are actually different from the original ones

    @staticmethod
    def randomize_section(col, section_size=5, offset=default_offset):
        ''' Easy to detect
            Set random values within a section '''
        anomalies_ranges, original_values = ([], [])
        original_values.append(col[offset-1:offset+section_size+1].copy())
        col[offset:offset+section_size] = np.random.randint(col.min(), col.max(), section_size)
        anomalies_ranges.append((offset-1,offset+section_size))
        # col_ground_truth = pd.Series(index=col.index, dtype=bool)

        # copy original column to preserve NaN 
        col_ground_truth = col.copy()
        col_ground_truth[ ~col_ground_truth.isnull() ] = False
        col_ground_truth[offset : offset + section_size] = True
        return col, anomalies_ranges, original_values, col_ground_truth


    @staticmethod 
    def slightly_randomize_section(col, section_size=5, offset=default_offset):
        ''' Harder to detect
            Slightly modify specified section (100:105 by default)
            (by adding or subtracting multiply of 8) '''
        anomalies_ranges, original_values = ([], [])
        original_values.append( col[ offset-1:offset+section_size+1 ].copy() )
        col[offset:offset+section_size] += np.random.randint(-3, 3, section_size) * 8 
        anomalies_ranges.append( (offset-1, offset+section_size) )
        # col_ground_truth = pd.Series(index=col.index, dtype=bool)

        # copy original column to preserve NaN 
        col_ground_truth = col.copy()
        col_ground_truth[ ~col_ground_truth.isnull() ] = False
        col_ground_truth[offset : offset + section_size] = True
        return col, anomalies_ranges, original_values, col_ground_truth


    @staticmethod
    def minimal(col, to_add=8, offset=default_offset):
        ''' Hard to detect.
            Modify a single value by adding 8 to it. '''
        anomalies_ranges, original_values = ([], [])

        original_values.append( col[offset-1:offset+2].copy() )
        col[offset] += to_add
        anomalies_ranges.append( (offset-1, offset+1) )

        # col_ground_truth = pd.Series(index=col.index, dtype=bool)

        # copy original column to preserve NaN 
        col_ground_truth = col.copy()
        col_ground_truth[ ~col_ground_truth.isnull() ] = False
        col_ground_truth[offset : offset + 1] = True
        return col, anomalies_ranges, original_values, col_ground_truth

