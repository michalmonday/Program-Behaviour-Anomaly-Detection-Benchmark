import pandas as pd
import numpy as np
import random

# def matching_values(df, df2):
#     return df[df==df2]

min_section_size = 2
max_section_size = 10

class Artificial_Anomalies:
    # all_methods = [
    #         __class__.randomize_section,
    #         __class__.slightly_randomize_section,
    #         __class__.minimal
    #         ]


    @staticmethod
    def randomize_section(col, section_size=None, offset=None):
        ''' Easy to detect
            Set random values within a section '''
        if section_size is None:
            section_size = __class__.generate_random_section_size(col)
        if offset is None:
            offset = __class__.generate_random_offset(col, section_size)

        anomalies_ranges, original_values = ([], [])
        original_values.append(col[offset-1:offset+section_size+1].copy())
        anomalies_ranges.append((offset-1,offset+section_size))

        new_values = []
        for i in range(section_size):
            original_value = original_values[-1].iloc[i+1]
            val = original_value
            while val == original_value:
                val = random.randint(col.min(), col.max())
            new_values.append(val)

        col[offset:offset+section_size] = new_values
        col_ground_truth = __class__.generate_ground_truth_column(col, offset, section_size)
        return col, anomalies_ranges, original_values, col_ground_truth


    @staticmethod 
    def slightly_randomize_section(col, section_size=None, offset=None):
        ''' Harder to detect
            Slightly modify specified section (100:105 by default)
            (by adding or subtracting multiply of 8) '''
        if section_size is None:
            section_size = __class__.generate_random_section_size(col)
        if offset is None:
            offset = __class__.generate_random_offset(col, section_size)

        anomalies_ranges, original_values = ([], [])
        original_values.append( col[ offset-1:offset+section_size+1 ].copy() )
        anomalies_ranges.append( (offset-1, offset+section_size) )

        # col[offset:offset+section_size] += np.random.randint(-3, 3, section_size) * 8 
        values_to_add = [-24, -16, -8, 8, 16, 24]
        col[offset:offset+section_size] += np.random.choice(values_to_add, section_size)

        col_ground_truth = __class__.generate_ground_truth_column(col, offset, section_size)
        return col, anomalies_ranges, original_values, col_ground_truth


    @staticmethod
    def minimal(col, offset=None):
        ''' Hard to detect.
            Modify a single value by adding 8 to it. '''
        if offset is None:
            offset = __class__.generate_random_offset(col, 1)

        anomalies_ranges, original_values = ([], [])
        original_values.append( col[offset-1:offset+2].copy() )
        anomalies_ranges.append( (offset-1, offset+1) )

        # add or subtract 8 from a single program counter
        col[offset] += 8 if random.randint(1,2) == 1 else -8

        col_ground_truth = __class__.generate_ground_truth_column(col, offset, 1)
        return col, anomalies_ranges, original_values, col_ground_truth


    #########################################################
    #  Helper functions

    @staticmethod
    def generate_ground_truth_column(col, offset, section_size):
        # copy original column to preserve NaN 
        gt = col.copy()
        gt[ ~gt.isnull() ] = False
        gt[offset : offset + section_size] = True
        return gt

    @staticmethod
    def generate_random_offset(col, section_size):
        ''' Disallowing to modify the first and the last program counter
            helps with operating on data frames (e.g. melting all df columns
            into one without worrying that anomalies from the end of one run
            will become consecutive to anomalies from the begining of another
            run, which would disrupt evaluation). '''
        # -1 will disallow modifying the last program counter
        # col.count() is used instead of col.shape[0] because it may
        # have NaN values
        max_offset = col.count() - section_size - 1
        # 1 will disallow modifying the first program counter
        return random.randint(1, max_offset)

    @staticmethod
    def generate_random_section_size(col):
        max_size = min(10, col.shape[0]-2)
        return random.randint(2, max_size)

# Testing code
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    df = pd.DataFrame([1,2,3]*5)
    ax = df.iloc[:,0].plot()
    col, anomalies_ranges, original_values, col_ground_truth = Artificial_Anomalies.randomize_section(df.iloc[:,0], offset=2, section_size=10)
    col.plot(ax=ax)
    # plt.show()
    
    # print(col)
    # print()
    # print(col_ground_truth)
    # print()
    # print(original_values)

    unchanged = col[col == col_ground_truth].index.values.tolist()
    print(unchanged)

