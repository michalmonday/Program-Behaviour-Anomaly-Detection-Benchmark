import pandas as pd
import numpy as np
import random
import utils
import logging

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
    def randomize_section(col, col_instr, instruction_types, section_size=None, offset=None):
        print('shapes', col.shape, col_instr.shape)
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
        new_instr = []
        for i in range(section_size):
            original_value = original_values[-1].iloc[i+1]
            val = original_value
            while val == original_value:
                val = random.randint(col.min(), col.max())
            new_values.append(val)
            new_instr.append(random.choice(list(instruction_types.keys())))

        col[offset:offset+section_size] = new_values
        try:
            col_instr[offset:offset+section_size] = new_instr
        except Exception as e:
            import pdb; pdb.set_trace()
            print(e)
        col_ground_truth = __class__.generate_ground_truth_column(col, offset, section_size)
        return col, col_instr, anomalies_ranges, original_values, col_ground_truth


    @staticmethod 
    def slightly_randomize_section(col, col_instr, instruction_types, section_size=None, offset=None):
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
        col_instr[offset:offset+section_size] += np.random.choice(list(instruction_types.keys()), section_size)

        col_ground_truth = __class__.generate_ground_truth_column(col, offset, section_size)
        return col, col_instr, anomalies_ranges, original_values, col_ground_truth


    @staticmethod
    def minimal(col, col_instr, instruction_types, offset=None):
        ''' Hard to detect.
            Modify a single value by adding 8 to it. '''
        if offset is None:
            offset = __class__.generate_random_offset(col, 1)

        anomalies_ranges, original_values = ([], [])
        original_values.append( col[offset-1:offset+2].copy() )
        anomalies_ranges.append( (offset-1, offset+1) )

        # add or subtract 8 from a single program counter
        col[offset] += 8 if random.randint(1,2) == 1 else -8
        col_instr[offset] = random.choice(list(instruction_types.keys()))

        col_ground_truth = __class__.generate_ground_truth_column(col, offset, 1)
        return col, col_instr,  anomalies_ranges, original_values, col_ground_truth



    # @staticmethod
    # def reduce_loops(col):
    #     ''' Attempts to find and reduce all loops (starting with the largest ones). 
    #         This may be really slow especially for large sequences. 
    #         There should be a warning about it being displayed. '''
    #     col = col.dropna()
    #     # x = s.rolling(3).apply(is_repetition)
    #     # get all possible sliding windows for each possible window size
    #     # starting with size equal to half length of sequence and ending with 2
    #     logging.info(f'Reducing loops for {col.name}')
    #     for window_size in reversed(range(2, col.shape[0]//2 + 1)):
    #         unique_windows = utils.pc_df_to_sliding_windows(col, window_size, unique=True).dropna()
    #         for i, window in unique_windows.iterrows():
    #             order = window.values
    #             def is_repetition(rol):
    #                 return np.array_equal(rol.values, order)
    #             try:
    #                 reps = col.rolling(window_size).apply(is_repetition)
    #                 reps_count = reps.sum() - 1
    #                 if reps_count:
    #                     logging.info(f'window_size={window_size} had {reps_count} repetitions.')
    #                     import pdb; pdb.set_trace()
    #             except Exception as e:
    #                 logging.error(e)
    #                 import pdb; pdb.set_trace()

    @staticmethod
    def reduce_loops(col, col_instr, instruction_types, min_iteration_size=2):
        ''' It must return col_ground_truth.
            I have to decide which part to mark as anomalous:
            - only the last index of first loop iteration + next index that follows it
            - the whole first loop iteration + surrounding indices
             '''
        orig_size = col.shape[0]
        orig_size_not_null = col[col.notnull()].shape[0]
        all_reduced_ranges = []
        all_reduced_rows = set()
        anomaly_ranges = []
        all_first_iteration_ranges = [] # is used for setting ground truth labels (to True, meaning anomalous)
        # gt = col.copy() # gt = ground_truth
        # gt[ gt.notnull() ] = False
        gt = pd.Series([np.NaN]*col.shape[0])
        # iteration size is the number of program counter values that are part
        # of a single iteration of a loop
        for iteration_size in reversed(range(min_iteration_size, col.shape[0]//2 + 1)):
            # col shape is dynamic as more loops are reduced.
            # (so the condition below improves performance)
            if iteration_size > (col.shape[0] // 2 + 1):
                continue
            col, col_instr, reduced_ranges, reduced_rows, first_iteration_ranges = __class__.reduce_loops_single_size(col, col_instr, iteration_size, all_reduced_rows)
            # print(f'Size: {iteration_size}\ncol:\n{col}\nreduced_ranges:\n{reduced_ranges}\n')
            all_reduced_rows |= reduced_rows 
            # for i, r in enumerate(reduced_ranges):
            #     reduced_ranges[i] = (r[0] + len(all_reduced_rows),r[1] + len(all_reduced_rows))

            # for i, fir in enumerate(first_iteration_ranges):
            #     first_iteration_ranges[i] = (fir[0] + len(all_reduced_rows),fir[1] + len(all_reduced_rows))

            all_reduced_ranges.extend(reduced_ranges)
            all_first_iteration_ranges.extend(first_iteration_ranges)
            for start, end in first_iteration_ranges:
                # print(f'start={start}, end={end}')
                try:
                    iend = gt.index.get_loc(end)
                except Exception as e:
                    import pdb; pdb.set_trace()
                    print(e)
                gt.iloc[iend:iend+2] = True # last program counter of first iteration and the following program counter
            for start, end in reduced_ranges:
                # print('reduced_ranges:' ,start,end)
                # gt[start:end][gt.isnull()] = False
                

                # not_null_labels = gt.loc[start:end][gt.notnull()]
                # if not_null_labels.shape[0] > 0:
                #     logging.debug(f'{not_null_labels.shape[0]} not null labels will be dropped')
                #     logging.debug(not_null_labels)
                gt.drop(gt.loc[start:end].index, inplace=True)
                # logging.info(f'len(reduced_rows)={len(reduced_rows):<3} len(all_reduced_rows)={len(all_reduced_rows):<3} start={start:<3} end={end:<3} iteration_size={iteration_size:<3} col.shape[0]={col.shape[0]} orig_size_not_null={orig_size_not_null} discrepancy={orig_size_not_null - col[col.notnull()].shape[0] - len(all_reduced_rows)}')
            
        col_size = col[col.notnull()].shape[0] 
        # gt.iloc[:col_size][gt.isnull()] = False
        gt.loc[ gt.iloc[:col_size][gt.isnull()].index ]  = False
        gt.iloc[col_size:] = np.NaN
        # print(f'col_size={col_size}')
        # print('gt[gt==True].shape[0] =',gt[gt==True].shape[0])
        # print('gt[gt.notnull()].shape[0] =',gt[gt.notnull()].shape[0])
        # print(f'len(all_first_iteration_ranges) = {len(all_first_iteration_ranges)}')
        gt = gt.reset_index(drop=True)
        col = col.reset_index(drop=True)
        col_instr = col_instr.reset_index(drop=True)
        # import pdb; pdb.set_trace()
        return col, col_instr, sorted(all_first_iteration_ranges), sorted(all_reduced_ranges), gt



    #########################################################
    #  Helper functions
            
    @staticmethod
    def get_repetition_ranges(col, size, reduced_rows=set()):
        repetition_ranges = []
        first_iteration_ranges = []
        for offset in range(size):
            for i in range(1, col.shape[0] // size):
                start = i * size + offset
                end = start + size
                chunk = col.iloc[start:end]
                prev_chunk = col.iloc[start-size:end-size]
                if np.array_equal(chunk.values, prev_chunk.values):
                    first = chunk.index[0]
                    last = chunk.index[-1]
                    rows_to_reduce = set(range(first,last+1))
                    if rows_to_reduce & reduced_rows:
                        continue
                    # if 908 in rows_to_reduce:
                    #     print('908 found')
                    #     import pdb; pdb.set_trace()
                    repetition_ranges.append((first, last))
                    reduced_rows |= rows_to_reduce
                    # fir = first repetition range
                    fir = (
                            prev_chunk.index[0],
                            prev_chunk.index[-1] 
                            )
                    if fir not in repetition_ranges:
                        first_iteration_ranges.append(fir)
        return repetition_ranges, reduced_rows, first_iteration_ranges

    @staticmethod
    def reduce_loops_single_size(col, col_instr, size, reduced_rows):
        reduced_ranges, reduced_rows, first_iteration_ranges = __class__.get_repetition_ranges(col, size, reduced_rows)
        for first, last in reduced_ranges:
            col.drop(col.loc[first:last].index, inplace=True)
            col_instr.drop(col_instr.loc[first:last].index, inplace=True)
            # col = col.reset_index(drop=True)
        return col, col_instr, reduced_ranges, reduced_rows, first_iteration_ranges

    @staticmethod
    def generate_ground_truth_column(col, offset, section_size):
        # copy original column to preserve NaN 
        gt = col.copy()
        gt[ gt.notnull() ] = False
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
        min_size = 10
        max_size = min(50, col.shape[0]-2)
        return random.randint(min_size, max_size)

    # @staticmethod
    # def fixed_reduce_loops(col):
    #     # order = [1,2,3]
    #     # def is_repetition(rol):
    #     #     np.array_equal(rol.values, order)
    #     # x = s.rolling(3).apply(is_repetition)
    #     if col.name == 'normal: stack-mission_riscv64_normal.pc':
    #         pass # TODO
    #     if col.name == 'normal: stack-mission_riscv64_normal_mimic_payload.pc':
    #         pass # TODO
    #     import pdb; pdb.set_trace()

    @staticmethod
    def overlaps_any_range(r, ranges_):
        for r2 in ranges_:
            if range(max(r[0], r2[0]), min(r[-1], r2[-1])+1):
                return True
        return False


    @staticmethod
    def generate(df_n, df_n_instr, instruction_types, offsets_count, anomaly_methods=[], reduce_loops=True, min_iteration_size=50):
        '''  
        min_iteration_size is used only for loop reducing anomalies
        return_anomalies_only=True can be used to avoid returning the whole program counter dataframes 
        (and only return the anomalous sizes)
        return_anomalies_only_padding can be used to extend "edges" of anomalies with normal program counters
        '''
        if not anomaly_methods:
            anomaly_methods = [
                __class__.randomize_section
                # Artificial_Anomalies.slightly_randomize_section,
                # Artificial_Anomalies.minimal
                ]
        anomalies_ranges = []
        pre_anomaly_values = []
        df_a = pd.DataFrame()
        df_a_instr = pd.DataFrame()
        df_a_ground_truth = pd.DataFrame(dtype=bool)
        # Introduce artificial anomalies for all the files, resulting in the following testing examples:
        # - method 1 with file 1
        # - method 1 with file 2
        # - method 2 with file 1
        # - method 2 with file 2
        # - method 3 with file 1
        # - method 3 with file 2
        # 
        # Where "method" is a one of the methods from "Artificial_Anomalies" class (e.g. randomize_section, slightly_randomize_section, minimal)
        # and where "file" is a normal/baseline file containing program counters.
        # Example above shows only 3 methods and 2 files, but the principle applies for any number.
        # So with 5 methods and 5 normal pc files there would be 25 testing examples.

        for i, method in enumerate(anomaly_methods):
            # for each normal/baseline append column with introduced anomalies into into "df_a"
            for j, column_name in enumerate(df_n):
                for k in range(offsets_count):
                    # introduce anomalies
                    col_a, col_a_instr, ar, pav, col_a_ground_truth = method(df_n[column_name].copy(), df_n_instr[column_name].copy(), instruction_types)
                    # keep record of anomalies and previous values (for plotting later)
                    anomalies_ranges.append(ar)
                    pre_anomaly_values.append(pav)
                    # rename column
                    new_column_name = column_name.replace('normal', f'{method.__name__}_({i},{j},{k})', 1)
                    df_a[new_column_name] = col_a
                    df_a_instr[new_column_name] = col_a_instr
                    df_a_ground_truth[new_column_name] = col_a_ground_truth

        # REDUCE LOOPS
        # Reducing loops can't be very randomized so it's done after all other 
        # anomalies are introduced (where program counter values are randomized).
        if reduce_loops:
            for j, column_name in enumerate(df_n):
                new_column_name = column_name.replace('normal', f'reduced_loops', 1)
                col, col_a_instr, first_iteration_ranges, reduced_ranges, col_a_ground_truth = __class__.reduce_loops(
                        df_n[column_name].copy(),
                        df_n_instr[column_name].copy(),
                        instruction_types,
                        min_iteration_size=min_iteration_size
                        )
                new_column = pd.Series([np.NaN]*df_n.shape[0])
                new_column_instr = new_column.copy()
                new_column[0:col.shape[0]] = col
                new_column_instr[0:col.shape[0]] = col_a_instr
                # logging.info(f'new_column: {new_column}')
                # pav = pd.Series()
                df_a[new_column_name] = new_column
                df_a_instr[new_column_name] = new_column_instr
                df_a_ground_truth[new_column_name] = col_a_ground_truth
                pav = []
                # TODO: append original values based on "col.probably_loc[reduced_range] for reduced_range in reduced_ranges"
                for r in sorted(reduced_ranges):
                    # pav = pav.combine(col.loc[r[0]:r[1]], max, fill_value=-1)
                    pav.append(df_n[column_name].loc[r[0]:r[1]].copy())
                # specific_values=[True] will return anomaly ranges only
                ar = utils.get_same_consecutive_values_ranges(col_a_ground_truth, specific_values=[True])
                pre_anomaly_values.append(pav)
                anomalies_ranges.append(ar)

        return df_a, df_a_instr, df_a_ground_truth, anomalies_ranges, pre_anomaly_values

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

