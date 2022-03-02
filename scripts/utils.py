import pandas as pd
import numpy as np
import logging
import random

def read_pc_values(f_name, relative_pc=False, ignore_non_jumps=False, load_address=0):
    with open(f_name) as f:
        pcs = [int(line.strip(), 16) + load_address for line in f.readlines() if line]

    # IGNORE_NON_JUMP AND RELATIVE_PC OPTIONS COMBINED TOGETHER CREATE 
    # THE QUESTION: SHOULD RELATIVE PC BE RELATIVE TO ANY LAST PC OR THE LAST 
    # NON-JUMP PC
    if ignore_non_jumps:
        # [0] is inserted at the begining of relative pcs so the length matches
        rel_pcs = [0] + relative_from_absolute_pc(pcs)
        if relative_pc:
            return [rel_pc for rel_pc in rel_pcs if abs(int(rel_pc)) > 4]
        else:
            return [pc for pc,rel_pc in zip(pcs, rel_pcs) if abs(int(rel_pc)) > 4]
        # rel_pcs = [0] + relative_from_absolute_pc(pcs)
        # return [pc for pc,rel_pc in zip(pcs, rel_pcs) if abs(int(rel_pc)) > 4]
    if relative_pc:
        pcs = relative_from_absolute_pc(pcs)
    return pcs

def df_from_pc_files(f_list, column_prefix='', relative_pc=False, ignore_non_jumps=False, load_address=0):
    if (len(f_list) == 1 and type(f_list != str)) or type(f_list[0]) != str:
        f_list = [item.name for item in f_list]
    all_pc = []
    for f_name in f_list:
        pc_chunk = read_pc_values(f_name, relative_pc=relative_pc, ignore_non_jumps=ignore_non_jumps, load_address=load_address) 
        all_pc.append(pc_chunk)

    def short_name(f_name):
        return f_name.split('/')[-1] 

    column_names = [column_prefix + short_name(f_name) for f_name in f_list]
    df = pd.DataFrame(all_pc, dtype=np.int64, index=column_names).T
    return df

def relative_from_absolute_pc(pc_collection):
    ''' Turn program counters into relative ones (not from begining, 
        but from the last counter value). For example:
        1,2,3 would turn into 0,1,1
        1,3,2 would turn into 0,2,-1 '''
    if type(pc_collection) == list:
        return [pc_collection[i+1] - pc for i,pc in enumerate(pc_collection[:-1])]
    raise Exception('relative_from_absolute_pc only accepts a list as an input.')
    # if dataframe
    # import pdb; pdb.set_trace()
    # return pc_collection.diff()


# separator_value is a value that will separate program counters of
# different programs after they're stacked together 
separator_value = 123456789
    
def series_to_sliding_windows(series, window_size):
    ''' Series may include program counter values from multiple files,
        in such case these must be separated by the "separator_value" '''
    windows = pd.DataFrame( [ w.to_list() for w in series.rolling(window=window_size) if separator_value not in w.to_list() and len(w.to_list()) == window_size] )#.reshape(-1, window_size)
    return windows

def merge_pc_df_columns(df):
    ''' Creates df with "all_pc" column that contains program counters from 
        multiple files (columns), separated by the "separator_value" '''
    # add separator_value row to each column (to avoid recognizing the last PC of 1 run as first PC of 2nd run)
    df = df.append(pd.Series(), ignore_index=True)
    df.iloc[-1] = separator_value
    # stack all columns on top of each other
    df = df.melt(value_name='all_pc').drop('variable', axis=1)
    df = df.dropna()
    return df

def pc_df_to_sliding_windows(df, window_size, unique=False):
    ''' df contains a column per each ".pc" file where each row contains
        program counter values '''
    if type(df) == pd.core.series.Series:
        windows = series_to_sliding_windows(df, window_size)
    else:
        df = merge_pc_df_columns(df)
        windows = series_to_sliding_windows(df['all_pc'], window_size)
    if unique:
        windows = windows.drop_duplicates()
    return windows

# def multiple_files_df_program_counters_to_unique_sliding_windows(df, window_size):
#     return multiple_files_df_program_counters_to_sliding_windows(df, window_size).drop_duplicates()

def plot_pc_histogram(df, function_ranges={}, bins=100, function_line_width=0.7, title='Histogram of program counters (frequency distribution)'):
    ax = df.plot.hist(bins=bins, alpha=1/df.shape[1], title=title)
    ax.get_xaxis().set_major_formatter(lambda x,pos: f'0x{int(x):X}')
    ax.get_yaxis().set_major_formatter(lambda x,pos: f'{int(x)}')
    x_start, x_end = ax.get_xticks()[0], ax.get_xticks()[-1]
    y_top = ax.transAxes.to_values()[3]
    i = 0
    for func_name, (start, end) in function_ranges.items():
        if func_name == 'total': continue
        if start < x_start or end > x_end: continue
        ax.axvline(x=start, color='lightgray', linewidth=function_line_width, linestyle='dashed')
        # ax.axvline(x=end, color='black')
        # ax.text(start, y_top*(0.8-i/100), func_name, rotation=90, va='bottom', fontsize='x-small')
        # ax.text(start, y_top*0.7, func_name, rotation=90, va='bottom', fontsize='x-small')
        i += 1
    # ax_t.xaxis.tick_top()
    # ax.legend(bbox_to_anchor=(0.1, 0.7))
    ax_t = ax.twiny()
    ax_t.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)
    ax_t.set_xticks([v[0] for v in function_ranges.values()])
    ax_t.set_xticklabels(list(function_ranges.keys()), fontdict={'fontsize':7}, rotation=90)
    ax_t.set_xlim(*ax.get_xlim())

    ax.set_xlabel('Program counter (address)')
    ax.set_ylabel('Frequency')
    return ax
    
def plot_pc_timeline(df, function_ranges={}, function_line_width=0.7, ax=None, title='Timeline of program counters', xlabel='Instruction index' , ylabel='Program counter (address)', **plot_kwargs_):
    plot_kwargs = {
            'linewidth':0.7,
            'title': title,
            'markersize': 2,
            'marker': 'h',
            **plot_kwargs_
            }
    if ax:
        plot_kwargs['ax'] = ax
    ax = df.plot(**plot_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    y_start, y_end = ax.get_yticks()[0], ax.get_yticks()[-1]
    ax.get_yaxis().set_major_formatter(lambda x,pos: f'0x{int(x):X}')
    i = 0
    for func_name, (start, end) in function_ranges.items():
        if func_name == 'total': continue
        # print(start, end, y_start, y_end)
        if start < y_start or end > y_end: continue
        # print(start)
        ax.axhline(y=start, color='lightgray', linewidth=function_line_width, linestyle='dashed')
        # ax.text(df.shape[0]*(0.9-i/80), start, func_name, rotation=0, va='bottom', fontsize='x-small')
        # ax.text(df.shape[0]*0.8, start, func_name, rotation=0, va='bottom', fontsize='x-small')
        i += 1
    ax_t = ax.twinx()
    ax_t.set_yticks([v[0] for v in function_ranges.values()])
    ax_t.set_yticklabels(list(function_ranges.keys()), fontdict={'fontsize':7})
    ax_t.set_ylim(*ax.get_ylim())
    # df.plot(ax=ax, marker='h', markersize=2, linestyle='none', legend=None)
    return ax

def plot_vspans(ax, values, size, color='red', alpha=0.15):
    for i in values:
        ax.axvspan(i, i+size, color=color, alpha=alpha)

def plot_vspans_ranges(ax, ranges, color='red', alpha=0.15):
    for start, end in ranges:
        ax.axvspan(start, end, color=color, alpha=alpha)

def introduce_artificial_anomalies(df):
    ''' Currently (10/02/22) only single anomalous file is supported.
        It returns modified dataframe and ranges of indices where
        anomalies were introduced (so it can be marked on the plot 
        later with "plot_vspans" for example.  '''
    col = df.iloc[:,0]

    anomalies_ranges = []
    original_values = []

    # EASY TO DETECT
    #     set values at index 10,11,12,13,14 to random values
    how_many = 5
    original_values.append(col[10-1:10+how_many+1].copy())
    col[10:10+how_many] = np.random.randint(col.min(), col.max(), how_many)
    anomalies_ranges.append((10-1,10+how_many))

    # HARDER TO DETECT
    #     slightly modify values at index 60,61,62,63,64 
    #     (by adding or subtracting multiply of 8)
    original_values.append(col[60-1:60+how_many+1].copy())
    col[60:60+how_many] += np.random.randint(-3, 3, how_many) * 8 
    anomalies_ranges.append((60-1,60+how_many))

    # HARD TO DETECT
    #     modify a single value at index 110 by adding 8 to it
    original_values.append(col[110-1:111+1].copy())
    col[110] += 8
    anomalies_ranges.append((110-1, 110+1))
    return df, anomalies_ranges, original_values

default_offset = 100
class Artificial_Anomalies:
    

    # all_methods = [
    #         __class__.randomize_section,
    #         __class__.slightly_randomize_section,
    #         __class__.minimal
    #         ]

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


def print_config(c):
    print_header('CONFIG')
    for s in c.sections():
        logging.info(f'[{s}]')
        for k,v in c.items(s):
            logging.info(f'    {k} = {v}')
        logging.info('')
    logging.info('')

def print_header(h):
    size = max(50, len(h)+4)
    logging.info('\n\n' + "#" * size)
    logging.info("#{0:^{size}}\n".format(h, size=size))
    # logging.info("#" * 20)

def windowize_ground_truth_labels(ground_truth_labels, window_size):
    ''' When using sliding_window '''
    return ground_truth_labels.rolling(window_size).apply(any)



