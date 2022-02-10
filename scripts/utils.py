import pandas as pd
import numpy as np

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

    column_names = [column_prefix + short_name(f_name) for f in f_list]
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
    
def plot_pc_timeline(df, function_ranges={}, function_line_width=0.7, ax=None, title='Timeline of program counters'):
    if ax:
        ax2 = df.plot(linewidth=0.7, ax=ax, title=title)
    else:
        ax2 = df.plot(linewidth=0.7, title=title)
    ax2.set_xlabel('Instruction index')
    ax2.set_ylabel('Program counter (address)')
    y_start, y_end = ax2.get_yticks()[0], ax2.get_yticks()[-1]
    ax2.get_yaxis().set_major_formatter(lambda x,pos: f'0x{int(x):X}')
    i = 0
    for func_name, (start, end) in function_ranges.items():
        if func_name == 'total': continue
        # print(start, end, y_start, y_end)
        if start < y_start or end > y_end: continue
        # print(start)
        ax2.axhline(y=start, color='lightgray', linewidth=function_line_width, linestyle='dashed')
        # ax2.text(df.shape[0]*(0.9-i/80), start, func_name, rotation=0, va='bottom', fontsize='x-small')
        # ax2.text(df.shape[0]*0.8, start, func_name, rotation=0, va='bottom', fontsize='x-small')
        i += 1
    ax2_t = ax2.twinx()
    ax2_t.set_yticks([v[0] for v in function_ranges.values()])
    ax2_t.set_yticklabels(list(function_ranges.keys()), fontdict={'fontsize':7})
    ax2_t.set_ylim(*ax2.get_ylim())
    df.plot(ax=ax2, marker='h', markersize=1, linestyle='none', legend=None)
    return ax2

def plot_vspans(ax, values, size, color='red', alpha=0.15):
    for i in values:
        ax.axvspan(i, i+size, color=color, alpha=alpha)

