import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse

from extract_function_ranges_from_llvm_objdump import extract_function_ranges_from_llvm_objdump


def get_pc(fname, main_addr, max_pc=None):
    ''' df = pd.DataFrame
        main_addr = 0x11fb2 (for example)
        max_pc = 0xffffff (for example, may be used to exclude library code) 
        '''
    pc = []
    main_line_start = f'[0:0] 0x{main_addr:016x}'
    found_main = False
    with open(fname) as f:
        for i, line in enumerate(f.readlines()):
            if not found_main and line.startswith(main_line_start): found_main = True
            if not found_main: continue
            val =  re.search(r'\[0:0\] (0x[^:]+)', line)
            if not val: continue
            val = int( val.group(1) , 16)
            if max_pc and max_pc < val: continue
            pc.append(val)
    return pd.DataFrame(pc, columns=[fname])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'logfile',
            type=argparse.FileType('r'),
            help='Log file as outputted by qtrace -u exec ./program < normal_input.txt'
            )

    parser.add_argument(
            '-max'
            '--max-pc',
            type=int,
            required=False,
            default=None,
            help='Program counters above this value will not be included in output.'
            )

    parser_group = parser.add_mutually_exclusive_group(required=True)
    parser_group.add_argument(
            '-L',
            '--llvm-objdump',
            type=argparse.FileType('r'),
            required=False,
            help='File name containing output of llvm-objdump.'
            )

    parser_group.add_argument(
            '-m'
            '--main-addr',
            type=int,
            required=False,
            default=None,
            help='Entry address of main function.'
            )

    args = parser.parse_args()
    df = get_pc(args.normal_log.name, main_addr, max_pc = args.max_pc)

    extract_function_ranges_from_llvm_objdump()
