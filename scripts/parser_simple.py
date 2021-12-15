#!/usr/bin/python3
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import sys

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
    # return pd.DataFrame(pc, columns=[fname])
    return pc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'logfile',
            type=argparse.FileType('r'),
            help='Log file as outputted by qtrace -u exec ./program < normal_input.txt'
            )

    parser.add_argument(
            '-max',
            '--max-pc',
            type=int,
            required=False,
            default=None,
            help='Program counters above this value will not be included in output.'
            )

    parser_group = parser.add_mutually_exclusive_group(required=True)
    parser_group.add_argument(
            '-fr',
            '--function-ranges',
            type=argparse.FileType('r'),
            help='File name containing output of extract_function_ranges_from_llvm_objdump.py'
            )

    #parser_group.add_argument(
    #        '-L',
    #        '--llvm-objdump',
    #        type=argparse.FileType('r'),
    #        required=False,
    #        help='File name containing output of llvm-objdump.'
    #        )


    #parser_group.add_argument(
    #        '-m',
    #        '--main-addr',
    #        type=int,
    #        required=False,
    #        default=None,
    #        help='Entry address of main function.'
    #        )

    args = parser.parse_args()

    #if args.main_addr:
    #    main_addr = args.main_addr
    #elif args.llvm_objdump:
    #    ranges = extract_function_ranges_from_llvm_objdump(args.llvm_objdump.name)
    #    main_addr = ranges['main'][0]
    #elif args.function_ranges:
    #    main_addr = json.load(args.function_ranges)['main'][0]

    program_counters = get_pc(args.logfile.name, main_addr, max_pc = args.max_pc)
    
    print(program_counters)
    # for pc in program_counters:
    #     print(pc)
    
    

