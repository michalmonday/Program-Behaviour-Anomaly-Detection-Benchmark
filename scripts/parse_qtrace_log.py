#!/usr/bin/python3
import re
import argparse
import sys
import json

from extract_function_ranges_from_llvm_objdump import extract_function_ranges_from_llvm_objdump

def bold(text): 
    return "\033[1m" + text + "\033[0m"

def get_pc(fname, main_addr, max_pc=None):
    ''' fname = log file filename
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
            '-o',
            metavar='',
            type=str,
            required=False,
            help='Output file name. Default is the same as input (but with ".pc" extension.'

#    parser.add_argument(
#            '-max',
#            '--max-pc',
#            type=int,
#            required=False,
#            default=None,
#            help='Program counters above this value will not be included in output.'
#            )

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

    function_ranges = json.load(args.function_ranges)
    main_addr = function_ranges['main'][0]
    max_addr = function_ranges['total'][1]

    program_counters = get_pc(args.logfile.name, main_addr, max_pc = max_addr)
    
    # for pc in program_counters:
    #     print(pc)

    output_fname = args.o if args.o else args.logfile.name.split('.')[0] + '.pc'
    with open(output_fname, 'w') as f:
        f.write('\n'.join([f'{pc:X}' for pc in program_counters]))

    print(sys.argv[0], 'outputted program counters to', bold(output_fname))

    
    

