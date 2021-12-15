#!/usr/bin/python3
import argparse
import re
import json
import os
from pprint import pprint

def bold(text): 
    return "\033[1m" + text + "\033[0m"

def extract_function_ranges_from_llvm_objdump(f_name):
    with open(f_name) as f:
        data = f.read()

    text_section = re.search(
            r'Disassembly of section \.text:(.+?)Disassembly of section', 
            data, 
            re.MULTILINE | re.DOTALL
            ).group(1)

    funcs = re.findall(
            r'\d+\s<([^>]+)>:(.+?)\n\n', 
            text_section, 
            re.MULTILINE | re.DOTALL
            )

    ranges = {}
    ranges['total'] = [999999999999999, -1] # min, max starting values
    for name, body in funcs:
        addresses = re.findall(r'\n\s+([0-9a-fA-F]+)', body, re.MULTILINE | re.DOTALL)
        if not addresses: continue
        start = int(addresses[0], 16)
        end = int(addresses[-1], 16)

        while name in ranges:
            name = name + '_'
        ranges[name] = [start, end]
        
        if start < ranges['total'][0]: 
            ranges['total'][0] = start
        if end > ranges['total'][1]: 
            ranges['total'][1] = end
    return ranges
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'Extracts memory locations of functions as '
        'outputted by "llvm-objdump -sSD binary_file_name".\n'
        'Output is stored in a json file, it can be used '
        'when plotting program counter traces obtained '
        'with: qtrace -u exec ./program'
        ))
    parser.add_argument(
            'file_name',
            type=argparse.FileType('r'),
            help='File containing output of llvm-objdump.'
            )
    parser.add_argument(
            '-o',
            type=str,
            required=False,
            metavar='',
            help='Output json file name.'
            )
    parser.add_argument(
            '--output-to-stdout',
            required=False,
            action='store_true',
            help='Outputs to stdout as well as the file.'
            )

    args = parser.parse_args()

    ranges = extract_function_ranges_from_llvm_objdump(args.file_name.name)

    if args.output_to_stdout:
        pprint(ranges)

    if args.o:
        out_fname = args.o if args.o.endswith('.json') else args.o + '.json'
    else:
        out_fname = os.path.basename(args.file_name.name).split('.')[0] + '_ranges.json'
    with open(out_fname, 'w') as f:
        json.dump(ranges, f, indent=4)

    print('Outputted to :', bold(out_fname))



