import os
import random

file_name = 'main.c'
executable_name = file_name.split('.')[0]
func_prefix = 'func_'
func_count = 100
shuffle_funcs = True

normal_sequences_count = 1
normal_sequence_size = 10

normal_sequences = generate_normal_sequences(
        normal_sequences_count,
        normal_sequence_size,
        funcs_count
        )

abnormal_sequences = generate_abnormal_sequences(
        normal_sequences,
        normal_sequence_size, 
        funcs_count
        )

def generate_normal_sequences(count, seq_size, funcs_count):
    sequences = []
    for i in range(count):
        seq = []
        for j in range(seq_size):
            seq += str( random.randint(0, func_count) ) + ' '
        sequences.append(seq.strip())
    return sequences

def generate_abnormal_sequences(normal_sequences, seq_size, funcs_count):
    ''' Generate different types of abnormal program sequences
        with increasing difficulty of detection. '''
    sequences = []

    seq = []
    for i in range(seq_size):
        seq.append(  )
    
    for i in range(5):
        seq = []
        for j in range(seq_size):
            seq += str( random.randint(0, func_count) ) + ' '
        sequences.append(seq)
    return sequences.strip()

def generate_functions(count, func_prefix, shuffle):
    funcs_str = ''
    funcs_names = []
    indices = range(count)
    if shuffle:
        random.shuffle(indices)


    for i in range(count):
        func_name = f'{func_prefix}{i}';
        funcs_names.append(func_name)
        funcs_str += (
                f'void {func_name}() {{ printf("{func_name} "); }}\n'
                )

    return funcs_str, funcs_names


functions, funcs_names = generate_functions(func_count, func_prefix=func_prefix, shuffle=shuffle_funcs)

content = f'''
#include <stdio.h>
#include <stdlib.h>

{functions}

void (*func_ptr[{len(funcs_names)}])() = {{ {', '.join(funcs_names)} }};

int main(int argc, char *argv[]) {{
    
    for (int i=1; i < argc; i++) {{
        int index = atoi(argv[i]);
        func_ptr[index]();
    }}

    return 0;
}}
'''

with open(file_name, 'w') as f:
    f.write(content)

print('Compling...')
print( os.popen(f'gcc {file_name} -o {executable_name}').read() )

print('Running normal sequences:')
for seq in normal_sequences:
    seq = ' '.join(seq)
    print( os.popen(f'./{executable_name} {seq}').read())

print()
print('Running abnormal sequences:')
for seq in abnormal_sequences:
    print( os.popen(f'./{executable_name} {seq}').read())




