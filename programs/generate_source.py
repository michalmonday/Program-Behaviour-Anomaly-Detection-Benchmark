import os

file_name = 'main.c'
executable_name = file_name.split('.')[0]
func_prefix = 'func_'

normal_sequences = [
    "0 1 2"
    ]

abnormal_sequences = [
    "2 1 1",
    "1 1 0"
    ]


def generate_functions(count, func_prefix):
    funcs_str = ''
    funcs_names = []
    for i in range(count):
        func_name = f'{func_prefix}{i}';
        funcs_names.append(func_name)
        funcs_str += (
                f'void {func_name}() {{ printf("{func_name} "); }}\n'
                )

    return funcs_str, funcs_names


functions, funcs_names = generate_functions(10, func_prefix=func_prefix)

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




