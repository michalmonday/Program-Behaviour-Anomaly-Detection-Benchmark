#    printf("Cookie monster is hungry, provide some cookies!\n");
#    printf("'=' skips the next %zu bytes\n", sizeof(void *));
#    printf("'-' skips to the next character\n");
#    printf("XX as two hex digits stores a single cookie\n");
#    printf("> ");
import random

how_many = 10

def generate_input():
    # tokens = ['=', '-', 'number']
    tokens = ['=', '-', 'AA']
    input_gen = ''
    for i in range(random.randint(1,18)):
        # tok = random.choice(tokens)
        # if tok == 'number':
        #     num = random.randint(0,255)
        #     input_gen += f'{num:02X}'
        # else:
        #     input_gen += tok
        input_gen += random.choice(tokens)
    return input_gen

inputs = list( set(generate_input() for _ in range(how_many)) )
for i, inp in enumerate(inputs):
    print(f'{i+1}. {inp}')
     
