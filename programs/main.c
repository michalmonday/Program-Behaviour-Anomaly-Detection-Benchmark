
#include <stdio.h>
#include <stdlib.h>

void func_0() { printf("func_0 "); }
void func_1() { printf("func_1 "); }
void func_2() { printf("func_2 "); }
void func_3() { printf("func_3 "); }
void func_4() { printf("func_4 "); }
void func_5() { printf("func_5 "); }
void func_6() { printf("func_6 "); }
void func_7() { printf("func_7 "); }
void func_8() { printf("func_8 "); }
void func_9() { printf("func_9 "); }


void (*func_ptr[10])() = { func_0, func_1, func_2, func_3, func_4, func_5, func_6, func_7, func_8, func_9 };

int main(int argc, char *argv[]) {
    
    for (int i=1; i < argc; i++) {
        int index = atoi(argv[i]);
        func_ptr[index]();
    }

    return 0;
}
