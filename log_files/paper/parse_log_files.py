import os
for i in range(10):
    print( os.popen(f'parse_qtrace_log.py log/normal_{i+1}.log -o csv/normal_{i+1}.csv -fr stack-mission_riscv64_llvm_objdump_ranges.json').read() )



