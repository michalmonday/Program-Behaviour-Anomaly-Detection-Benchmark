if (($# != 1)); then
    >&2 echo ""
    >&2 echo "Usage: $(basename $0) file.c"
    >&2 echo ""
    exit 1
fi
    
# out_fname=$(realpath $(echo "$1" | cut -f 1 -d '.'))
out_fname=$(echo "$1" | cut -f 1 -d '.')
llvm_objdump=/tools/RISC-V/emulator/cheri/output/sdk/bin/llvm-objdump
llvm_objdump_output_fname="${out_fname}_llvm_objdump.output"

# compile
echo "Compiling ${out_fname}"
ccc riscv64 $1 -o $out_fname

# copy to qemu togeter with source
copy_to_qemu.sh /research $out_fname $1

# produce llvm-objdump output
$llvm_objdump -sSD $out_fname > $llvm_objdump_output_fname

echo $llvm_objdump_output_fname
# extract function ranges (json) from llvm-objdump output
extract_function_ranges_from_llvm_objdump.py $llvm_objdump_output_fname -o "${out_fname}_function_ranges.json"
