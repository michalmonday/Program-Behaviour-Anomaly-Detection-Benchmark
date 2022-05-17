List below shows what input was used to uninitialized stack variable cheri-exercise program to obtain which log:  

```
normal_1.log  ==AA==AA==-=-AA====-
normal_2.log  =
normal_3.log  AA=-==-AAAA-=AA
normal_4.log  =-=----AA=AA==AAAAAAAA
normal_5.log  --=AA==-AA-==AA-=
normal_6.log  AA-=AA=--
normal_7.log  AA=
normal_8.log  AAAAAA=
normal_9.log  -=-=
normal_10.log AA--
```

The meaning of each token (=, -, AA) is described in the source code at:  
https://ctsrd-cheri.github.io/cheri-exercises/missions/uninitialized-stack-frame-control-flow/index.html  

Example command ran to obtain normal_2.log:   
> qtrace -u exec ./stack-mission_riscv64 < file_containing_normal_2_input_line.txt

Then "ctrl+a, ctrl+c" would be used in qemu and the following command would be input:  
> logfile normal_3.log

This would change the log file to a new one (so Qemu would output "qtrace" to new file when it's used next time).  
