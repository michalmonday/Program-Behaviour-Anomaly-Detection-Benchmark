bold=$(tput bold)
normal=$(tput sgr0)

if [ $# -ne 1 ]; then
LOG_FNAME=$(date +"%Y_%m_%d-%H:%M:%S").log
else
LOG_FNAME=test
fi
echo ""
echo "Log file will be stored in: ${bold}${LOG_FNAME}${normal}"
echo ""
sleep 2

/tools/RISC-V/emulator/cheri/output/sdk/bin/qemu-system-riscv64cheri -M virt -m 2048 -nographic -bios bbl-riscv64cheri-virt-fw_jump.bin -kernel /tools/RISC-V/emulator/cheri/output/rootfs-riscv64-purecap/boot/kernel/kernel -drive if=none,file=/tools/RISC-V/emulator/cheri/output/cheribsd-riscv64-purecap.img,id=drv,format=raw -device virtio-blk-device,drive=drv -device virtio-net-device,netdev=net0 -netdev 'user,id=net0,smb=/tools/RISC-V/emulator/cheri<<<source_root@ro:/tools/RISC-V/emulator/cheri/build<<<build_root:/tools/RISC-V/emulator/cheri/output<<<output_root@ro:/tools/RISC-V/emulator/cheri/output/rootfs-riscv64-purecap<<<rootfs,hostfwd=tcp::10019-:22' -device virtio-rng-pci -D ${LOG_FNAME}

