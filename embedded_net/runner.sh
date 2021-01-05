#!/usr/bin/env sh
set -e

riscv-nuclei-elf-objcopy -O binary $1 firmware.bin

NAME="$(basename firmware.bin)"
SIZE_TEXT="$(riscv-nuclei-elf-size "$1" | tail -1 | cut -f1)"
echo $SIZE_TEXT
SIZE_DATA="$(riscv-nuclei-elf-size "$1" | tail -1 | cut -f2)"
SIZE_BSS="$(riscv-nuclei-elf-size "$1" | tail -1 | cut -f3)"

printf "\n"
printf "Program:             %s\n" "$NAME"
printf "Size:\n"
printf "   .text   %s (exact: %d)\n" "$(numfmt --to=si --padding=9 "$SIZE_TEXT")" "$SIZE_TEXT"
printf "   .data   %s (exact: %d)\n" "$(numfmt --to=si --padding=9 "$SIZE_DATA")" "$SIZE_DATA"
printf "   .bss    %s (exact: %d)\n" "$(numfmt --to=si --padding=9 "$SIZE_BSS")" "$SIZE_BSS"
printf "\n"
printf "Please bring up the bootloader and press ENTER!\n"
read -r REPLY
printf "Attempting to flash ...\n"
printf "\n"

dfu-util -a 0 -s 0x08000000:leave -D firmware.bin