
if (($# < 2)); then
    >&2 echo ""
    >&2 echo "Usage: ./$(basename $0) destination_at_qemu filenames_to_copy_separated_by_space"
    >&2 echo ""
    exit 1
fi

dst=$1
echo "Destination at qemu is: \"${dst}\""
for v in "${@:2}"
do
    # echo "Copying ${v}..."
    scp -P 10019 $v root@localhost:${dst}
done

echo "Done"

