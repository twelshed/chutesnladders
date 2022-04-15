
i=1
exp_dir=$1
count=$(ls $exp_dir/*.json | wc -l)
for f in $exp_dir/*.json;
do
    echo $f $i of $count
    python run.py --paramset $f;

    i=$((i + 1))

done