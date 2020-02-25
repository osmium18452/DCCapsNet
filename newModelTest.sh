if [$#==1]
then
  model=$1
else
  model=4
fi

python ./train.py -e 50 -b 50 -l 0.0001 -g 0 -r 0.05 -a 1 -p 7 -m $model --data 3