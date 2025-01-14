python main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 5 \
  --n-epochs 500 \
  --model graphsage \
  --n-layers 3 \
  --n-hidden 128 \
  --log-every 10 \
  --enable-pipeline \
  --use-pp --corr_momentum 1


