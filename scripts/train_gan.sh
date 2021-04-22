python roofGAN.py --train \
                    --data_root /local-scratch/yimingq/house/normalmap \
                    --critic_iters 1 \
                    --n_iters 200000 \
                    --batch_size 16 \
                    --proj_dir roofgan \
                    -g 0
