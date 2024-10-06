# single 
python idbm_hilbert.py --batch_dim 128
# multi
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 --master_port 26600 idbm_hilbert_multi.py --batch_dim 128

nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 --master_port 26600 idbm_hilbert_mnist.py --batch_dim 128" >1.out 2>1.err & disown


nohup bash -c "CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node=4 --master_port 26600 dbfs_quadratic.py --load False" >11.out 2>11.err & disown


##################
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 --master_port 26600 dbfs_afhq.py --load=False" >7.out 2>7.err & disown


