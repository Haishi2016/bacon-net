pip install torch torchvision torch-explain 
python mnist_addition_dcr.py --epochs 120 --batch-size 128 --train-pairs 80000  --tau-start 2.5 --tau-end 0.8 --emb 30 --aux 1.0 --entropy 0.0005 --lr 1e-3
