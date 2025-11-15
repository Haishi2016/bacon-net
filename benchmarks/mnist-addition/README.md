pip install torch torchvision torch-explain 
python mnist_addition_dcr.py --epochs 120 --batch-size 128 --train-pairs 80000  --tau-start 2.5 --tau-end 0.8 --emb 30 --aux 1.0 --entropy 0.0005 --lr 1e-3
# Evaluate only with a pretrained model (skips training, prints accuracy and rules)
python mnist_addition_dcr.py --pretrained best_dcr_mnist_addition.pt --test-pairs 5000 --batch-size 128 --tau-end 0.8


python benchmarks\mnist-addition\dcr\mnist_addition_bacon.py --epochs 120 --batch-size 128 --train-pairs 80000 --tau-start 2.5 --tau-end 0.8 --lr 1e-3

python benchmarks\mnist-addition\dcr\mnist_addition_bacon_multiclass.py --auto-refine --refine-acc-gate 0.99 --refine-tau-gate 2.0 --epochs 120 --batch-size 128 --train-pairs 80000 --tau-start 2.5 --tau-end 0.8 --lr 1e-3


 python benchmarks\mnist-addition\dcr\mnist_addition_bacon_multiclass.py --auto-refine --refine-acc-gate 0.99 --refine-tau-gate 2.0 --epochs 120 --batch-size 128 --train-pairs 80000 --tau-start 2.5 --tau-end 0.8 --lr 1e-3

python benchmarks\mnist-addition\dcr\mnist_addition_bacon_multiclass.py --epochs 120 --batch-size 128 --train-pairs 80000 --tau-start 2.5 --tau-end 0.8 --lr 1e-3

python benchmarks\mnist-addition\dcr\mnist_addition_bacon_multiclass.py --pretrained benchmarks\mnist-addition\dcr\best_bacon_mnist_addition_multiclass-9910.pt --test-pairs 5000 --batch-size 128 --tau-end 0.8

=====

python .\mnist_addition_bacon_multiclass.py --train-pairs 80000 --test-pairs 5000 --batch-size 128 --epochs 120 --lr 1e-3 --entropy 0.0005 --tau-start 3.0 --tau-end 0.6 --seed 0 --perm-search

99.06%
 
======

python .\mnist_addition_dcr.py --train-pairs 80000 --test-pairs 5000 --batch-size 128 --epochs 120 --lr 1e-3 --entropy 0.0005 --tau-start 3.0 --tau-end 0.6 --seed 0

=======

| Run # | DeepProbLog | DCR | BACON|
|--------|--------|--------|--------|
| 1 | 99.08%  | 95.66% | 99.06% |
