echo "=========== CIFAR10"
echo "====== 1-shot"
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/SimCLR/128_0.5_200_256_200_20221104-1010_model.pth --n-way 5 --n-support 1 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/SimCLR+ours/128_0.5_200_256_200_20221103-0900_model.pth --n-way 5 --n-support 1 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/DCL/128_0.5_200_256_200_20221105-1304_model.pth --n-way 5 --n-support 1 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/DCL+ours/128_0.5_200_256_200_20221104-1431_model.pth --n-way 5 --n-support 1 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/HCL/128_0.5_200_256_200_20221025-1027_model.pth --n-way 5 --n-support 1 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/HCL+ours/128_0.5_200_256_200_20221025-1023_model.pth --n-way 5 --n-support 1 --iter-num 100 --n-query 10

echo "====== 5-shot"
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/SimCLR/128_0.5_200_256_200_20221104-1010_model.pth --n-way 5 --n-support 5 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/SimCLR+ours/128_0.5_200_256_200_20221103-0900_model.pth --n-way 5 --n-support 5 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/DCL/128_0.5_200_256_200_20221105-1304_model.pth --n-way 5 --n-support 5 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/DCL+ours/128_0.5_200_256_200_20221104-1431_model.pth --n-way 5 --n-support 5 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/HCL/128_0.5_200_256_200_20221025-1027_model.pth --n-way 5 --n-support 5 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/HCL+ours/128_0.5_200_256_200_20221025-1023_model.pth --n-way 5 --n-support 5 --iter-num 100 --n-query 10

echo "====== 10-shot"
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/SimCLR/128_0.5_200_256_200_20221104-1010_model.pth --n-way 5 --n-support 10 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/SimCLR+ours/128_0.5_200_256_200_20221103-0900_model.pth --n-way 5 --n-support 10 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/DCL/128_0.5_200_256_200_20221105-1304_model.pth --n-way 5 --n-support 10 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/DCL+ours/128_0.5_200_256_200_20221104-1431_model.pth --n-way 5 --n-support 10 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/HCL/128_0.5_200_256_200_20221025-1027_model.pth --n-way 5 --n-support 10 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/HCL+ours/128_0.5_200_256_200_20221025-1023_model.pth --n-way 5 --n-support 10 --iter-num 100 --n-query 10

echo "====== 20-shot"
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/SimCLR/128_0.5_200_256_200_20221104-1010_model.pth --n-way 5 --n-support 20 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/SimCLR+ours/128_0.5_200_256_200_20221103-0900_model.pth --n-way 5 --n-support 20 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/DCL/128_0.5_200_256_200_20221105-1304_model.pth --n-way 5 --n-support 20 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/DCL+ours/128_0.5_200_256_200_20221104-1431_model.pth --n-way 5 --n-support 20 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/HCL/128_0.5_200_256_200_20221025-1027_model.pth --n-way 5 --n-support 20 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR10 --model results/CIFAR10/HCL+ours/128_0.5_200_256_200_20221025-1023_model.pth --n-way 5 --n-support 20 --iter-num 100 --n-query 10











echo "=========== CIFAR100"
echo "====== 1-shot"
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/SimCLR/128_0.5_200_256_200_20221103-0903_model.pth --n-way 5 --n-support 1 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/SimCLR+ours/128_0.5_200_256_200_20221101-1938_model.pth --n-way 5 --n-support 1 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/DCL/128_0.5_200_256_200_20221023-1048_model.pth --n-way 5 --n-support 1 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/DCL+ours/128_0.5_200_256_200_20221023-1528_model.pth --n-way 5 --n-support 1 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/HCL/128_0.5_200_256_200_20221026-1119_model.pth --n-way 5 --n-support 1 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/HCL+ours/128_0.5_200_256_200_20221026-1535_model.pth --n-way 5 --n-support 1 --iter-num 100 --n-query 10

echo "====== 5-shot"
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/SimCLR/128_0.5_200_256_200_20221103-0903_model.pth --n-way 5 --n-support 5 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/SimCLR+ours/128_0.5_200_256_200_20221101-1938_model.pth --n-way 5 --n-support 5 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/DCL/128_0.5_200_256_200_20221023-1048_model.pth --n-way 5 --n-support 5 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/DCL+ours/128_0.5_200_256_200_20221023-1528_model.pth --n-way 5 --n-support 5 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/HCL/128_0.5_200_256_200_20221026-1119_model.pth --n-way 5 --n-support 5 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/HCL+ours/128_0.5_200_256_200_20221026-1535_model.pth --n-way 5 --n-support 5 --iter-num 100 --n-query 10


echo "====== 10-shot"
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/SimCLR/128_0.5_200_256_200_20221103-0903_model.pth --n-way 5 --n-support 10 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/SimCLR+ours/128_0.5_200_256_200_20221101-1938_model.pth --n-way 5 --n-support 10 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/DCL/128_0.5_200_256_200_20221023-1048_model.pth --n-way 5 --n-support 10 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/DCL+ours/128_0.5_200_256_200_20221023-1528_model.pth --n-way 5 --n-support 10 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/HCL/128_0.5_200_256_200_20221026-1119_model.pth --n-way 5 --n-support 10 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/HCL+ours/128_0.5_200_256_200_20221026-1535_model.pth --n-way 5 --n-support 10 --iter-num 100 --n-query 10


echo "====== 20-shot"
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/SimCLR/128_0.5_200_256_200_20221103-0903_model.pth --n-way 5 --n-support 20 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/SimCLR+ours/128_0.5_200_256_200_20221101-1938_model.pth --n-way 5 --n-support 20 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/DCL/128_0.5_200_256_200_20221023-1048_model.pth --n-way 5 --n-support 20 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/DCL+ours/128_0.5_200_256_200_20221023-1528_model.pth --n-way 5 --n-support 20 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/HCL/128_0.5_200_256_200_20221026-1119_model.pth --n-way 5 --n-support 20 --iter-num 100 --n-query 10
CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset CIFAR100 --model results/CIFAR100/HCL+ours/128_0.5_200_256_200_20221026-1535_model.pth --n-way 5 --n-support 20 --iter-num 100 --n-query 10

















# echo "=========== STL10"
# echo "====== 1-shot"
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/SimCLR/128_0.5_200_256_200_20221031-1609_model.pth --n-way 5 --n-support 1 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/SimCLR+ours/128_0.5_200_256_200_20221030-1012_model.pth --n-way 5 --n-support 1 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/DCL/128_0.5_200_256_200_20221029-1308_model.pth --n-way 5 --n-support 1 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/DCL+ours/128_0.5_200_256_200_20221022-2202_model.pth --n-way 5 --n-support 1 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/HCL/128_0.5_200_256_200_20221027-1134_model.pth --n-way 5 --n-support 1 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/HCL+ours/128_0.5_200_256_200_20221027-1910_model.pth --n-way 5 --n-support 1 --iter-num 100

# echo "====== 5-shot"
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/SimCLR/128_0.5_200_256_200_20221031-1609_model.pth --n-way 5 --n-support 5 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/SimCLR+ours/128_0.5_200_256_200_20221030-1012_model.pth --n-way 5 --n-support 5 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/DCL/128_0.5_200_256_200_20221029-1308_model.pth --n-way 5 --n-support 5 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/DCL+ours/128_0.5_200_256_200_20221022-2202_model.pth --n-way 5 --n-support 5 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/HCL/128_0.5_200_256_200_20221027-1134_model.pth --n-way 5 --n-support 5 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/HCL+ours/128_0.5_200_256_200_20221027-1910_model.pth --n-way 5 --n-support 5 --iter-num 100

# echo "====== 10-shot"
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/SimCLR/128_0.5_200_256_200_20221031-1609_model.pth --n-way 5 --n-support 10 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/SimCLR+ours/128_0.5_200_256_200_20221030-1012_model.pth --n-way 5 --n-support 10 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/DCL/128_0.5_200_256_200_20221029-1308_model.pth --n-way 5 --n-support 10 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/DCL+ours/128_0.5_200_256_200_20221022-2202_model.pth --n-way 5 --n-support 10 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/HCL/128_0.5_200_256_200_20221027-1134_model.pth --n-way 5 --n-support 10 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/HCL+ours/128_0.5_200_256_200_20221027-1910_model.pth --n-way 5 --n-support 10 --iter-num 100

# echo "====== 20-shot"
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/SimCLR/128_0.5_200_256_200_20221031-1609_model.pth --n-way 5 --n-support 20 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/SimCLR+ours/128_0.5_200_256_200_20221030-1012_model.pth --n-way 5 --n-support 20 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/DCL/128_0.5_200_256_200_20221029-1308_model.pth --n-way 5 --n-support 20 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/DCL+ours/128_0.5_200_256_200_20221022-2202_model.pth --n-way 5 --n-support 20 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/HCL/128_0.5_200_256_200_20221027-1134_model.pth --n-way 5 --n-support 20 --iter-num 100
# CUDA_VISIBLE_DEVICES=6 python few_shot.py --dataset STL10 --model results/STL10/HCL+ours/128_0.5_200_256_200_20221027-1910_model.pth --n-way 5 --n-support 20 --iter-num 100



















