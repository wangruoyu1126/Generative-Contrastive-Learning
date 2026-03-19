echo "=========== CIFAR10"
CUDA_VISIBLE_DEVICES=1 python linear.py --dataset CIFAR10 --model_path results/CIFAR10/SimCLR/128_0.5_200_256_200_20221104-1010_model.pth --batch_size 256 --epochs 100
CUDA_VISIBLE_DEVICES=1 python linear.py --dataset CIFAR10 --model_path results/CIFAR10/SimCLR+ours/128_0.5_200_256_200_20221103-0900_model.pth --batch_size 256 --epochs 100
CUDA_VISIBLE_DEVICES=1 python linear.py --dataset CIFAR10 --model_path results/CIFAR10/DCL/128_0.5_200_256_200_20221105-1304_model.pth --batch_size 256 --epochs 100
CUDA_VISIBLE_DEVICES=1 python linear.py --dataset CIFAR10 --model_path results/CIFAR10/DCL+ours/128_0.5_200_256_200_20221104-1431_model.pth --batch_size 256 --epochs 100
CUDA_VISIBLE_DEVICES=1 python linear.py --dataset CIFAR10 --model_path results/CIFAR10/HCL/128_0.5_200_256_200_20221025-1027_model.pth --batch_size 256 --epochs 100
CUDA_VISIBLE_DEVICES=1 python linear.py --dataset CIFAR10 --model_path results/CIFAR10/HCL+ours/128_0.5_200_256_200_20221025-1023_model.pth --batch_size 256 --epochs 100












