echo "=========== CIFAR100"
CUDA_VISIBLE_DEVICES=1 python linear.py --dataset CIFAR100 --model_path results/CIFAR100/SimCLR/128_0.5_200_256_200_20221103-0903_model.pth --batch_size 256 --epochs 100
CUDA_VISIBLE_DEVICES=1 python linear.py --dataset CIFAR100 --model_path results/CIFAR100/SimCLR+ours/128_0.5_200_256_200_20221101-1938_model.pth --batch_size 256 --epochs 100
CUDA_VISIBLE_DEVICES=1 python linear.py --dataset CIFAR100 --model_path results/CIFAR100/DCL/128_0.5_200_256_200_20221023-1048_model.pth --batch_size 256 --epochs 100
CUDA_VISIBLE_DEVICES=1 python linear.py --dataset CIFAR100 --model_path results/CIFAR100/DCL+ours/128_0.5_200_256_200_20221023-1528_model.pth --batch_size 256 --epochs 100
CUDA_VISIBLE_DEVICES=1 python linear.py --dataset CIFAR100 --model_path results/CIFAR100/HCL/128_0.5_200_256_200_20221026-1119_model.pth --batch_size 256 --epochs 100
CUDA_VISIBLE_DEVICES=1 python linear.py --dataset CIFAR100 --model_path results/CIFAR100/HCL+ours/128_0.5_200_256_200_20221026-1535_model.pth --batch_size 256 --epochs 100












