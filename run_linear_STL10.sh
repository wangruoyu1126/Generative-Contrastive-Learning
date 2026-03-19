echo "=========== STL10"
CUDA_VISIBLE_DEVICES=2 python linear.py --dataset STL10 --model_path results/STL10/SimCLR/128_0.5_200_256_200_20221031-1609_model.pth --batch_size 256 --epochs 100
CUDA_VISIBLE_DEVICES=2 python linear.py --dataset STL10 --model_path results/STL10/SimCLR+ours/128_0.5_200_256_200_20221030-1012_model.pth --batch_size 256 --epochs 100
CUDA_VISIBLE_DEVICES=2 python linear.py --dataset STL10 --model_path results/STL10/DCL/128_0.5_200_256_200_20221029-1308_model.pth --batch_size 256 --epochs 100
CUDA_VISIBLE_DEVICES=2 python linear.py --dataset STL10 --model_path results/STL10/DCL+ours/128_0.5_200_256_200_20221022-2202_model.pth --batch_size 256 --epochs 100
CUDA_VISIBLE_DEVICES=2 python linear.py --dataset STL10 --model_path results/STL10/HCL/128_0.5_200_256_200_20221027-1134_model.pth --batch_size 256 --epochs 100
CUDA_VISIBLE_DEVICES=2 python linear.py --dataset STL10 --model_path results/STL10/HCL+ours/128_0.5_200_256_200_20221027-1910_model.pth --batch_size 256 --epochs 100












