# BrepGen: A B-rep Generative Diffusion Model with Structured Latent Geometry (SIGGRAPH 2024)
## 原文链接：https://github.com/samxuxiang/BrepGen

# Environment
- Linux
- Python 3.9
- CUDA 11.8
- Pytorch 2.2
- Diffusers 0.27

# Dependencies
## 1. 安装相关依赖与chamferdist包
```
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

pip install chamferdist
```

chamferdist若下载失败，使用官方提供的chamferdist源下载
```
python setup.py install
```

测试chamferdist是否安装成功
```python
import torch
from chamferdist import ChamferDistance

source_cloud = torch.randn(1, 100, 3).cuda()
target_cloud = torch.randn(1, 50, 3).cuda()

chamferDist = ChamferDistance()

dist_forward = chamferDist(source_cloud, target_cloud)
print(dist_forward.detach().cpu().item())
```
无报错则安装成功！

## 2. 安装pythonOCC与OCCWL
```
conda install -c conda-forge pythonocc-core=7.8.1

pip install git+https://github.com/AutodeskAILab/occwl
```

# Data & Training
训练过程中缺少_parsed数据集，因此无法训练，使用原链接提供的训练好的模型进行测试

# Generation & Evalution
## 1. 随机从高斯噪声中生成B-reps（STEP和STL），使用的是eval_config.yaml中的配置
```
python sample.py --mode abc  # abc数据集，生成数据保存在samples_abc文件夹中

python sample.py --mode deepcad  # deepcad数据集，生成数据保存在samples_deepcad文件夹中
```

## 2. 将生成的B-reps转成点云，之后与测试集中的点云计算评价指标
```
# abc生成的点云数据保存在sampled_pointcloud_abc文件夹中
python sample_points.py --in_dir samples_abc --out_dir sampled_pointcloud_abc

# deepcad生成的点云数据保存在sampled_pointcloud_deepcad文件夹中
python sample_points.py --in_dir samples_deepcad --out_dir sampled_pointcloud_deepcad
```

## 3. 与测试集点云比较，计算评价指标（需要提前从原文链接下载test数据集）
```
# 计算abc数据集的JSD、MMD和COV分数，结果保存在sampled_pointcloud_abc_results.txt中
python pc_metric.py  --fake sampled_pointcloud_abc --real abc_test_pcd

# 计算deepcad数据集的JSD、MMD和COV分数，结果保存在sampled_pointcloud_deepcad_results.txt中
python pc_metric.py  --fake sampled_pointcloud_deepcad --real deepcad_test_pcd
```

## 4. pc_metric代码_pairwise_CD函数部分改动记录
```python
for sample_b_start in pbar:
    sample_batch = sample_pcs[sample_b_start]  # 当前处理的生成点云数据

    cd_lst = []  # 存储每个生成点云与所有测试点云的CD值 1*3000
    emd_lst = []  # 存储所有生成点云与所有测试点云的CD值 3000*3000
    for ref_b_start in range(0, N_ref, 1):  # 由于chamfer_dist函数只能计算batch_size相同的两批点云的CD值，因此将batch_size改成1,和生成点云批次大小（1）一致
        ref_b_end = min(N_ref, ref_b_start + 1)
        ref_batch = ref_pcs[ref_b_start:ref_b_end]  # 每次也取一个测试点云
        
        batch_size_ref = ref_batch.size(0)  # 当前真实点云批次的大小为1
        sample_batch_exp = sample_batch.view(1, -1, 3)  # 将其转为三维张量（1,2000,3），1个点云，2000个点，点的维度为3

        dist_forward1 = chamfer_dist(sample_batch_exp, ref_batch)  # 从生成点云到测试点云的最短距离
        dist_forward2 = chamfer_dist(ref_batch, sample_batch_exp)  # 从测试点云到生成点云的最短距离
        dist_forward_ave = dist_forward1 + dist_forward2  # 两个点云的CD值
        cd_lst.append(torch.tensor([dist_forward_ave]))  # 将其加入到cd_lst中，要先转成张量

    cd_lst = torch.cat(cd_lst, dim=0)  # 将cd_lst中的值按第一维度（由于其中每个张量都是1维，所以直接按顺序）拼接，最后是1*3000，表示第一个生成点云与3000个测试点云的CD值
    all_cd.append(cd_lst)  # 将cd_lst的值放到all_cd中

all_cd = torch.stack(all_cd, dim=0)  # 将all_cd中的张量按行堆砌，最后是3000*3000，每行表示一个生成点云与3000个测试点云的CD值
return all_cd  # 返回CD距离矩阵
```

## 5. 评价结果
|Method|COV % ↑|MMD ↓|JSD ↓|
|-|-|-|-|
|BrepGen（论文中，DeepCAD）|71.26|1.04|0.09|
|BrepGen（复现，DeepCAD，CD1）|64.60|7.97|0.011|
|BrepGen（复现，DeepCAD，CD2）|62.12|6.94|0.013|
|BrepGen（复现，DeepCAD，CD3）|73.64|10.32|0.012|
|BrepGen（复现，DeepCAD，CD4）|73.33|20.60|0.014|
|BrepGen（论文中，ABC）|57.92|1.35|3.69|
|BrepGen（复现，ABC，CD1）|54.17|7.87|0.009|
|BrepGen（复现，ABC，CD2）|58.40|8.05|0.009|
|BrepGen（复现，ABC，CD3）|70.90|12.11|0.009|
|BrepGen（复现，ABC，CD4）|70.11|23.46|0.009|

（1）生成点云与测试点云各3000个，每个点云采样2000个点，运行十次取结果的平均值

（2）倒角距离的计算方法：

    - CD1：只使用前向距离，即 生成点云中所有点到测试点云中最近点的距离之和；
    
    - CD2：只使用后向距离，即 测试点云中所有点到生成点云中最近点的距离之和；
    
    - CD3：（前向距离 + 后向距离） / 2，即（生成点云中所有点到测试点云中最近点的距离之和 + 测试点云中所有点到生成点云中最近点的距离之和） / 2；
    
    - CD4：前向距离 + 后向距离，即 生成点云中所有点到测试点云中最近点的距离之和 + 测试点云中所有点到生成点云中最近点的距离之和。

# Citation
```
@article{xu2024brepgen,
  title={BrepGen: A B-rep Generative Diffusion Model with Structured Latent Geometry},
  author={Xu, Xiang and Lambourne, Joseph G and Jayaraman, Pradeep Kumar and Wang, Zhengqing and Willis, Karl DD and Furukawa, Yasutaka},
  journal={arXiv preprint arXiv:2401.15563},
  year={2024}
}
```
