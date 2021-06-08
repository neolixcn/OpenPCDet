# prepare calib data for rpn quantization.

## How to use


### step0. 创建工作目录结构（optional，在8851上不需要做）
目前8851上的目录结构如下：

1274_pcdet (此即是后面的 your/work/space，是自己创建的)
├── OpenPCDet
├── calib_dataset
    ├── original_bin
├── infer_out_useless
├── vfe_weight_dir

如果是在一个新目录下生成各类数据，也需要创建成一样的形式，通过如下命令：
```shell script
cd your/work/space
git clone https://github.com/neolixcn/OpenPCDet.git
mkdir infer_out_useless
mkdir vfe_weight_dir
mkdir calib_dataset
mkdir calib_datset/original_bin
cd OpenPCDet
python setup.py develop #配置各种环境
```

### step1. 准备配置文件
```shell script
#检查配置文件
code OpenPCDet/blob/rpn-quantize/pcdet/pointpillar_quantize_config/rpn_quantize_config.json
```
如果是8851服务器上量化1022模型，用默认的配置文件即可，更换服务器或是量化新的rpn网络，则需要修改配置文件
按照参数说明填写配置文件，说明如下：

```shell script
{
  "original_all_pc_dir": 
  说明：所有原始点云的位置
  eg. "/nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/training/velodyne",
  
  "copied_calib_pc_dir": 
  说明：我们选择出来的均衡点云数据，需要放置的位置，在我们的calib_dataset下
  eg. "/home/songhongli/1274_pcdet/calib_dataset/original_bin/",
  
  "val_txt_file": 
  说明：原始点云数据的验证集 txt 文件
  eg. "/nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/ImageSets/val.txt",
  
  "eval_data_dir": 
  说明：原始点云数据的 label 目录
  eg. "/nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/training/label_2",
  
  
  "generated_calib_list_file": 
  说明：生成的校准集的list文件，需要放到calib_dataset下
  eg. "/home/songhongli/1274_pcdet/calib_dataset/calib_set_list_rpn.txt",
  
  
  "calib_rpn_input_dir": 
  说明：需要生成的 rpn input data 的放置位置，因为生成的文件较大，建议放在nas服务器上
  eg. "/nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/rpn_input_bin_for_calib/",
  
  
  "rpn_tarfile_name": 
  说明：最后打包的所有 rpn input 还有一个 data_list 文件的包名，这个也就是需要传输到 xavier 上生成量化模型的数据
  eg. "/home/songhongli/1274_pcdet/calib_dataset/tar_file_to_xavier/calib_dataset_and_list.tar.gz",
  
  
  "eval_result_txt_dir": 
  说明：用来验证结果的数据文件夹，从 cpp 端的 pointpillar 的工具可以生成结果，放置到该路径下可以验证结果，
  eg. "/home/songhongli/huxi/1022_80epoch/out_txt",
  
  "vfe_onnx_file": 
  说明：vfe 网络的 onnx 路径
  eg. "/nfs/nas/model_release/pcdet_pointpillars/onnx_ID1022/vfe_1022.onnx",
  
  "vfe_exported_weight_file":
  说明：需要生成的 vfe weight 文件名，之后需要传输到 xavier 上构建 vfe 网络
  eg. "/home/songhongli/1274_pcdet/vfe_weight_dir/vfe_1022_80_onnx.weight",
  
  "eval_or_calib": 
  说明：表明当前任务是评测指标还是生成量化数据，可选 "eval" 和 "calib"，
  eg. "calib"
}
```

### step2. 生成量化数据（或评测指标）
```shell script
cd your/work/space
cd OpenPCDet/tools/script
chmod 777 rpn_quantize.sh
./rpn_quantize.sh
```

注意：如果选择评测指标，则需要把cpp端生成的数据放到 "eval_result_txt_dir" 路径下。
