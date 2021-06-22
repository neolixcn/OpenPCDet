#!/usr/bin/env bash

set -e #任意 command fails 触发

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

#step1. 检查模型
cd ../
source activate torch1.3.1
pwd
pwddir=${PWD} #tools
parentdir="$(dirname "${pwddir}")" #pcdet
eval_or_calib=`cat $parentdir/pcdet/pointpillar_quantize_config/rpn_quantize_config.json | \
    python -c 'import json,sys;obj=json.load(sys.stdin);print(obj["eval_or_calib"])'`;

dataset_yaml_file=`cat $parentdir/pcdet/pointpillar_quantize_config/rpn_quantize_config.json | \
    python -c 'import json,sys;obj=json.load(sys.stdin);print(obj["dataset_yaml_file"])'`;

checkpoint_pth=`cat $parentdir/pcdet/pointpillar_quantize_config/rpn_quantize_config.json | \
    python -c 'import json,sys;obj=json.load(sys.stdin);print(obj["checkpoint_pth"])'`;


echo $eval_or_calib
if [ "$eval_or_calib" = "eval" ]; then
  python test_evaluate_cpp_result.py --cfg_file cfgs/neolix_models/$dataset_yaml_file --ckpt $checkpoint_pth --batch_size 1 --workers 1
else
  #step2. 生成量化表
  # optional: 选择校准数据集，如果之后的数据集没有变化，这一步骤运行时输入除了 ‘y’之外的任意键即可
  python $pwddir/calib_utils/cal_data_distrib.py
  python $pwddir/calib_utils/mv_calib_datasets.py

  copied_calib_pc_dir=`cat $parentdir/pcdet/pointpillar_quantize_config/rpn_quantize_config.json | \
      python -c 'import json,sys;obj=json.load(sys.stdin);print(obj["copied_calib_pc_dir"])'`;

  # --------- 清理掉之前的文件 ---------
  calib_rpn_input_dir=`cat $parentdir/pcdet/pointpillar_quantize_config/rpn_quantize_config.json | \
      python -c 'import json,sys;obj=json.load(sys.stdin);print(obj["calib_rpn_input_dir"])'`;
  echo $calib_rpn_input_dir
  rm -rf $calib_rpn_input_dir/*.*
  # --------^ 清理掉之前的文件 ^--------

  python demo_for_calib.py --cfg_file cfgs/neolix_models/$dataset_yaml_file --ckpt $checkpoint_pth --data_path $copied_calib_pc_dir

  python $pwddir/calib_utils/generate_calib_list.py

  python $pwddir/onnx_utilis/export_vfe_weight.py

  vfe_exported_weight_file=`cat $parentdir/pcdet/pointpillar_quantize_config/rpn_quantize_config.json | \
      python -c 'import json,sys;obj=json.load(sys.stdin);print(obj["vfe_exported_weight_file"])'`;

  rpn_tarfile_name=`cat $parentdir/pcdet/pointpillar_quantize_config/rpn_quantize_config.json | \
      python -c 'import json,sys;obj=json.load(sys.stdin);print(obj["rpn_tarfile_name"])'`;

  echo "ready to scp to xavier: check the file:"
  echo $rpn_tarfile_name
  echo $vfe_exported_weight_file

  echo "done!"
fi
