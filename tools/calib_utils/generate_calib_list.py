import numpy
import glob
# added by huxi, load rpn config
from pcdet.pointpillar_quantize_config import load_rpn_config_json
# ==============================



def list_all_bin(calib_rpn_in_dir):
    bin_files = glob.glob(calib_rpn_in_dir + '*.bin')
    with open(calib_rpn_in_dir + "calib_rpn_list.txt", "w") as f:       
        for bin_file in bin_files:
            f.writelines(bin_file[len(calib_rpn_in_dir):])
            f.writelines("\n")

def generate_targz(calib_rpn_in_dir, tarfile_name):
    import tarfile
    import os
    try:
        with tarfile.open(tarfile_name, "w:gz") as tar:
            tar.add(calib_rpn_in_dir, arcname=os.path.basename(calib_rpn_in_dir))
        return True

    except Exception as e:
        print(e)
        return False

if __name__ == '__main__':

    config_dict = load_rpn_config_json.get_config()
    calib_rpn_in_dir = config_dict["calib_rpn_input_dir"]

    #calib_rpn_in_dir = "/nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/rpn_input_bin_for_calib/"
    list_all_bin(calib_rpn_in_dir)
    print("we have all data for calib now, let's start to generate tar file for xavier!")
    tarfile_name = config_dict["rpn_tarfile_name"]
    #tarfile_name = "/home/songhongli/1274_pcdet/calib_dataset/tar_file_to_xavier/calib_dataset_and_list.tar.gz"
    
    generate_targz(calib_rpn_in_dir, tarfile_name)

