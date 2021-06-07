import numpy
import glob



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

    calib_rpn_in_dir = "/nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/rpn_input_bin_for_calib/"
    list_all_bin(calib_rpn_in_dir)
    print("we have all data for calib now, let's start to generate tar file for xavier!")
    tarfile_name = "/home/songhongli/1274_pcdet/calib_dataset/tar_file_to_xavier/calib_dataset_and_list.tar.gz"
    generate_targz(calib_rpn_in_dir, tarfile_name)

