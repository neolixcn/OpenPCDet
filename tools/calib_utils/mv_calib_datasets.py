import os
import shutil

def move_data_to_dir(file_name, src_path, des_path):
    f_src = os.path.join(src_path, file_name)
    if not os.path.exists(des_path):
        os.mkdir(des_path)
    f_des = os.path.join(des_path, file_name)
    shutil.copyfile(f_src, f_des)

def read_calib_set_list(file_name, src_data_path, des_data_path):
    with open(file_name, 'r') as f:
        for line in f.readlines():
            data_name = line.strip() + '.bin'
            move_data_to_dir(data_name, src_data_path, des_data_path)

if __name__ == '__main__':

    file_name = '/home/songhongli/1274_pcdet/calib_dataset/calib_set_list_rpn.txt'
    import os
    if os.path.isfile(file_name) == False:
        print("we didn't find " + file_name + ", pls check!")
    else:
        
        data_path = '/nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/training/velodyne'
        des_data_path = '/home/songhongli/1274_pcdet/calib_dataset/original_bin/'
        if os.path.isdir(des_data_path[0:-1]) == False:
            print("we didn't find a dir called " + des_data_path + ", now we gonna create it.")
            os.mkdir(des_data_path)
            print("copying pc ...")
            read_calib_set_list(file_name, data_path, des_data_path)
        else:
            print("we find a dir called " + des_data_path + ", now we gonna clean it.")
            shutil.rmtree(des_data_path)
            os.mkdir(des_data_path)
            print("copying pc ...")
            read_calib_set_list(file_name, data_path, des_data_path)
