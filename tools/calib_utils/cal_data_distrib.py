import math
# added by huxi, load rpn config
from pcdet.pointpillar_quantize_config import load_rpn_config_json
# ==============================


'''
func: 输入文件列表,解析每个文件，统计各个类别在不同的距离下的个数
param1: 文件列表 => list
param2: 文件所在路径 => string
output: 统计字典 => dict: {'file0':{ 0: {'vehicle': 10, 'pedestrain': 5, ...}, 1: {'vehicle': 20, 'pedestrain': 2, ...}...}}
'''

'''
Unknown 0 1 0 0.0 0.0 0.0 0.0 0.23014181559057478 0.34533842474954585 0.3839701660969853 -8.522485840062638 3.36335813253329 -0.4699711955074279 -0.016875835173386866
'''
def parse_file(txt_file, data_path):
    full_txt_name = data_path + '/' + txt_file
    dict_file_info = {}
    with open(full_txt_name, 'r') as f:
        class_dict = {'Vehicle':0, 'Pedestrian':0, 'Cyclist':0, 'Unknown':0, 'Large_vehicle':0}
        distance_dict = {0:{'Vehicle':0, 'Pedestrian':0, 'Cyclist':0, 'Unknown':0, 'Large_vehicle':0}, 
                         1:{'Vehicle':0, 'Pedestrian':0, 'Cyclist':0, 'Unknown':0, 'Large_vehicle':0}, 
                         2:{'Vehicle':0, 'Pedestrian':0, 'Cyclist':0, 'Unknown':0, 'Large_vehicle':0}, 
                         3:{'Vehicle':0, 'Pedestrian':0, 'Cyclist':0, 'Unknown':0, 'Large_vehicle':0}}

        for line in f.readlines():
            all_items = line.strip().split(' ')
            cls_name =  all_items[0]
            px = float(all_items[-4])
            py = float(all_items[-3])
            distance = math.sqrt(px*px + py*py)
            #print(cls_name, px, py, distance)
            if(distance <= 10):
                distance_id = 0
            elif(distance > 10 and distance <= 30):
                distance_id = 1
            elif(distance > 30 and distance <= 50):
                distance_id = 2
            elif(distance > 50 and distance <= 70):
                distance_id = 3
            else:
                continue


            distance_dict[distance_id][cls_name] += 1

        return distance_dict


def accumulate_counts(range_idx, count_all_cls, reserve_file_list, dict_current_range, distrib_dict, aim_cls, threshold_amount):
    sorted_dict_current_range = dict(sorted(dict_current_range.items(), key=lambda item:item[1][aim_cls], reverse=True))
    idx = 0
    for key in (sorted_dict_current_range.keys()):
        idx += 1
        if idx == threshold_amount:
            break
        if(key not in reserve_file_list):
            reserve_file_list.append(key)
            for i in range(4):
                count_all_cls[0] += distrib_dict[key][i]["Vehicle"]
                count_all_cls[1] += distrib_dict[key][i]["Pedestrian"]
                count_all_cls[2] += distrib_dict[key][i]["Cyclist"]
                count_all_cls[3] += distrib_dict[key][i]["Unknown"]
                count_all_cls[4] += distrib_dict[key][i]["Large_vehicle"]
        else:
            continue
   
    print(len(reserve_file_list))
    print(count_all_cls)



def cal_distrib(txt_list, data_path, generated_file):
    distrib_dict = {}
    dict_range_0to10 = {}
    dict_range_10to30 = {}
    dict_range_30to50 = {}
    dict_range_50to70 = {}
    class_dict = {'Vehicle':0, 'Pedestrian':0, 'Cyclist':0, 'Unknown':0, 'Large_vehicle':0}
    statistic_table = {0:{'Vehicle':0, 'Pedestrian':0, 'Cyclist':0, 'Unknown':0, 'Large_vehicle':0}, 
                         1:{'Vehicle':0, 'Pedestrian':0, 'Cyclist':0, 'Unknown':0, 'Large_vehicle':0}, 
                         2:{'Vehicle':0, 'Pedestrian':0, 'Cyclist':0, 'Unknown':0, 'Large_vehicle':0}, 
                         3:{'Vehicle':0, 'Pedestrian':0, 'Cyclist':0, 'Unknown':0, 'Large_vehicle':0}}
    
    for txt_file in txt_list:
        dict_file_info = parse_file(txt_file, data_path)
        distrib_dict[txt_file] = dict_file_info
        dict_range_0to10[txt_file] = dict_file_info[0]
        dict_range_10to30[txt_file] = dict_file_info[1]
        dict_range_30to50[txt_file] = dict_file_info[2]
        dict_range_50to70[txt_file] = dict_file_info[3]

        for i in range(4):
            statistic_table[i]['Vehicle'] += dict_file_info[i]['Vehicle']
            statistic_table[i]['Pedestrian'] += dict_file_info[i]['Pedestrian']
            statistic_table[i]['Cyclist'] += dict_file_info[i]['Cyclist']
            statistic_table[i]['Unknown'] += dict_file_info[i]['Unknown']
            statistic_table[i]['Large_vehicle'] += dict_file_info[i]['Large_vehicle']  

    print("statistic table over all ranges and classes:")    
    print(statistic_table)
    input("pause, press enter to further select data.")

    import numpy as np

    count_all_cls = np.array([0, 0, 0, 0, 0])
    class_name_list = ['Vehicle', 'Pedestrian', 'Cyclist', 'Unknown', 'Large_vehicle']

    reserve_file_list = []
    times = 0
    print("0 t0 10:")
    accumulate_counts(0, count_all_cls, reserve_file_list, dict_range_0to10, distrib_dict, "Large_vehicle", 80)
    accumulate_counts(0, count_all_cls, reserve_file_list, dict_range_0to10, distrib_dict, "Cyclist", 80)
    accumulate_counts(0, count_all_cls, reserve_file_list, dict_range_0to10, distrib_dict, "Pedestrian", 80)
    print("=========")

    print("0 t0 30:")
    accumulate_counts(0, count_all_cls, reserve_file_list, dict_range_10to30, distrib_dict, "Large_vehicle", 50)
    accumulate_counts(0, count_all_cls, reserve_file_list, dict_range_10to30, distrib_dict, "Cyclist", 100)
    accumulate_counts(0, count_all_cls, reserve_file_list, dict_range_10to30, distrib_dict, "Pedestrian", 100)
    print("=========")

    
    print("0 t0 50:")
    accumulate_counts(0, count_all_cls, reserve_file_list, dict_range_30to50, distrib_dict, "Large_vehicle", 70)
    accumulate_counts(0, count_all_cls, reserve_file_list, dict_range_30to50, distrib_dict, "Cyclist", 70)
    accumulate_counts(0, count_all_cls, reserve_file_list, dict_range_30to50, distrib_dict, "Pedestrian", 70)
    print("=========")

    print("0 t0 70:")
    accumulate_counts(0, count_all_cls, reserve_file_list, dict_range_50to70, distrib_dict, "Large_vehicle", 50)
    accumulate_counts(0, count_all_cls, reserve_file_list, dict_range_50to70, distrib_dict, "Cyclist", 50)
    accumulate_counts(0, count_all_cls, reserve_file_list, dict_range_50to70, distrib_dict, "Pedestrian", 50)
    print("=========")
    
    
    output_list = []
    for i in reserve_file_list:
        output_list.append(i[0:-4]+'\n')
    with open(generated_file, 'w') as f:
        f.writelines(output_list)


    class_dict = {'Vehicle':0, 'Pedestrian':0, 'Cyclist':0, 'Unknown':0, 'Large_vehicle':0}
    statistic_table = {0:{'Vehicle':0, 'Pedestrian':0, 'Cyclist':0, 'Unknown':0, 'Large_vehicle':0}, 
                         1:{'Vehicle':0, 'Pedestrian':0, 'Cyclist':0, 'Unknown':0, 'Large_vehicle':0}, 
                         2:{'Vehicle':0, 'Pedestrian':0, 'Cyclist':0, 'Unknown':0, 'Large_vehicle':0}, 
                         3:{'Vehicle':0, 'Pedestrian':0, 'Cyclist':0, 'Unknown':0, 'Large_vehicle':0}}
    
    for txt_file in reserve_file_list:
        dict_file_info = parse_file(txt_file, data_path)
        for i in range(4):
            statistic_table[i]['Vehicle'] += dict_file_info[i]['Vehicle']
            statistic_table[i]['Pedestrian'] += dict_file_info[i]['Pedestrian']
            statistic_table[i]['Cyclist'] += dict_file_info[i]['Cyclist']
            statistic_table[i]['Unknown'] += dict_file_info[i]['Unknown']
            statistic_table[i]['Large_vehicle'] += dict_file_info[i]['Large_vehicle']  

    print("statistic table after selecting:")    
    print(statistic_table)

if __name__ == '__main__':
    
    config_dict = load_rpn_config_json.get_config()
    txt_file = config_dict["val_txt_file"]
    eval_path = config_dict["eval_data_dir"]
    generated_file = config_dict["generated_calib_list_file"]

    #txt_file = '/nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/ImageSets/val.txt'
    #eval_path = '/nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/training/label_2'
    eval_name_list = []
    #generated_file = '/home/songhongli/1274_pcdet/calib_dataset/calib_set_list_rpn.txt'
    import os.path 
    if os.path.isfile(generated_file):
        key_input = input("we already have a file named " + generated_file + ", if you are sure to replace it, pls enter 'y'. other pls enter any other key, we will end this process:")
        if key_input == 'y' or key_input == 'Y':
            with open(txt_file, 'r') as f:
                for line in f.readlines():
                    eval_name_list.append(line.strip() + '.txt')
            cal_distrib(eval_name_list, eval_path, generated_file)
