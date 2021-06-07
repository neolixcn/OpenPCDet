
import glob
import numpy as np
import torch

'''
func:   加载从point pillar unit test中生成的 txt 结果，并转换为可以直接替代 test.py 中的 annos 形式，
        以便直接替代原本的网络输入完成评测
param1: pointpillar 的输出 txt 文件名，eg. "000001.txt"
param2: pointpillar 的输出 txt 文件的路径，eg. "/home/songhongli/huxi/out_txt_fp16"

其中 txt 的形式为每一个 bbox 占两行：
box_id x y z dx dy dz yaw
prob_cls1 prob_cls2 prob_cls3 prob_cls4 prob_cls5

eg.
0 -9.840708 -14.545975 0.146788 4.590909 1.862477 1.579407 6.328176
0.911058 0.000000 0.000000 0.000000 0.000000 
1 -10.030737 -5.604728 0.257478 4.549955 1.903857 1.572961 6.251483
0.909733 0.000000 0.000000 0.000000 0.000000 
2 -10.033637 -8.149426 0.204048 4.394748 1.816366 1.534151 6.314526
0.903779 0.000000 0.000000 0.000000 0.000000 
...

'''

def load_txt_data(pp_txt_file, data_dir):
    pred_dict = {}
    batch_dict = {}
    class_names = []
    file_name = data_dir + "/" + pp_txt_file + ".pcd.txt"
    print(file_name)
    with open(file_name, 'r') as f:
        frame_id = pp_txt_file[0:-8]
        batch_dict["frame_id"] = [frame_id] #dummy batch_dict ， batch 设为1就好
        pred_boxes = []
        pred_scores = []
        pred_labels = []
        for id,line in enumerate(f):
            if(id%2 == 0): #object_id + boxes
                all_num_str_list = line.strip().split(" ")
                
                temp_list = []
                for i in range(1,len(all_num_str_list)):
                    temp_list.append(float(all_num_str_list[i]))
                pred_boxes.append(temp_list)
            elif(id%2 == 1): #scores labels
                all_prob_str_list = line.strip().split(" ")
                temp_list = []
                for i in range(0,len(all_prob_str_list)):
                    temp_list.append(float(all_prob_str_list[i]))
                temp_array = np.array(temp_list)
                pred_scores.append(np.max(temp_array))
                pred_labels.append(np.argmax(temp_array))

        pred_boxes = np.array(pred_boxes)
        pred_scores = np.array(pred_scores)
        pred_labels = np.array(pred_labels) + 1

        pred_boxes = torch.from_numpy(pred_boxes).float().to("cuda:0")
        pred_scores = torch.from_numpy(pred_scores).float().to("cuda:0")
        pred_labels = torch.from_numpy(pred_labels).int().to("cuda:0")
        pred_dict = {"pred_boxes": pred_boxes, "pred_scores": pred_scores, "pred_labels": pred_labels}
        class_names = ['Vehicle', 'Pedestrian', 'Cyclist', 'Unknown', 'Large_vehicle']
        output_path = None
    return batch_dict, [pred_dict], class_names, output_path


def to_vis_format(pp_txt_file):
    file_name = pp_txt_file
    print(file_name)
    with open(file_name, 'r') as f:
        vis_format_list = []
        temp_vis_list = []
        for id,line in enumerate(f):
            
            if(id%2 == 0): #object_id + boxes
                temp_vis_list = [0]
                all_num_str_list = line.strip().split(" ")
                temp_list = []
                for i in range(1,len(all_num_str_list)):
                    temp_list.append(float(all_num_str_list[i]))
                temp_vis_list.extend(temp_list)
            elif(id%2 == 1): #scores labels
                temp_list = []
                all_prob_str_list = line.strip().split(" ")
                for i in range(0,len(all_prob_str_list)):
                    temp_list.append(float(all_prob_str_list[i]))
                
                temp_array = np.array(temp_list)
                temp_vis_list[0] = np.argmax(temp_array)
                temp_vis_list.append(np.max(temp_array))

                vis_format_list.append(temp_vis_list)
                temp_vis_list = []

        vis_array = np.array(vis_format_list)
        return vis_array
            

                    

if __name__ == "__main__":
    data_dir = "/home/songhongli/huxi/1022_80epoch/out_txt" #glob 不识别 ～符号作为目录

    for idx, pp_txt_file in enumerate(glob.glob(data_dir+"/*")):

        vis_result = to_vis_format(pp_txt_file)
        print(vis_result)
        np.save(pp_txt_file + '.npy', vis_result)
        if idx >= 10:
            break
        