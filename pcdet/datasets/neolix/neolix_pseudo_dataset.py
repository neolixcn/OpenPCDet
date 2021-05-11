import copy
import pickle
import os
from pathlib import Path

import numpy as np
import torch
from skimage import io
import concurrent.futures as futures

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_neolix, common_utils, object3d_neolix
from ..dataset import DatasetTemplate

data_id = -1
write_data_id = -1


class NeolixDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, inference=False):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.neolix_infos = []
        self.include_neolix_data(self.mode)
        self.inference = inference
        self.process_results_dir = '/nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1026/process_data/%s_process_results' % self.split

    def include_neolix_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Neolix dataset')
        neolix_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                neolix_infos.extend(infos)

        self.neolix_infos.extend(neolix_infos)

        if self.logger is not None:
            self.logger.info('Total samples for Neolix dataset: %d' % (len(neolix_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        if not self.inference:
            lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
            assert lidar_file.exists()
            pc = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
            return pc
        else:
            root_path = "/home/songhongli/bins/"
            # root_path = "/nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/training/val_pc/"
            fl_ls = sorted(os.listdir(root_path))
            global data_id
            data_id += 1
            print("fl_ls[data_id]", fl_ls[data_id])
            lidar_file = root_path + fl_ls[data_id]
            pc = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
            pc[:, 2] = pc[:, 2] + 1.4
            return pc

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        # print(label_file)
        assert label_file.exists()
        return object3d_neolix.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_neolix.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            # image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            # info['image'] = image_info
            # calib = self.get_calib(sample_idx)

            # P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            # R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            # R0_4x4[3, 3] = 1.
            # R0_4x4[:3, :3] = calib.R0
            # V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            # calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            # info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                # loc_lidar = calib.rect_to_lidar(loc)
                loc_lidar = loc
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    # calib = self.get_calib(sample_idx)
                    # pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    # fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    # pts_fov = points[fov_flag]
                    pts_fov = points
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        # print(sample_id_list)
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train',num_workers=4):

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('neolix_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        def create_single_gt_points(k):
            single_db_infos = {}
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                # Pyten-mkdir
                filepath.parent.mkdir(parents=True, exist_ok=True)
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    # db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                    #            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                    #            'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    db_info = {'name': names[i], 'path': db_path, 'pc_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in single_db_infos:
                        single_db_infos[names[i]].append(db_info)
                    else:
                        single_db_infos[names[i]] = [db_info]
            return single_db_infos

        with futures.ThreadPoolExecutor(num_workers) as executor:
            for result in executor.map(create_single_gt_points, range(len(infos))):
                for k in result:
                    if k in all_db_infos:
                        all_db_infos[k].extend(result[k])
                    else:
                        all_db_infos[k] = result[k]

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None, inference=False, inference_results=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            # calib = batch_dict['calib'][batch_index]
            # image_shape = batch_dict['image_shape'][batch_index]
            # pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            # pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
            #     pred_boxes_camera, calib, image_shape=image_shape
            # )
            pred_boxes_camera = box_utils.boxes3d_lidar_to_neolix_camera(pred_boxes)

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            # pred_dict['bbox'] = pred_boxes_img
            pred_dict['bbox'] = np.zeros([pred_boxes.shape[0], 4])
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]
            class_dict = {'Vehicle': 0, 'Large_vehicle': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Bicycle': 4,
                          'Unknown_movable':5 , 'Unknown_unmovable': 6}
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)
            # output_path = './new_label/'
            if output_path is not None:
                inference = False
                if inference:
                    # label_path = inference_results
                    label_path = './val_label/'
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl
                    content = []
                    for idx in range(len(bbox)):
                        type_name = class_dict[single_pred_dict['name'][idx]]
                        prob_ls = list(np.zeros(7))
                        prob_ls[type_name] = single_pred_dict['score'][idx]

                        content.append([loc[idx][0], loc[idx][1], loc[idx][2] , dims[idx][2],
                                        dims[idx][0], dims[idx][1],  -np.pi/2-single_pred_dict['rotation_y'][idx],
                                        type_name] + prob_ls)
                    np.array(content).astype(np.float32).tofile(label_path + batch_dict['frame_id'][index]+'.bin')
                    # with open(label_path + batch_dict['frame_id'][index] + '.txt', 'w') as f:
                    #     bbox = single_pred_dict['bbox']
                    #     loc = single_pred_dict['location']
                    #     dims = single_pred_dict['dimensions']  # lhw -> hwl
                    #     content = []
                    #     for idx in range(len(bbox)):
                    #         content.append('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                    #               % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                    #                  bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                    #                  dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                    #                  loc[idx][1], loc[idx][2]-dims[idx][1]/2, -np.pi/2-single_pred_dict['rotation_y'][idx],
                    #                  single_pred_dict['score'][idx]))
                    #     f.write("\n".join(content))
                else:
                    pass
                    # cur_det_file = output_path / ('%s.txt' % frame_id)
                    # with open(cur_det_file, 'w') as f:
                    #     bbox = single_pred_dict['bbox']
                    #     loc = single_pred_dict['location']
                    #     dims = single_pred_dict['dimensions']  # lhw -> hwl
                    #     for idx in range(len(bbox)):
                    #         print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                    #               % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                    #                  bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                    #                  dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                    #                  loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                    #                  single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, det_range=None, **kwargs):
        if 'annos' not in self.neolix_infos[0].keys():
            return None, {}

        from .neolix_object_eval_python import eval as neolix_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.neolix_infos]
        ap_result_str, ap_dict = neolix_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names, det_range=det_range)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.neolix_infos) * self.total_epochs

        return len(self.neolix_infos)


    def __generateitem__(self, num_workers=4):
        len_info = len(self.neolix_infos)
        print(f"generating {len_info} data_dict as npy format in {self.process_results_dir}")
        os.makedirs(self.process_results_dir, exist_ok=True)
        def process_single_data(index):
            if self._merge_all_iters_to_one_epoch:
                index = index % len(self.neolix_infos)

            info = copy.deepcopy(self.neolix_infos[index])

            sample_idx = info['point_cloud']['lidar_idx']

            points = self.get_lidar(sample_idx)
            input_dict = {
                'points': points,
                'frame_id': sample_idx,
            }

            if 'annos' in info:
                annos = info['annos']
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_names = annos['name']
                gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_neolix_camera_to_lidar(gt_boxes_camera)

                input_dict.update({
                    'gt_names': gt_names,
                    'gt_boxes': gt_boxes_lidar
                })
                road_plane = self.get_road_plane(sample_idx)
                if road_plane is not None:
                    input_dict['road_plane'] = road_plane

            data_dict = self.prepare_data(data_dict=input_dict)

            np.save('%s/%d.npy' % (self.process_results_dir, index), data_dict)
        
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_data, range(len_info))


    def __getitem__(self, index):
        data_dict = np.load('%s/%d.npy' % (self.process_results_dir, index), allow_pickle=True).item()
        return data_dict


    def __getitemv2__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.neolix_infos)

        info = copy.deepcopy(self.neolix_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']

        points = self.get_lidar(sample_idx)
        # calib = self.get_calib(sample_idx)

        # img_shape = info['image']['image_shape']
        self.dataset_cfg.FOV_POINTS_ONLY = False
        if self.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            # 'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            # gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            # gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_neolix_camera_to_lidar(gt_boxes_camera)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        data_dict = self.prepare_data(data_dict=input_dict)

        # data_dict['image_shape'] = img_shape
        return data_dict


def create_neolix_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = NeolixDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('neolix_infos_%s.pkl' % train_split)
    val_filename = save_path / ('neolix_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'neolix_infos_trainval.pkl'
    test_filename = save_path / 'neolix_infos_test.pkl'

    print('---------------Start to generate data infos---------------')
    
    dataset.set_split(train_split)
    neolix_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(neolix_infos_train, f)
    print('Neolix info train file is saved to %s' % train_filename)
    
    dataset.set_split(val_split)
    neolix_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(neolix_infos_val, f)
    print('Neolix info val file is saved to %s' % val_filename)
    
    with open(trainval_filename, 'wb') as f:
        pickle.dump(neolix_infos_train + neolix_infos_val, f)
    print('Neolix info trainval file is saved to %s' % trainval_filename)

    # # dataset.set_split('test')
    # # neolix_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    # # with open(test_filename, 'wb') as f:
    # #     pickle.dump(neolix_infos_test, f)
    # # print('Neolix info test file is saved to %s' % test_filename)
    
    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    
    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_neolix_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        # ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        # ROOT_DIR = Path('/nfs/neolix_data1/neolix_dataset/develop_dataset/obstacle_detect/songhongli/shanghai32_lidars')
        ROOT_DIR = Path('/nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1026/')
        create_neolix_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Large_vehicle', 'Pedestrian', 'Cyclist', 'Bicycle', 'Unknown_movable', 'Unknown_unmovable'],
            data_path=ROOT_DIR,
            save_path=ROOT_DIR
            # data_path=ROOT_DIR / 'data' / 'neolix',
            # save_path=ROOT_DIR / 'data' / 'neolix'
        )#class_names=['Vehicle', 'Pedestrian', 'Cyclist', 'Unknown', 'Large_vehicle']
        #['Vehicle', 'Large_vehicle', 'Pedestrian', 'Cyclist', 'Bicycle', 'Unknown_movable', 'Unknown_unmovable']