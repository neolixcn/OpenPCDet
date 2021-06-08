import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

# ============ added by huxi =============
import eval_utils.cpp_result_load_utils as cpp_result_load

# added by huxi, load rpn config
from pcdet.pointpillar_quantize_config import load_rpn_config_json
# ========================================

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, val=False):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        '''
        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        '''

        # === added by huxi ===
        # I (huxi) replaced the result predicted by network with txt result returned by cpp code,
        # so we can use the same metrics to evaluate the result.
        
        config_dict = load_rpn_config_json.get_config()

        data_dir = config_dict["eval_result_txt_dir"]
        #data_dir = "/home/songhongli/huxi/1022_80epoch/out_txt"
        print(str(batch_dict["frame_id"][0]))

        batch_dict_, pred_dicts_, class_names_, output_path_ = cpp_result_load.load_txt_data(str(batch_dict["frame_id"][0]), data_dir)
        annos_ = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts_, class_names,
                output_path=final_output_dir if save_to_file else None
        )

        det_annos += annos_
        # =====================

        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        # logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        # logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)
    # info_f = open('/home/liuwanqiang/OpenPCDet-master/OpenPCDet-master/output/neolix_models/pointpillar_1031/default/eval/epoch_80/val/default/result.pkl', 'rb')
    # det_annos = pickle.load(info_f)
    det_range_ls = None
    # det_range_ls = [[-10, 10, 0, 10], [-10, 10, 10, 30], [-10, 10, 30, 50], [-10, 10, 50, 70]]
    det_range_ls = [[-10, 10, 0, 30]]
    # det_range_ls = [[-10, 10, 0, 10]]
    if not det_range_ls is None:
        for detect_range in det_range_ls:
            print("*" * 60)
            print("Eval range is abs(x) <10, %d < abs(y) < %d" % (detect_range[2], detect_range[3]))
            result_str, result_dict, f2score = dataset.evaluation(
                det_annos, class_names, det_range=detect_range,
                eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
                output_path=final_output_dir
            )

            logger.info(result_str)
            ret_dict.update(result_dict)
            print('The f2score of model epoch%s is %f' % (epoch_id, f2score))
            logger.info('Result is save to %s' % result_dir)
            logger.info('****************Evaluation done.*****************')
        return ret_dict
    else:
        detect_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
        detect_range = [0, detect_range[3], 0, detect_range[4]]
        result_str, result_dict = dataset.evaluation(
            det_annos, class_names, det_range=detect_range,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=final_output_dir
        )

        logger.info(result_str)
        ret_dict.update(result_dict)

        logger.info('Result is save to %s' % result_dir)
        logger.info('****************Evaluation done.*****************')
        return ret_dict


if __name__ == '__main__':
    pass