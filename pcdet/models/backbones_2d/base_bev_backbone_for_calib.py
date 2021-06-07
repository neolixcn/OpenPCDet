import numpy as np
import torch
import torch.nn as nn

rpn_id = -1
class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        global rpn_id
        rpn_id += 1
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features

        print("we are generating rpn out, if you don't aim to do so, pls take care if you are using the correct base_bev_backbone.py!!!")
        # ===================================================== #
        
        bin_array_to_save = x.detach().cpu().numpy()
        print(bin_array_to_save.shape)
        print(bin_array_to_save.dtype)
        bin_array_to_save_1d = bin_array_to_save.flatten()
        rpn_input_data_path = "/nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/rpn_input_bin_for_calib/"
        import os
        
        if os.path.isdir(rpn_input_data_path[0:-1]) == False:
            print("we didn't find a dir called " + rpn_input_data_path + ", now we gonna create it.")
            os.mkdir(rpn_input_data_path)
        else:
            print("we find a dir called " + rpn_input_data_path + ", now we gonna generate rpn input data to it.")

        bin_array_to_save_1d.tofile(rpn_input_data_path + str(rpn_id) +'.bin')
        
        # ====================== for calib ==================== #

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict

    # forward for export pointpillars onnx
    # def forward(self, spatial_features):
    #     """
    #     Args:
    #         data_dict:
    #             spatial_features
    #     Returns:
    #     """
    #     # spatial_features = data_dict['spatial_features']
    #     ups = []
    #     ret_dict = {}
    #     x = spatial_features
    #     for i in range(len(self.blocks)):
    #         x = self.blocks[i](x)
    #
    #         # stride = int(spatial_features.shape[2] / x.shape[2])
    #         stride = torch.floor_divide(spatial_features.shape[2], x.shape[2])
    #         ret_dict['spatial_features_%dx' % stride] = x
    #         if len(self.deblocks) > 0:
    #             ups.append(self.deblocks[i](x))
    #         else:
    #             ups.append(x)
    #
    #     if len(ups) > 1:
    #         x = torch.cat(ups, dim=1)
    #     elif len(ups) == 1:
    #         x = ups[0]
    #
    #     if len(self.deblocks) > len(self.blocks):
    #         x = self.deblocks[-1](x)
    #
    #     # data_dict['spatial_features_2d'] = x
    #
    #     # return data_dict
    #
    #     return x