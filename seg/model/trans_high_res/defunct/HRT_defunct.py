import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
import yaml

from .parts_defunct import BasicBlock, Bottleneck, HighResolutionModule, SpatialGather_Module, SpatialOCR_Module

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

ALIGN_CORNERS = True

class HRT(nn.Module):
    def __init__(
        self,
        config_path,
    ):
        super(HRT, self).__init__()
        with open(config_path, "r") as stream:
            try:
                self.config = yaml.load(stream, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)

        # model construction 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=self.config['BM_MOMENTUM'])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=self.config['BM_MOMENTUM'])
        self.relu = nn.ReLU(inplace=False)

        # stage 1 
        self.stage1_config = self.config['STAGE1']
        num_channels = self.stage1_config['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_config['BLOCK']]
        num_blocks = self.stage1_config['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = self.config['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = self.config['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = self.config['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        last_inp_channels = np.int(np.sum(pre_stage_channels))
        ocr_mid_channels = self.config['OCR_MID_CHANNELS']
        ocr_key_channels = self.config['OCR_KEY_CHANNELS']

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(last_inp_channels, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=False),
        )
        self.ocr_gather_head = SpatialGather_Module(self.config['NUM_CLASSES'])

        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, self.config['NUM_CLASSES'], kernel_size=1, stride=1, padding=0, bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(last_inp_channels, self.config['NUM_CLASSES'],
                      kernel_size=1, stride=1, padding=0, bias=True)
        )


    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], momentum=self.config['BM_MOMENTUM']),
                        nn.ReLU(inplace=False)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=self.config['BM_MOMENTUM']),
                        nn.ReLU(inplace=False)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=self.config['BM_MOMENTUM']),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels 

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.layer1(x)

        # just temporary DELETE THIS 
        # x_list = []
        # for i in range(self.stage2_cfg['NUM_BRANCHES']):
        #     if self.transition1[i] is not None:
        #         x_list.append(self.transition1[i](x))
        #     else:
        #         x_list.append(x)
        # y_list = self.stage2(x_list)

        # x_list = []
        # for i in range(self.stage3_cfg['NUM_BRANCHES']):
        #     if self.transition2[i] is not None:
        #         if i < self.stage2_cfg['NUM_BRANCHES']:
        #             x_list.append(self.transition2[i](y_list[i]))
        #         else:
        #             x_list.append(self.transition2[i](y_list[-1]))
        #     else:
        #         x_list.append(y_list[i])
        # y_list = self.stage3(x_list)

        # x_list = []
        # for i in range(self.stage4_cfg['NUM_BRANCHES']):
        #     if self.transition3[i] is not None:
        #         if i < self.stage3_cfg['NUM_BRANCHES']:
        #             x_list.append(self.transition3[i](y_list[i]))
        #         else:
        #             x_list.append(self.transition3[i](y_list[-1]))
        #     else:
        #         x_list.append(y_list[i])
        # x = self.stage4(x_list)

        # # Upsampling
        # x0_h, x0_w = x[0].size(2), x[0].size(3)
        # x1 = F.interpolate(x[1], size=(x0_h, x0_w),
        #                 mode='bilinear', align_corners=ALIGN_CORNERS)
        # x2 = F.interpolate(x[2], size=(x0_h, x0_w),
        #                 mode='bilinear', align_corners=ALIGN_CORNERS)
        # x3 = F.interpolate(x[3], size=(x0_h, x0_w),
        #                 mode='bilinear', align_corners=ALIGN_CORNERS)

        # feats = torch.cat([x[0], x1, x2, x3], 1)

        # out_aux_seg = []

        # # ocr
        # out_aux = self.aux_head(feats)
        # # compute contrast feature
        # feats = self.conv3x3_ocr(feats)

        # context = self.ocr_gather_head(feats, out_aux)
        # feats = self.ocr_distri_head(feats, context)

        # out = self.cls_head(feats)

        # out_aux_seg.append(out_aux)
        # out_aux_seg.append(out)

        return x

if __name__ == '__main__':
    hrt = HRT(config_path = 'seg/model/trans_high_res/stages.yaml')
    
    from torchsummary import summary
    
    summary(
        model = hrt.cuda(),
        input_size = (3, 256, 256),
        batch_size = 10,
    )