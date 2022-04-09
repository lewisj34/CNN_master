import torch
import yaml 
import torch.nn as nn 

from .parts import Bottleneck, BasicBlock, Elementary, iElementaryPlusInterChannels


ALLOWABLE_MODULE_TYPES = ['bottleneck', 'basic', 'elementary', 'ielementaryplus']

class Stage(nn.Module):
    def __init__(
        self,
        stage_dict,
    ):
        super(Stage, self).__init__()
        self.stage_checker(stage_dict)
        # print(stage_dict)
        num_levels = len(stage_dict['levels'])

        levels = list()
        for i in range(num_levels):

            modules_for_level = stage_dict['levels']['level' + str(i+1)]['module_type']
            channels_for_level = stage_dict['levels']['level' + str(i+1)]['channels']

            # generate layer information 
            assert len(modules_for_level) == len(channels_for_level), \
                f'num_modules !== num_channels for curr_stage, check yaml config'
            num_modules = len(modules_for_level)

            level_i = list()
            for j in range(num_modules):
                if modules_for_level[j] == 'bottleneck':
                    assert len(channels_for_level[j]) == 3, f'(Level, module/chan_num): {(i+1, j+1)}. {modules_for_level[j]} module requires definition of input, intermediate, and output channels.'
                    level_i.append(Bottleneck(
                        in_chans = channels_for_level[j][0],
                        inter_chans = channels_for_level[j][1],
                        out_chans = channels_for_level[j][2],
                        stride=1,
                        bn_momentum=0.1,
                        relu_inplace=False,
                        downsample=None,
                    ))
                elif modules_for_level[j] == 'basic':
                    assert len(channels_for_level[j]) == 2, f'(Level, module/chan_num): {(i+1, j+1)}. {modules_for_level[j]} module requires definition of input and output channels.'
                    level_i.append(BasicBlock(
                        in_chans = channels_for_level[j][0],
                        out_chans = channels_for_level[j][1],
                        stride=1, 
                        bn_momentum=0.1, 
                        relu_inplace=False,
                        downsample=None,
                    ))
                elif modules_for_level[j] == 'elementary':
                    assert len(channels_for_level[j]) == 2, f'(Level, module/chan_num): {(i+1, j+1)}. {modules_for_level[j]} module requires definition of input and output channels.'
                    level_i.append(Elementary(
                        in_chans = channels_for_level[j][0],
                        out_chans = channels_for_level[j][1],
                        bn_momentum=0.1, 
                        relu_inplace=False,
                        fuse_type='sum',
                    ))
                elif modules_for_level[j] == 'ielementaryplus':
                    assert len(channels_for_level[j]) == 3, f'(Level, module/chan_num): {(i+1, j+1)}. {modules_for_level[j]} module requires definition of input, intermediate, and output channels.'
                    level_i.append(iElementaryPlusInterChannels(
                        in_chans = channels_for_level[j][0],
                        inter_chans = channels_for_level[j][1],
                        out_chans = channels_for_level[j][2],
                        bn_momentum=0.1,
                        relu_inplace=False,
                        fuse_type='sum',
                    ))
                else:
                    raise ValueError(f'modules_for_level[j], j: {modules_for_level[j], j}, invalid. Allowable modules: {ALLOWABLE_MODULE_TYPES}')
            # now append the complete individual level list to the master list 
            levels.append(level_i)
        
        for i in range(len(levels)):
            for j in range(len(levels[i])):
                    print(f'Content of (level: {i, j}): {levels[i][j]._get_name()}')
        print(f'length of the levels list: {len(levels)}')
        print(f'length of the levels[0] list: {len(levels[0])}')
        print(f'length of the levels[1] list: {len(levels[1])}')
        print(f'length of the levels[2] list: {len(levels[2])}')

    def forward(self, x):
        x = nn.Conv2d(x)

    def allUnique(self, x):
        seen = set()
        return not any(i in seen or seen.add(i) for i in x)

    def allSame(self, vals):
        return all(x == vals[0] for x in vals)

    def stage_checker(self, stage_dict):
        '''
        Should be of format: 
        levels:
            level1:
                channels:
                    - [3, 64]     # input, inter
                    - [64, 64]    # inter, inter
                    - [64, 128]   # inter, output 
                module_type:
                    - bottleneck
                    - basic
                    - basic
            level2:
                channels:
                    - [3, 64]     # input, inter
                    - [64, 64]    # inter, inter
                    - [64, 128]   # inter, output 
                module_type:
                    - basic
                    - basic
                    - bottleneck 
        3 checks performed here:
        check 1: number of channel entries == number of modules (so that we can 
            set up nn.Conv, nn.etc properly )
        check 2: make sure number of output channels is the same for all levels 
            (output_chans)
        check 3: assert that we have a unique identifier for each number of 
            levels (dicitonaries shouldn't be able to have repeats but the way 
            we have this set up is that we can have a repeated level1, and it 
            only registers one level1 and ignores the first)
        check 4: module_types in each level are defined in ALLOWABLE_MODULE_TYPES
        The other checks worthy of note:
        check 4: number of intermediate channels is cohesive across each succ.
            module, should be caught when creating the nn.Conv/nn.Layer structs
        checks that need to be formed BY YOURSELF
        check 5: does the paramater num_levels = the number of levels written down 

        '''
        num_levels = len(stage_dict['levels'])
        assert num_levels == stage_dict['num_levels'], f'If youre getting an error here, its likely because you have level1 and then another level1 or a duplicate of some kind in your sublevels (leveli) stack '
        all_output_channels = list()
        all_input_channels = list()
        for i in range(num_levels):
            num_channel_pairs = len(stage_dict['levels']['level' + str(i+1)]['channels'])
            num_module_types = len(stage_dict['levels']['level' + str(i+1)]['module_type'])
            assert num_channel_pairs == num_module_types, \
                f'Check yaml file for index: {i+1} pair of channels and module type, need to be the same num of each'
            output_channel_for_level = stage_dict['levels']['level' + str(i+1)]['channels'][-1][-1]
            input_channel_for_level = stage_dict['levels']['level' + str(i+1)]['channels'][0][0]
            all_output_channels.append(output_channel_for_level)
            all_input_channels.append(input_channel_for_level)

            modules_for_level = stage_dict['levels']['level' + str(i+1)]['module_type']
            
            for j in range(len(modules_for_level)):
                assert modules_for_level[j] in ALLOWABLE_MODULE_TYPES, f'Level: `{"level" + str(i + 1)}`, module {str(j + 1)}, module type: `{modules_for_level[j]}` invalid. Allowable module types: {ALLOWABLE_MODULE_TYPES}'

        # check all output channel dims are the same
        assert self.allSame(all_output_channels), f'Output channels in yaml config are not same across levels'
        assert self.allSame(all_input_channels), f'Input channels in yaml config are not same across levels'

        # this shouldnt even work, its the the following check from above: 
        #       assert num_levels == stage_dict['num_levels'] 
        # that actually checks to make sure we have unique identifiers and the 
        # structure of the for loop that will also catch it if we've hit a number 
        # above the set number of levels, but this is here anyways i guess. 
        assert self.allUnique(list(stage_dict['levels'])) 

class HRT(nn.Module):
    def __init__(
        self,
        config,
    ):
        self.config = config 
        print(f'Model: {self._get_name()} initialized')

        self.stages = list()
        for i in range(len(self.config['stages'])):
            self.stages.append(Stage(self.config['stages']['stage_' + str(i+1)]))
    def forward(self, x):
        # this is obviously extremely incomplete. 
        for i in range(len(self.stages)):
            out_i = self.stages[i](x)

        out = torch.cat([out[0], out[1], out[2]])
        return out 
    
    def init_weight(self, x):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, 0, 0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)



if __name__ == '__main__':
    from torchsummary import summary
    
    config_path = 'seg/model/trans_high_res/stages.yaml'
    with open(config_path, "r") as stream:
        try:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    
    hrt = Stage(stage_dict=config['stage_1'])

    # for i in range(3):
    #     level_num = i + 1
    #     stride = level_num 
    #     print(f'(level_num, stride): {level_num, stride}')

    # test_bottleneck = Elementary(
    #     in_chans = 3, 
    #     out_chans = 64, 
    #     fuse_type='sum',
    # ).cuda()
    # summary(
    #     model = test_bottleneck.cuda(),
    #     input_size = [(3, 512, 512), (64, 512, 512)], 
    #     batch_size = 2,
    # )


    # print(len(config['stage_1']['levels']))
    # for i in range(len(config['stage_1']['levels'])):
    #     print(config['stage_1']['levels']['level'+ str(i+1)])

    # hrt = stage(
    #     stage_dict = stage1_dict,
    # )

    # from torchsummary import summary
    
    # summary(
    #     model = hrt.cuda(),
    #     input_size = (3, 512, 512),
    #     batch_size = 10,
    # )