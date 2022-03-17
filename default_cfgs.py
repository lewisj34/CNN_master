import yaml
from pathlib import Path

def get_cfgs(
    dataset='kvasir',
    model_name='trans', # defunct rn 
    backbone='vit_base_patch16_384',
    decoder='linear',
    cnn_model_name='unet',
    cnn_backbone='resnet18',
    image_height=256,
    image_width=256,
    dropout=0.0,
    drop_path_rate=0.1,
    n_cls=1,
    save_dir='seg/data',
    checkpt_save_dir='seg/current_checkpoints/',
    num_epochs=5,
    learning_rate=7e-5,
    batch_size=16,
    model_checkpoint_name='Trans_CNN',
    speed_test=False,
):
    '''
    imports the necessary cfgs that are gathered in main() in train.py, 
    just makes it more portable tho 
    returns:
        trans_model_cfg, cnn_model_cfg
    '''
    # load in cfgs 
    trans_cfg = yaml.load(open(Path(__file__).parent / "vit_config.yml", "r"), 
        Loader=yaml.FullLoader)
    cnn_cfg = yaml.load(open(Path(__file__).parent / "cnn_config.yml", "r"), 
        Loader=yaml.FullLoader)

    # transformer branch 
    trans_model_cfg = trans_cfg['model'][backbone]
    trans_model_cfg['image_size'] = (image_height, image_width)
    trans_model_cfg["dropout"] = dropout
    trans_model_cfg["drop_path_rate"] = drop_path_rate
    trans_model_cfg['backbone'] = backbone 
    trans_model_cfg['n_cls'] = n_cls
    decoder_cfg = trans_cfg["decoder"][decoder]
    decoder_cfg['name'] = decoder 
    trans_model_cfg['decoder'] = decoder_cfg

    patch_size = trans_model_cfg['patch_size']

    assert trans_model_cfg['patch_size'] == 16 \
        or trans_model_cfg['patch_size'] == 32, 'Patch size == {16, 32}'
    # cnn branch 
    cnn_model_cfg = cnn_cfg['model'][cnn_model_name]
    cnn_model_cfg['image_size'] = (image_height, image_width)  
    cnn_model_cfg['patch_size'] = patch_size
    cnn_model_cfg['batch_size'] = batch_size
    cnn_model_cfg['num_classes'] = n_cls
    cnn_model_cfg['in_channels'] = 3 # this isn't anywhere in transformer but 
    cnn_model_cfg['backbone'] = cnn_backbone

    # printing for no reason 
    # print(f'cnn_model_cfg:\n {cnn_model_cfg}')
    # print(f'trans_model_cfg:\n {trans_model_cfg}')
    # print(f'decoder_cfg:\n {decoder_cfg}')

    return trans_model_cfg, cnn_model_cfg