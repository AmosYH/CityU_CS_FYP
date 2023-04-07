# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch

# from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
import yaml

def read_and_modify_one_block_of_yaml_data(filename, key, value):
    with open(f'{filename}.yml', 'r') as f:
        data = yaml.safe_load(f)
        data['img_path'][f'{key}'] = f'{value}' 
    with open(f'{filename}.yml', 'w') as f:
        yaml.dump(data, f, sort_keys=False)


for i in range(0, 100):

    opt = parse_options(is_train=False)

    img_path = opt['img_path'].get('input_img')
    output_path = opt['img_path'].get('output_img')


    ## 1. read image
    file_client = FileClient('disk')

    img_bytes = file_client.get(img_path, None)
    try:
        img = imfrombytes(img_bytes, float32=True)
    except:
        raise Exception("path {} not working".format(img_path))

    img = img2tensor(img, bgr2rgb=True, float32=True)



    ## 2. run inference
    model = create_model(opt)
    model.single_image_inference(img, output_path)

    print('inference {} .. finished.'.format(img_path))

    read_and_modify_one_block_of_yaml_data('options/demo/demo', key='input_img', value='./demo/DCE+++SCI/' + str(i + 1) + '.png')
    read_and_modify_one_block_of_yaml_data('options/demo/demo', key='output_img', value='./demo/Denoised/' + str(i + 1) + '.png')