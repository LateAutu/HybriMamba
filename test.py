import os
from options.test_options import TestOptions
from data import create_dataset
from utils import utils
from PIL import Image
from models.HibriMambaBlock import *
import torch
from psnr_ssim import psnr_ssim_dir

if __name__ == '__main__':
    device_ids = [0]
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    if len(opt.save_as_dir):
        save_dir = opt.save_as_dir
    else:
        save_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
        if opt.load_iter > 0:  # load_iter is 0 by default
            save_dir = '{:s}_iter{:d}'.format(save_dir, opt.load_iter)
    os.makedirs(save_dir, exist_ok=True)
    network = HibriMambaBlock(embed_dim=32)
    network = nn.DataParallel(network, device_ids=device_ids)
    model_path = opt.pretrain_model_path

    state_dict = torch.load(model_path)
    # state_dict = torch.load(model_path, map_location='cuda:7')
    network.load_state_dict(state_dict)
    for data in dataset:
        inp = data['LR']
        with torch.no_grad():
            output_SR = network(inp)
        img_path = data['LR_paths']
        output_sr_img = utils.tensor_to_img(output_SR, normal=True)

        save_path = os.path.join(save_dir, img_path[0].split('/')[-1])
        save_img = Image.fromarray(output_sr_img)
        save_img.save(save_path)
