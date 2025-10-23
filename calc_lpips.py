import lpips
from PIL import Image
import numpy as np
import torch
import os
import sys

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

img_path1 = sys.argv[1]
img_path2 = sys.argv[2]

img_names = [x for x in os.listdir(img_path1)]
sum = []
for img_name in img_names:
    path1 = os.path.join(img_path1, img_name)
    path2 = os.path.join(img_path2, img_name)
    # .replace('_face00.', '.')
    # .replace('jpg', 'png')
    hr_img1 = (torch.from_numpy(np.array(Image.open(path1).convert('RGB'))))
    hr_img2 = (torch.from_numpy(np.array(Image.open(path2).convert('RGB'))))
    hr_img1 = ((hr_img1/127.5)-1).permute(2,0,1)
    hr_img2 = ((hr_img2/127.5)-1).permute(2,0,1)
    d = loss_fn_alex(hr_img1, hr_img2)
    # print(img_name, d.detach().numpy())
    sum.append(d.detach().numpy())
print('lpips:', np.mean(sum))

