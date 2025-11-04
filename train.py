import os
from options.train_options import TrainOptions
from data import create_dataset
import torch.optim as optim
from models.HibriMambaBlock import *
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms as T
import kornia.augmentation as K


if __name__ == '__main__':
    # device_ids = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES').split(',')]
    device_ids = [0,1,2]
    opt = TrainOptions().parse()

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    iters_perepoch = dataset_size // opt.batch_size
    print('The number of training images = %d' % dataset_size)
    single_epoch_iters = (dataset_size // opt.batch_size)
    total_iters = opt.total_epochs * single_epoch_iters 
    cur_iters = opt.resume_iter + opt.resume_epoch * single_epoch_iters
    start_iter = opt.resume_iter

    generator = HibriMambaBlock(embed_dim=32)
    generator = nn.DataParallel(generator, device_ids=device_ids)
    generator = generator.cuda(device=device_ids[0])

    generator.train()
    criterionL1 = nn.L1Loss()
    optimizer_G = optim.AdamW(generator.parameters(), lr=opt.lr, betas=(0.9, 0.99), weight_decay=0.02)
    scheduler_G = lr_scheduler.MultiStepLR(optimizer_G, milestones=[20, 40, 60, 80, 100], gamma=0.5)

    # continue train strategy
    if opt.continue_train:
        latest_ckpt = './ckpt/demo_model560000.pt'  
        if os.path.exists(latest_ckpt):
            print(f"Continuing training from {latest_ckpt}...")
            generator.load_state_dict(torch.load(latest_ckpt))
            
            opt.resume_epoch = int(latest_ckpt.split('demo_model')[-1].split('.pt')[0]) // iters_perepoch
            opt.resume_iter = int(latest_ckpt.split('demo_model')[-1].split('.pt')[0])
            print(f"Resuming from epoch {opt.resume_epoch}, iter {opt.resume_iter}")

            for _ in range(opt.resume_epoch):
                scheduler_G.step()          
    else:
        print("Warning: No latest checkpoint found! Starting from scratch.")
        opt.resume_epoch = 0
        opt.resume_iter  = 0
    cur_iters = opt.resume_iter + opt.resume_epoch * single_epoch_iters
    start_iter = opt.resume_iter

    print('Start training from epoch: {:05d}; iter: {:07d}'.format(opt.resume_epoch, opt.resume_iter))
    for epoch in range(opt.resume_epoch, opt.total_epochs + 1):
        for i, data in enumerate(dataset, start=start_iter):
            cur_iters += 1
            hr=data['HR'].cuda(device=device_ids[0], non_blocking=True)
            lr=data['LR'].cuda(device=device_ids[0], non_blocking=True)

            output = generator(lr)
            loss = criterionL1(hr,output)
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            if cur_iters % opt.print_freq == 0:
                print('===============')
                print("Iter:[%d | %d / %d]" %(epoch+1,cur_iters,iters_perepoch))
                print("Loss_Pix:%f"%(loss.item()))
            if cur_iters % opt.save_iter_freq == 0:
                print("saving ckpt")
                torch.save(generator.state_dict(),'./ckpt/demo_model%03d.pt'%cur_iters)
            if cur_iters % opt.save_latest_freq == 0:
                print("saving lastest ckpt")
                torch.save(generator.state_dict(),'./ckpt/latest_demo_model.pt')
        scheduler_G.step()
        lr_G = scheduler_G.get_lr()
        print("current learning rate is:")
        print(lr_G)
print("finish!!!")
