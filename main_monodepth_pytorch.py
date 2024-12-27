import argparse
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.optim as optim

# custom modules

from loss import MonodepthLoss
from utils import get_model, to_device, prepare_dataloader
from log_output import tflog2pandas

from easydict import EasyDict as edict

# plot params

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15, 10)


def return_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Monodepth')

    parser.add_argument('data_dir',
                        help='path to the dataset folder. \
                        It should contain subfolders with following structure:\
                        "image_02/data" for left images and \
                        "image_03/data" for right images'
                        )
    parser.add_argument('val_data_dir',
                        help='path to the validation dataset folder. \
                            It should contain subfolders with following structure:\
                            "image_02/data" for left images and \
                            "image_03/data" for right images'
                        )
    parser.add_argument('model_path', help='path to the trained model')
    parser.add_argument('output_directory',
                        help='where save dispairities\
                        for tested images'
                        )
    parser.add_argument('--input_height', type=int, help='input height',
                        default=256)
    parser.add_argument('--input_width', type=int, help='input width',
                        default=512)
    parser.add_argument('--model', default='resnet18_md',
                        help='encoder architecture: ' +
                        'resnet18_md or resnet50_md ' + '(default: resnet18)'
                        + 'or torchvision version of any resnet model'
                        )
    parser.add_argument('--pretrained', default=False,
                        help='Use weights of pretrained model'
                        )
    parser.add_argument('--mode', default='train',
                        help='mode: train or test (default: train)')
    parser.add_argument('--epochs', default=50,
                        help='number of total epochs to run')
    parser.add_argument('--learning_rate', default=1e-4,
                        help='initial learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', default=256,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--adjust_lr', default=True,
                        help='apply learning rate decay or not\
                        (default: True)'
                        )
    parser.add_argument('--device',
                        default='cuda:0',
                        help='choose cpu or cuda:0 device"'
                        )
    parser.add_argument('--do_augmentation', default=True,
                        help='do augmentation of images or not')
    parser.add_argument('--augment_parameters', default=[
        0.8,
        1.2,
        0.5,
        2.0,
        0.8,
        1.2,
        ],
            help='lowest and highest values for gamma,\
                        brightness and color respectively'
            )
    parser.add_argument('--print_images', default=False,
                        help='print disparity and image\
                        generated from disparity on every iteration'
                        )
    parser.add_argument('--print_weights', default=False,
                        help='print weights of every layer')
    parser.add_argument('--input_channels', default=3,
                        help='Number of channels in input tensor')
    parser.add_argument('--num_workers', default=4,
                        help='Number of workers in dataloader')
    parser.add_argument('--use_multiple_gpu', default=False)
    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR\
        decayed by 2 every 10 epochs after 30 epoches"""

    if epoch >= 30 and epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


class Model:

    def __init__(self, args):
        self.args = args
        ## tensorboard #pu
        self.logger = SummaryWriter(args.output_directory)

        # Set up model
        self.device = args.device
        self.model = get_model(args.model, input_channels=args.input_channels, pretrained=args.pretrained)
        print('torch.cuda.device_count(): ',torch.cuda.device_count())#pu
        if args.use_multiple_gpu:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'#pu
            print('self.device',self.device)
            self.model = torch.nn.DataParallel(self.model)
        
        self.model = self.model.to(self.device)
        
        if args.mode == 'train':
            self.loss_function = MonodepthLoss(
                n=4,
                SSIM_w=0.85,
                disp_gradient_w=0.1, lr_w=1).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=args.learning_rate)
            print('Getting Val_loader... ')
            self.val_n_img, self.val_loader = prepare_dataloader(args.val_data_dir, args.mode,
                                                                 args.augment_parameters,
                                                                 False, args.batch_size,
                                                                 (args.input_height, args.input_width),
                                                                 args.num_workers)
        else:
            self.model.load_state_dict(torch.load(args.model_path))
            args.augment_parameters = None
            args.do_augmentation = False
            args.batch_size = 1

        # Load data
        self.output_directory = args.output_directory
        self.input_height = args.input_height
        self.input_width = args.input_width
        print('Getting Loader... ')
        self.n_img, self.loader = prepare_dataloader(args.data_dir, args.mode, args.augment_parameters,
                                                     args.do_augmentation, args.batch_size,
                                                     (args.input_height, args.input_width),
                                                     args.num_workers)


        if 'cuda' in self.device:
            torch.cuda.synchronize()


    def train(self):
        losses = []
        val_losses = []
        best_loss = float('Inf')
        best_val_loss = float('Inf')

        running_val_loss = 0.0
        self.model.eval()
        for data in self.val_loader:
            data = to_device(data, self.device)
            left = data['left_image']
            right = data['right_image']
            disps = self.model(left)
            loss = self.loss_function(disps, [left, right])
            val_losses.append(loss.item())
            running_val_loss += loss.item()

        running_val_loss /= self.val_n_img / self.args.batch_size
        print('Val_loss:', running_val_loss)
        self.logger.add_scalar('val_loss_before_training', running_val_loss, 0)

        for epoch in range(self.args.epochs):
            if self.args.adjust_lr:
                adjust_learning_rate(self.optimizer, epoch,
                                     self.args.learning_rate)
            c_time = time.time()
            running_loss = 0.0
            self.model.train()
            for data in self.loader:
                # Load data
                data = to_device(data, self.device)
                left = data['left_image'] # b,c,h,w
                right = data['right_image']

                # One optimization iteration
                self.optimizer.zero_grad()
                disps = self.model(left)
                loss = self.loss_function(disps, [left, right])
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

                # Print statistics
                if self.args.print_weights:
                    # print(self.model.named_parameters)
                    import math
                    p=0
                    for (name, parameter) in self.model.named_parameters():
                        if name.split(sep='.')[-1] == 'weight':
                            p+=1
                    # print(p, 'and',math.ceil(p/5))
                    j = 1
                    
                    plt.figure()
                    for (name, parameter) in self.model.named_parameters():
                        # print('name:',name)
                        
                        if name.split(sep='.')[-1] == 'weight':##
                            #plt.subplot(5, 9, j)
                            plt.subplot(5,math.ceil(p/5),j)#pu
                            plt.hist(parameter.data.cpu().view(-1))#pu
                            plt.xlim([-1, 1])
                            plt.title(name.split(sep='.')[0])
                            j += 1
                    #plt.show()
                    plt.savefig('print_weights.jpg')

                if self.args.print_images:
                    print('disp_left_est[0]')
                    plt.figure()
                    plt.imshow(np.squeeze(
                        np.transpose(self.loss_function.disp_left_est[0][0,
                                     :, :, :].cpu().detach().numpy(),
                                     (1, 2, 0))))
                    #plt.show()
                    plt.savefig('disp_left_est[0].jpg')
                    print('left_est[0]')
                    plt.figure()
                    plt.imshow(np.transpose(self.loss_function\
                        .left_est[0][0, :, :, :].cpu().detach().numpy(),
                        (1, 2, 0)))
                    #plt.show()
                    plt.savefig('left_est[0].jpg')
                    print('disp_right_est[0]')
                    plt.figure()
                    plt.imshow(np.squeeze(
                        np.transpose(self.loss_function.disp_right_est[0][0,
                                     :, :, :].cpu().detach().numpy(),
                                     (1, 2, 0))))
                    #plt.show()
                    plt.savefig('disp_right_est[0].jpg')
                    print('right_est[0]')
                    plt.figure()
                    plt.imshow(np.transpose(self.loss_function.right_est[0][0,
                               :, :, :].cpu().detach().numpy(), (1, 2,
                               0)))
                    #plt.show()
                    plt.savefig('right_est[0].jpg')
                running_loss += loss.item()

            running_val_loss = 0.0
            self.model.eval()
            for data in self.val_loader:
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']
                disps = self.model(left)
                loss = self.loss_function(disps, [left, right])
                val_losses.append(loss.item())
                running_val_loss += loss.item()

            # Estimate loss per image
            running_loss /= self.n_img / self.args.batch_size
            running_val_loss /= self.val_n_img / self.args.batch_size
            print (
                'Epoch:',
                epoch + 1,
                'train_loss:',
                running_loss,
                'val_loss:',
                running_val_loss,
                'time:',
                round(time.time() - c_time, 3),
                's',
                )
            self.logger.add_scalar('train_loss', running_loss, epoch+ 1)
            self.logger.add_scalar('val_loss', running_val_loss, epoch+ 1)
            self.logger.add_scalar('time(s)', round(time.time() - c_time, 3), epoch+ 1)

            self.save(self.args.model_path[:-4] + '_last.pth')
            if running_val_loss < best_val_loss:
                self.save(self.args.model_path[:-4] + '_cpt.pth')
                best_val_loss = running_val_loss
                print('Model_saved')
        self.logger.add_scalar('best_val_loss', best_val_loss, 0)
        print ('Finished Training. Best loss:', best_loss)
        tflog2pandas(self.output_directory)
        self.save(self.args.model_path)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def test(self):
        self.model.eval()
        disparities = np.zeros((self.n_img,
                               self.input_height, self.input_width),
                               dtype=np.float32)
        disparities_pp = np.zeros((self.n_img,
                                  self.input_height, self.input_width),
                                  dtype=np.float32)
        with torch.no_grad():
            for (i, data) in enumerate(self.loader):
                # Get the inputs
                data = to_device(data, self.device)
                left = data.squeeze()
                # Do a forward pass
                disps = self.model(left)
                disp = disps[0][:, 0, :, :].unsqueeze(1)
                disparities[i] = disp[0].squeeze().cpu().numpy()
                disparities_pp[i] = \
                    post_process_disparity(disps[0][:, 0, :, :]\
                                           .cpu().numpy())
                if i%10==0:
                    np.save(self.output_directory + '/disparities_%i.npy'%i, disparities[i])
                    np.save(self.output_directory + '/disparities_pp_%i.npy'%i,
                            disparities_pp[i])
                    np.save(self.output_directory + '/original_%i.npy'%i,
                            data.cpu().numpy())

        print('Finished Testing')


def main(args):
    args = return_arguments()
    if args.mode == 'train':
        model = Model(args)
        model.train()
    elif args.mode == 'test':
        model_test = Model(args)
        model_test.test()


if __name__ == '__main__':
    # main() ##pu

    dict_parameters = edict({'data_dir':'data/eye_original_frames_fps2_final/train/',#'data/kitti/train/'
                         'val_data_dir':'data/eye_original_frames_fps2_final/val/',
                         'model_path':'data/models/monodepth1_eyefp2_resnet18_20240607.pth',
                         'output_directory':'data/output/',
                         'input_height':256,
                         'input_width':512,
                         'model':'resnet18_md',#resnet50_md
                         'pretrained':False,#True
                         'mode':'train',
                         'epochs':500,#200s
                         'learning_rate':1e-4,
                         'batch_size': 32,#default 256#8
                         'adjust_lr':True,
                         'device':'cuda:0',
                         'do_augmentation':True,
                         'augment_parameters':[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                         'print_images':False,
                         'print_weights':False,
                         'input_channels': 3,
                         'num_workers': 4,#8
                         'use_multiple_gpu':True })#False

    dict_parameters_test = edict({'data_dir':'data/eye_original_frames_fps2_final/val',
                              'model_path':'data/models/monodepth1_eyefp2_resnet18_20240607_cpt.pth',
                              'output_directory':'data/output/',
                              'input_height':256,
                              'input_width':512,
                              'model':'resnet18_md',
                              'pretrained':False,
                              'mode':'test',
                              'device':'cuda:0',
                              'input_channels':3,
                              'num_workers':4,
                              'use_multiple_gpu':True})  
    # model = Model(dict_parameters)
    # model.train() 
    model_test = Model(dict_parameters_test)
    model_test.test()             

    # if dict_parameters.mode == 'train':
    #     model = Model(dict_parameters)
    #     model.train()
    # elif dict_parameters.mode == 'test':
    #     