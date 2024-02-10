import sys
import os
import torch
import numpy as np
from skimage import transform
import SimpleITK as sitk
import argparse
sys.path.append('../')
from dataloader.ProstateDataset import *
from torch.utils.data import DataLoader
from networks1.resnet_MFFusion import Encoder2_MFFusion
from networks.resnet3d import resnet50
# import cv2
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--exp_name', type=str, default='ResNet_144*144*200_Elasto_E_aug_fold1')
parser.add_argument('--data_root_path', type=str,
                    default='/hy-tmp/datasets/144%144%200ROINPZ')
parser.add_argument('--save_cam_path', type=str, default='/hy-tmp/code/ProstateSeg_yun/video/inference_ori')
parser.add_argument('--model_path', type=str, default='/hy-tmp/code/ProstateSeg_yun/video/model1104')
parser.add_argument('--infer_dir', type=str, default='layer3')
args = parser.parse_args()


class GradCAM:
    def __init__(self, model: torch.nn.Module, cam_size):
        self.model = model
        self.model.eval()
        getattr(self.model, 'layer3').register_forward_hook(self.__forward_hook)
        getattr(self.model, 'layer3').register_backward_hook(self.__backward_hook)

        self.num_cls = 2
        self.size = cam_size
        # self.size = [512, 490, 200]
        self.grads = []
        self.fmaps = []

    def __backward_hook(self, module, grad_in, grad_out):
        self.grads.append(grad_out[0].detach())

    def __forward_hook(self, module, input, output):
        self.fmaps.append(output)

    def __compute_loss(self, logit, index):
        BCE = torch.nn.CrossEntropyLoss()
        label = torch.LongTensor(index).to(logit.device)
        loss = BCE(logit, label)
        return loss

    def forward(self, img_arr, label):
        img_input = torch.Tensor(np.expand_dims(img_arr, axis=0).copy()).cuda()

        # forward
        output = self.model(img_input)

        # backward
        self.model.zero_grad()
        loss = self.__compute_loss(output, label)
        loss.backward()

        # generate CAM
        grads_val = self.grads[0].cpu().data.numpy().squeeze()
        fmap = self.fmaps[0].cpu().data.numpy().squeeze()
        cam = self.__compute_cam(fmap, grads_val)

        self.fmaps.clear()
        self.grads.clear()
        return cam

    def __compute_cam(self, feature_map, grads):

        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        alpha = np.mean(grads, axis=(1, 2, 3))  # GAP
        for k, ak in enumerate(alpha):
            cam += ak * feature_map[k]  # linear combination
        cam = np.maximum(cam, 0)  # relu
        cam = transform.resize(cam, self.size, order=3, preserve_range=True)
        cam = (cam - np.min(cam)) / np.max(cam)
        return cam


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    save_path = os.path.join(args.save_cam_path, args.exp_name, 'fold612', args.infer_dir)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path, exist_ok=True)

    net = resnet50(num_channels=3, num_classes=2).cuda()
    net.load_state_dict(torch.load('{}/{}/ckp_model/model_80.pth'.format(args.model_path, args.exp_name)))
    # case_id_list = os.listdir(args.data_root_path)
    # case_id_list = [i.split('.')[0] for i in case_id_list]
    case_id_list = ['021', '023', '060', '084', '090', '091', '117', '213', '215', '242', '247', '248',
                    '251', '030', '032', '043', '108', '109', '140', '141']
    z_num_list = [1105, 571, 954, 813, 822, 507, 514, 574, 662, 682, 708, 483,
                  394, 640, 935, 715, 779, 589, 686, 806]

    cam_dataset = PD3C(case_id_list, args.data_root_path, 1)
    cam_dataloader = DataLoader(cam_dataset, batch_size=None, shuffle=False)
    import cv2
    for i_batch, batch in enumerate(cam_dataloader):
        sample = batch['volume']
        label = batch['cspca']
        case_id = batch['name']
        print("case id: {}   label: {}".format(case_id, label))
        grad_cam = GradCAM(net, [512, 490, z_num_list[i_batch]])
        cam = grad_cam.forward(sample, label=np.ones(1))
        # c x y z -> z y x
        # sample = sample[0, ...].numpy().transpose(2, 1, 0)
        sample = sample[:, ...].numpy().transpose(3, 2, 1, 0) * 255
        # sample = sample[..., 0].numpy().transpose(2, 1, 0)
        cam = cam.transpose(2, 1, 0) * 255
        # cam = np.expand_dims(cam[..., 0], axis=2).repeat(3, 2).transpose(1, 0, 2)
        save_case_path = os.path.join(save_path, case_id)
        os.makedirs(save_case_path, exist_ok=True)
        for z_i in range(z_num_list[i_batch]):
            rgb_cam = cv2.cvtColor(cam[z_i], cv2.COLOR_BGR2RGB)
            # rgb_cam = cv2.applyColorMap(rgb_cam.astype(np.uint8), cv2.COLORMAP_JET)
            # sample_cam = sample[z_i, ...] + rgb_cam
            # alpha = 0.6  # 可以调整透明度
            # overlay_image = cv2.addWeighted(sample[z_i, ...].astype(np.uint8), alpha, rgb_cam, 1 - alpha, 0)
            cv2.imwrite('{}/{}.jpg'.format(save_case_path, z_i), cam[z_i])

        # plt.imshow(cam)
        # plt.axis('off')
        # plt.show()
        # cam_sample = sitk.GetImageFromArray(cam*255 + sample*255)
        # sitk.WriteImage(cam_sample, os.path.join('{}/{}_cam_img.nii.gz'.format(save_path, case_id)))
        # cam = (cam * 255)
        # cam = sitk.GetImageFromArray(cam)
        # sitk.WriteImage(cam, os.path.join('{}/{}_cam.nii.gz'.format(save_path, case_id)))
        #
        # sample = sitk.GetImageFromArray(sample)
        # sitk.WriteImage(sample, os.path.join('{}/{}_img.nii.gz'.format(save_path, case_id)))

