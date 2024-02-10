import datetime
import argparse
import time
import logging
import sys
import shutil
from utils.data_preparation import get_train_val_test_balance
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from dataloader.ProstateDataset import *
from dataloader.data_aug import *
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from validation import val
from networks1.resnet_MFFusion5 import Encoder2_MFFusion5
from networks1.resnet_MFFusion import Encoder2_MFFusion


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default='Encoder2_MFFusion5_orloss_144*144*200_Elasto_B_E_aug_fold1')
# parser.add_argument("--exp_name", type=str, default='debug')
parser.add_argument("--gpu", type=str, default='0, 1')
parser.add_argument("--fold", type=str, default='BMode612')
parser.add_argument("--fold_path", type=str, default='utils')
parser.add_argument("--data_path", type=str,
                    default='/hy-tmp/datasets/144%144%200ROINPZ')

parser.add_argument("--image_id", type=int, default=2)
parser.add_argument("--write_image", type=bool, default=True)
parser.add_argument('--num_channels', type=int, default=3)
parser.add_argument("--save_root_path", type=str, default='../model1121')
parser.add_argument("--max_epoch", type=int, default=1000)
parser.add_argument("--batchsize_positive", type=int, default=1)
parser.add_argument("--batchsize_negative", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--val_batchsize", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--seed1", type=int, default=1997)
parser.add_argument("--per_val_epoch", type=int, default=2)
parser.add_argument("--per_save_model", type=int, default=5)
parser.add_argument("--num_classes", type=int, default=2)
args = parser.parse_args()


def create_model():
    model = Encoder2_MFFusion5(num_channels=args.num_channels, num_classes=args.num_classes)
    model = nn.DataParallel(model)
    return model.cuda()


def save_parameter(exp_save_path, d=False):
    delete = True if os.path.basename(exp_save_path) == 'debug' else d
    if os.path.exists(exp_save_path) is True:
        assert delete is True
        shutil.rmtree(exp_save_path)
    os.makedirs(exp_save_path)
    os.makedirs(os.path.join(exp_save_path, 'ckp_model'))

    parameter_dict = {"fold": args.fold,
                      "data_root_path": args.data_path,
                      "gpu": args.gpu,
                      "seed": args.seed1,
                      "positive sample batch size": args.batchsize_positive,
                      "negative sample batch size": args.batchsize_negative,
                      "lr": args.lr,
                      "save_path": exp_save_path}
    with open(os.path.join(exp_save_path, 'parameter_log.txt'), mode='a', encoding='utf-8') as f:
        for key, value in parameter_dict.items():
            f.write(str(key) + ':' + str(value) + '\n')

    # save this .py
    py_path_old = sys.argv[0]
    py_path_new = os.path.join(exp_save_path, os.path.basename(py_path_old))
    shutil.copy(py_path_old, py_path_new)
    logging.basicConfig(filename=os.path.join(exp_save_path, 'log.txt'), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(parameter_dict)


def reproduce(seed1):
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(seed1)
    np.random.seed(seed1)
    torch.manual_seed(seed1)
    torch.cuda.manual_seed(seed1)


def train():
    reproduce(args.seed1)
    exp_time = time.localtime()
    exp_time_format = time.strftime("%m-%d-%H-%M", exp_time)
    exp_save_path = os.path.join(args.save_root_path, '{}'.format(args.exp_name))
    save_parameter(exp_save_path)
    writer = SummaryWriter(log_dir=exp_save_path)

    print('-------------------------------------- setting --------------------------------------')
    print("experiment name: {}".format(os.path.basename(exp_save_path)))
    print("time: ", exp_time_format)
    print('data name: {}'.format(os.path.basename(args.data_path)))
    print("fold: {}".format(args.fold))
    print("gpu: {}".format(args.gpu))
    print("save path: {}".format(exp_save_path))
    print('-------------------------------------- setting --------------------------------------')

    # load data
    pos_list, neg_list = get_train_val_test_balance(args.fold_path, args.fold)
    train_dataset_positive = PD6C(pos_list[0], args.data_path, args.image_id,
                                  transform=transforms.Compose([
                                    # SparseZSliceGauss(150),
                                    RandomRotateTransform(angle_range=(-10, 10), p_per_sample=0.2),
                                    MirrorTransform(axes=(-3, -2, -1)),
                                    ToTensor()]))
    train_dataset_negative = PD6C(neg_list[0], args.data_path, args.image_id,
                                  transform=transforms.Compose([
                                    # SparseZSliceGauss(150),
                                    RandomRotateTransform(angle_range=(-10, 10), p_per_sample=0.2),
                                    MirrorTransform(axes=(-3, -2, -1)),
                                    ToTensor()]))
    val_dataset = PD6C(pos_list[1]+neg_list[1], args.data_path, args.image_id,
                    transform=transforms.Compose([
                                                  # SparseZSliceGauss(150),
                                                  ToTensor()]))

    def worker_init_fn(worker_id):
        random.seed(args.seed1 + worker_id)

    train_dataloader_positive = DataLoader(train_dataset_positive,
                                           batch_size=args.batchsize_positive,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           pin_memory=False,
                                           worker_init_fn=worker_init_fn)

    train_dataloader_negative = DataLoader(train_dataset_negative,
                                           batch_size=args.batchsize_negative,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           pin_memory=False,
                                           worker_init_fn=worker_init_fn)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val_batchsize,
                                shuffle=True)

    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    celoss = nn.CrossEntropyLoss()
    # celoss = FocalLoss(alpha=0.67, gamma=2)

    def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
        return initial_lr * (1 - epoch / max_epochs) ** exponent

    n_total_iter = 0
    best_auc = 0

    for epoch in range(args.max_epoch):
        lr = poly_lr(epoch, args.max_epoch, args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        loss_epoch = 0.0
        or_loss_epoch = 0.0
        i_batch = 0

        train_prefetcher_pos = data_prefetcher(train_dataloader_positive)
        train_prefetcher_neg = data_prefetcher(train_dataloader_negative)
        pos_batch = train_prefetcher_pos.next()
        neg_batch = train_prefetcher_neg.next()

        y_true = []
        y_pred = []
        y_pred_sm = []

        model.train()
        while pos_batch is not None and neg_batch is not None:

            start_time = time.time()
            pos_volume, neg_volume = pos_batch['volume'], neg_batch['volume']
            pos_label, neg_label = pos_batch['cspca'], neg_batch['cspca']
            train_case = pos_batch['name'] + neg_batch['name']
            train_volume = torch.cat([pos_volume, neg_volume], dim=0).cuda().float()
            train_label = torch.cat([pos_label, neg_label], dim=0).cuda()
            optimizer.zero_grad()
            out = model(train_volume)
            out_sm = F.softmax(out, dim=1)
            y_true.extend(train_label)
            y_pred.extend(torch.max(out, 1)[1])
            y_pred_sm.extend(out_sm[:, 1])
            # so loss
            reg = 1e-6
            orth_loss = torch.zeros(1).cuda()
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    param_flat = param.view(param.shape[0], -1)
                    sym = torch.mm(param_flat, torch.t(param_flat))
                    sym -= torch.eye(param_flat.shape[0]).to(param.device)
                    orth_loss = orth_loss + (reg * sym.abs().sum())

            ce_loss = celoss(out, train_label)
            loss = ce_loss + orth_loss

            loss_epoch += loss
            or_loss_epoch += orth_loss
            loss.backward()
            optimizer.step()

            # write image
            if n_total_iter == 0 and args.write_image:
                if train_volume.size(1) == 3:
                    img1 = train_volume[0, :, :, :, ::2].permute(3, 0, 2, 1)
                else:
                    img1 = train_volume[0, 3:, :, :, ::2].permute(3, 0, 2, 1)
                grid_image = make_grid(img1, 8, normalize=False)
                writer.add_image('PreviewImage_positive_{}'.format(train_case[0]), grid_image, n_total_iter)
                if train_volume.size(1) == 3:
                    img2 = train_volume[-1, :, :, :, ::2].permute(3, 0, 2, 1)
                else:
                    img2 = train_volume[-1, 3:, :, :, ::2].permute(3, 0, 2, 1)
                grid_image = make_grid(img2, 8, normalize=False)
                writer.add_image('PreviewImage_negative_{}'.format(train_case[-1]), grid_image, n_total_iter)
            pos_batch = train_prefetcher_pos.next()
            neg_batch = train_prefetcher_neg.next()
            n_total_iter += 1
            end_time = time.time()
            used_time = datetime.timedelta(seconds=(end_time-start_time)).seconds
            logging.info("[Epoch: %4d/%d] [Train index: %2d/%d] [loss: %f] [used time: %ss]"
                         % (epoch, args.max_epoch, i_batch + 1, len(train_dataloader_negative),
                             loss.item(), used_time))
            # logging.info("case id: {}   label: {}".format(train_case, train_label.cpu()))
            i_batch += 1

        y_true = torch.stack(y_true, dim=0)
        # y_pred = torch.stack(y_pred, dim=0)
        y_pred_sm = torch.stack(y_pred_sm, dim=0)
        # train_acc = (y_pred == y_true).sum() / y_pred.size(0)
        fpr, tpr, thresholds_roc = roc_curve(y_true.cpu().data.numpy(), y_pred_sm.cpu().data.numpy(), pos_label=1)
        train_auc = auc(fpr, tpr)

        writer.add_scalar("Loss/loss", loss_epoch.item()/len(train_dataloader_negative), global_step=epoch)
        writer.add_scalar("Loss/loss", or_loss_epoch.item()/len(train_dataloader_negative), global_step=epoch)
        writer.add_scalar("Loss/lr", lr, global_step=epoch)
        writer.add_scalar("train/auc", train_auc, epoch)
        logging.info("epoch: {}  train auc: {}".format(epoch, train_auc))

        if epoch % args.per_val_epoch == 0:
            best_auc = val(model, val_dataloader, writer, epoch, exp_save_path, best_auc)

        if epoch % args.per_save_model == 0:
            torch.save(model.module.state_dict(), '{}/ckp_model/model_{}.pth'.format(exp_save_path, epoch))

    writer.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    train()