import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import logging
import torch.nn.functional as F


def val(model, val_dataloader, writer, epoch, exp_save_path, best_auc):
    model.eval()
    val_y_true = []
    # val_y_pred = []
    val_y_pred_sm = []
    with torch.no_grad():
        for val_batch in tqdm(val_dataloader):
            val_volume = val_batch['volume'].float().cuda()
            val_label = val_batch['cspca'].cuda()
            val_name = val_batch['name']
            y = model(val_volume)
            y_sm = F.softmax(y, dim=1)
            val_y_true.extend(val_label)
            # val_y_pred.extend(torch.max(y, 1)[1])
            val_y_pred_sm.extend(y_sm[:, 1])
            logging.info("case id: {}   label: {}   pred: {}".format(val_name, val_label.cpu(), torch.max(y, 1)[1].cpu()))

        val_y_true = torch.stack(val_y_true, dim=0)
        # val_y_pred = torch.stack(val_y_pred, dim=0)
        val_y_pred_sm = torch.stack(val_y_pred_sm, dim=0)
        # val_acc = (val_y_pred == val_y_true).sum() / val_y_true.size(0)
        fpr, tpr, thresholds_roc = roc_curve(val_y_true.cpu().data.numpy(), val_y_pred_sm.cpu().data.numpy(), pos_label=1)
        val_auc = auc(fpr, tpr)
        logging.info("evaluation result: AUC=%4f" % (val_auc))
        writer.add_scalar("val/auc", val_auc, global_step=epoch)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.module.state_dict(), '{}/ckp_model/model_best.pth'.format(exp_save_path))
        writer.add_scalar("val/best_model_result_auc", best_auc, global_step=epoch)
        return best_auc


def val_multioutput(model, val_dataloader, writer, epoch, exp_save_path, best_auc):
    model.eval()
    val_y_true = []
    # val_y_pred = []
    val_y_pred_sm = []
    with torch.no_grad():
        for val_batch in tqdm(val_dataloader):
            val_volume = val_batch['volume'].float().cuda()
            val_label = val_batch['cspca'].cuda()
            val_name = val_batch['name']
            y, _, _ = model(val_volume)
            y_sm = F.softmax(y, dim=1)
            val_y_true.extend(val_label)
            # val_y_pred.extend(torch.max(y, 1)[1])
            val_y_pred_sm.extend(y_sm[:, 1])
            logging.info("case id: {}   label: {}   pred: {}".format(val_name, val_label.cpu(), torch.max(y, 1)[1].cpu()))

        val_y_true = torch.stack(val_y_true, dim=0)
        # val_y_pred = torch.stack(val_y_pred, dim=0)
        val_y_pred_sm = torch.stack(val_y_pred_sm, dim=0)
        # val_acc = (val_y_pred == val_y_true).sum() / val_y_true.size(0)
        fpr, tpr, thresholds_roc = roc_curve(val_y_true.cpu().data.numpy(), val_y_pred_sm.cpu().data.numpy(), pos_label=1)
        val_auc = auc(fpr, tpr)
        logging.info("evaluation result: AUC=%4f" % (val_auc))
        writer.add_scalar("val/auc", val_auc, global_step=epoch)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.module.state_dict(), '{}/ckp_model/model_best.pth'.format(exp_save_path))
        writer.add_scalar("val/best_model_result_auc", best_auc, global_step=epoch)
        return best_auc


def val_camloss(model, val_dataloader, writer, epoch, exp_save_path, best_auc):
    model.eval()
    val_y_true = []
    # val_y_pred = []
    val_y_pred_sm = []
    with torch.no_grad():
        for val_batch in tqdm(val_dataloader):
            val_volume = val_batch['volume'].float().cuda()
            val_label = val_batch['cspca'].cuda()
            val_name = val_batch['name']
            y, _ = model(val_volume)
            y_sm = F.softmax(y, dim=1)
            val_y_true.extend(val_label)
            # val_y_pred.extend(torch.max(y, 1)[1])
            val_y_pred_sm.extend(y_sm[:, 1])
            logging.info("case id: {}   label: {}   pred: {}".format(val_name, val_label.cpu(), torch.max(y, 1)[1].cpu()))

        val_y_true = torch.stack(val_y_true, dim=0)
        # val_y_pred = torch.stack(val_y_pred, dim=0)
        val_y_pred_sm = torch.stack(val_y_pred_sm, dim=0)
        # val_acc = (val_y_pred == val_y_true).sum() / val_y_true.size(0)
        fpr, tpr, thresholds_roc = roc_curve(val_y_true.cpu().data.numpy(), val_y_pred_sm.cpu().data.numpy(), pos_label=1)
        val_auc = auc(fpr, tpr)
        logging.info("evaluation result: AUC=%4f" % (val_auc))
        writer.add_scalar("val/auc", val_auc, global_step=epoch)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.module.state_dict(), '{}/ckp_model/model_best.pth'.format(exp_save_path))
        writer.add_scalar("val/best_model_result_auc", best_auc, global_step=epoch)
        return best_auc


def val3model(model1, model2, model3, val_dataloader, writer, epoch, exp_save_path, best_auc):
    model1.eval()
    model2.eval()
    model3.eval()
    val_y_true = []
    # val_y_pred = []
    val_y_pred_sm = []
    with torch.no_grad():
        for val_batch in tqdm(val_dataloader):
            val_volume = val_batch['volume'].float().cuda()
            val_label = val_batch['cspca'].cuda()
            val_name = val_batch['name']

            val_volume1 = val_volume[:, :3, ...]
            val_volume2 = val_volume[:, 3:, ...]
            feature1, _ = model1(val_volume1)
            feature2, _ = model1(val_volume2)
            y = model3(feature1, feature2)
            y_sm = F.softmax(y, dim=1)
            val_y_true.extend(val_label)
            # val_y_pred.extend(torch.max(y, 1)[1])
            val_y_pred_sm.extend(y_sm[:, 1])
            logging.info("case id: {}   label: {}   pred: {}".format(val_name, val_label.cpu(), torch.max(y, 1)[1].cpu()))

        val_y_true = torch.stack(val_y_true, dim=0)
        # val_y_pred = torch.stack(val_y_pred, dim=0)
        val_y_pred_sm = torch.stack(val_y_pred_sm, dim=0)
        # val_acc = (val_y_pred == val_y_true).sum() / val_y_true.size(0)
        fpr, tpr, thresholds_roc = roc_curve(val_y_true.cpu().data.numpy(), val_y_pred_sm.cpu().data.numpy(), pos_label=1)
        val_auc = auc(fpr, tpr)
        logging.info("evaluation result: AUC=%4f" % (val_auc))
        writer.add_scalar("val/auc", val_auc, global_step=epoch)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model1.module.state_dict(), '{}/ckp_model/model1_best.pth'.format(exp_save_path))
            torch.save(model2.module.state_dict(), '{}/ckp_model/model2_best.pth'.format(exp_save_path))
            torch.save(model3.module.state_dict(), '{}/ckp_model/model3_best.pth'.format(exp_save_path))
        writer.add_scalar("val/best_model_result_auc", best_auc, global_step=epoch)
        return best_auc


def val_HF(model, val_dataloader, writer, epoch, exp_save_path, best_auc):
    model.eval()
    val_y_true = []
    # val_y_pred = []
    val_y_pred_sm = []
    with torch.no_grad():
        for val_batch in tqdm(val_dataloader):
            val_volume = val_batch['volume'].float().cuda()
            val_label = val_batch['cspca'].cuda()
            val_name = val_batch['name']
            y, _, _, _, _, _ = model(val_volume)
            y_sm = F.softmax(y, dim=1)
            val_y_true.extend(val_label)
            # val_y_pred.extend(torch.max(y, 1)[1])
            val_y_pred_sm.extend(y_sm[:, 1])
            logging.info("case id: {}   label: {}   pred: {}".format(val_name, val_label.cpu(), torch.max(y, 1)[1].cpu()))

        val_y_true = torch.stack(val_y_true, dim=0)
        # val_y_pred = torch.stack(val_y_pred, dim=0)
        val_y_pred_sm = torch.stack(val_y_pred_sm, dim=0)
        # val_acc = (val_y_pred == val_y_true).sum() / val_y_true.size(0)
        fpr, tpr, thresholds_roc = roc_curve(val_y_true.cpu().data.numpy(), val_y_pred_sm.cpu().data.numpy(), pos_label=1)
        val_auc = auc(fpr, tpr)
        logging.info("evaluation result: AUC=%4f" % (val_auc))
        writer.add_scalar("val/auc", val_auc, global_step=epoch)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.module.state_dict(), '{}/ckp_model/model_best.pth'.format(exp_save_path))
        writer.add_scalar("val/best_model_result_auc", best_auc, global_step=epoch)
        return best_auc