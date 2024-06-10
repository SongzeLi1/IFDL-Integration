import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import warnings
warnings.filterwarnings("ignore")

from torch import nn
from PIL import Image
from tqdm import tqdm
import shutil
import timm
import time
import yaml
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataset.mydataset import *
torch.backends.cudnn.enabled = False

from torchvision.utils import save_image
from sklearn.metrics import roc_auc_score

from tools.MetricMonitor import MetricMonitor
from tools.Metric import *
from tools.MultLoss import WeightedDiceBCE

from models.rrumodel import Ringed_Res_Unet
from models.mvssnet import get_mvss
from models.CFLNet.CFLNet import CFLNet
from models import denseFCN
from models.senet import Movenet
from models.ManTraNet.Mantra_Net import ManTraNet

######## For Model ###############
def create_model(params):
    if params == 'DFCN':
        model = denseFCN.normal_denseFCN(bn_in='bn')
    if params == 'senet':
        model = Movenet([512, 512])
    if params == 'rrunet':
        model = Ringed_Res_Unet(n_channels=3, n_classes=1)
    if params == 'mvss':
        model = get_mvss(backbone='resnet50',
                         pretrained_base=True,
                         nclass=1,
                         sobel=True,
                         constrain=True,
                         n_input=3,
                         )

    if params == 'CFLNet':
        with open('models/CFLNet/config/config.yaml', 'r') as file:
            cfg_cfl = yaml.load(file, Loader=yaml.FullLoader)
        with torch.no_grad():
            test_model = timm.create_model(cfg_cfl['model_params']['encoder'], pretrained=False, features_only=True,
                                           out_indices=[4])
            in_planes = test_model(torch.randn((2, 3, 512, 512)))[0].shape[1]
            del test_model
        model = CFLNet(cfg_cfl, in_planes)

    if params == 'MantraNet':
        model = ManTraNet()

    model = model.cuda()
    model = nn.DataParallel(model).cuda()
    return model


def load_model(model, params, optimizer):
    if os.path.exists(params):
        checkpoint = torch.load(params)
        # model.module.load_state_dict(checkpoint['model'], strict=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model_name = checkpoint['model_name']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('The { ' + model_name + ' } model load weight successful!')
        return start_epoch, best_acc
    else:
        print('The model path is not exists. We will train the model from scratch.')
        return 1, 0


def save_model(model, optimizer, save_dir, dataname, epoch=0, best_acc=0, model_name='model_name', f1=0, iou=0, auc=0):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint = {
        'best_acc': best_acc,
        'epoch': epoch,
        'model': model.state_dict(),
        'model_name': model_name,
        'optimizer': optimizer.state_dict(),
    }
    best_acc = round(best_acc, 4)
    f1 = round(f1, 4)
    iou = round(iou, 4)
    auc = round(auc, 4)
    best_acc = str(best_acc)
    f1 = str(f1)
    iou = str(iou)
    auc = str(auc)
    epoch = str(epoch)
    unique_name = f"{model_name}_{dataname}_best.pth"
    filepath = os.path.join(save_dir, unique_name)
    torch.save(checkpoint, filepath)

    print('Time: {}, Save weight successful! Best acc is: {}, F1: {}, IoU: {}, AUC: {}, Epoch: {}'.format(
        time.strftime('%H:%M:%S', time.localtime()),
        best_acc, f1, iou, auc, epoch))


######## For train ###############
def train(train_loader, model, criterion1, optimizer, epoch, params, global_step):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader, desc='processing', colour='CYAN')

    for i, (images, masks, _) in enumerate(stream, start=1):
        images = images.cuda(non_blocking=params['non_blocking_'])
        masks = masks.cuda(non_blocking=params['non_blocking_'])

        reg_outs = model(images)
        reg_outs = torch.sigmoid(reg_outs)

        loss_region = criterion1(reg_outs, masks)

        optimizer.zero_grad()
        loss_region.backward()

        optimizer.step()

        global_step += 1

        metric_monitor.update("Loss", loss_region.item())
        stream.set_description(
            "Epoch: {epoch}. Train. {metric_monitor} Time: {time}".format(epoch=epoch, metric_monitor=metric_monitor,
                                                                          time=time.strftime('%H:%M:%S',
                                                                                             time.localtime()))
        )


######## Cal Score ###############
def cal_score(val_img_dir, save_dir):
    dir_1 = save_dir
    dir_2 = val_img_dir
    file_list_1 = os.listdir(dir_1)
    file_list_2 = os.listdir(dir_2)
    file_list_1.sort()
    file_list_2.sort()
    f1_list = []
    iou_list = []
    auc_list = []

    stream = tqdm(file_list_1, desc='processing', colour='CYAN')
    for i, _ in enumerate(stream, start=0):
        file_path_1 = os.path.join(dir_1, file_list_1[i])
        file_path_2 = os.path.join(dir_2, file_list_2[i])

        img_1 = cv2.imread(file_path_1, cv2.IMREAD_GRAYSCALE)
        img_2 = cv2.imread(file_path_2, cv2.IMREAD_GRAYSCALE)
        h, w = img_1.shape
        h_, w_ = img_2.shape
        if h != h_ or w != w_:
            img_2 = cv2.resize(img_2, (w, h))

        img_1 = img_1 / 255.
        img_2 = img_2 / 255.
        img_1[img_1 >= 0.5] = 1
        img_1[img_1 < 0.5] = 0

        img_2_temp = img_2
        img_2_temp[img_2_temp >= 0.5] = 1
        img_2_temp[img_2_temp < 0.5] = 0

        # 计算f1 score 和 iou score
        f1, iou = metric_numpy(img_1, img_2_temp)

        # 计算auc
        img_1_ = img_1.flatten()
        img_2_ = img_2.flatten()
        try:
            # 计算auc score
            auc = roc_auc_score(img_2_, img_1_)
            auc_list.append(auc)
        except ValueError:
            pass

        f1_list.append(f1)
        iou_list.append(iou)

    f1_avg = np.mean(f1_list)
    iou_avg = np.mean(iou_list)
    auc_avg = np.mean(auc_list)
    score = f1_avg + iou_avg
    return score, f1_avg, iou_avg, auc_avg


######## For predict ###############
def predict(val_loader, model, params, threshold):
    model.eval()
    stream = tqdm(val_loader, desc='processing', colour='CYAN')
    with torch.no_grad():
        for step, (batch_x_val, batch_y_val, w_s, h_s, name) in enumerate(stream, start=1):
            masks = batch_y_val
            batch_x_val = batch_x_val.cuda(non_blocking=params['non_blocking_'])

            output_val = model(batch_x_val)

            batch_x_val_h_flip = batch_x_val.clone().detach()
            batch_x_val_h_flip = torch.flip(batch_x_val_h_flip, [3])

            batch_x_val_v_flip = batch_x_val.clone().detach()
            batch_x_val_v_flip = torch.flip(batch_x_val_v_flip, [2])

            image_h_flip = model(batch_x_val_h_flip)
            image_v_flip = model(batch_x_val_v_flip)
            image_h_flip = torch.flip(image_h_flip, [3])
            image_v_flip = torch.flip(image_v_flip, [2])

            result_output_1 = (output_val + image_h_flip + image_v_flip) / 3.

            result_output = result_output_1
            result_output = torch.sigmoid(result_output)
            for i in range(len(result_output)):
                orig_w = w_s[i]
                orig_h = h_s[i]
                result_output_ = F.interpolate(result_output[i:i + 1], size=[orig_h, orig_w], mode="bicubic",
                                               align_corners=False)
                result_output[result_output >= threshold] = 1
                result_output[result_output < threshold] = 0
                str_ = name[i].split('/')[-1]
                name_str = str_.replace('.jpg', '.png')
                save_img_name = os.path.join(save_dir, name_str)
                save_image(result_output_, save_img_name)


######## For train_predict ###############
def predict_simple(val_loader, model, params, threshold):
    model.eval()
    stream = tqdm(val_loader, desc='processing', colour='CYAN')
    with torch.no_grad():
        f1_list = []
        iou_list = []
        auc_list = []
        for step, (batch_x_val, batch_y_val, w_s, h_s, name) in enumerate(stream, start=1):
            masks = batch_y_val
            batch_x_val = batch_x_val.cuda(non_blocking=params['non_blocking_'])
            output_val = model(batch_x_val)
            result_output = torch.sigmoid(output_val)
            result_output = result_output.cpu().data.numpy()
            masks = masks.cpu().data.numpy()

            result_output[result_output >= threshold] = 1
            result_output[result_output < threshold] = 0
            masks[masks >= threshold] = 1
            masks[masks < threshold] = 0

            # 计算f1 score 和 iou score
            f1, iou = metric_numpy(result_output, masks)

            # 计算auc
            img_1_ = result_output.flatten()
            img_2_ = masks.flatten()
            try:
                # 计算auc score
                auc = roc_auc_score(img_2_, img_1_)
                auc_list.append(auc)
            except ValueError:
                pass

            f1_list.append(f1)
            iou_list.append(iou)

        f1_avg = np.mean(f1_list)
        iou_avg = np.mean(iou_list)
        auc_avg = np.mean(auc_list)
        score = f1_avg + iou_avg + auc_avg

        return score, f1_avg, iou_avg, auc_avg


######## For train_and_validate ###############
def train_and_validate(model, optimizer, train_dataset, val_dataset, infer_img, params, epoch_start=1, best_acc=0, best_epoch=1):
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["test_batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    # Define Loss
    criterion_1 = WeightedDiceBCE(dice_weight=0.3, BCE_weight=0.7, lovasz_weight=0).cuda()

    global_step = 0

    if params["mode"] == 'train':
        best_epoch = epoch_start
        for epoch in range(epoch_start, params["epochs"] + epoch_start):
            # ##### 测试训练流程
            train(train_loader, model, criterion_1, optimizer, epoch, params, global_step)
            cur_acc, f1, iou, auc = predict_simple(val_loader, model, params, threshold=0.5)
            print(
                'current model is:{} ,current dataset is:{},current epoch is:{} ,current score is:{} ,best score is:{}, best epoch is:{}'.format(
                    params["model_name"], params["dataset_name"], epoch, cur_acc, best_acc, best_epoch))
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_epoch = epoch
                save_model(model, optimizer, params["save_dir"], params["dataset_name"], epoch, best_acc,
                           params["model_name"], f1, iou, auc)


    elif params["mode"] == 'val':
        predict(val_loader, model, params, threshold=0.35)
        cur_acc, f1, iou, auc = cal_score(infer_mask_dir, save_dir)
        print('current score is:{:3f} f1 score is:{:3f} iou score is:{:3f} auc score is:{:3f}'.format(cur_acc, f1, iou,
                                                                                                      auc))

    elif params["mode"] == 'infer':
        print('go infer')
        filename = os.listdir(infer_img)
        threshold = 0.5
        N = len(filename)
        print("total number:", N)
        stream = tqdm(filename, desc='processing', colour='CYAN')
        for img_name in stream:
            img = Image.open(infer_img + img_name)
            img = img.convert('RGB')
            out = predict_img(model=model, img=img, crop_size=512, crop_stride=512, use_gpu=True)

            out = Image.fromarray((out * 255).astype(np.uint8))
            out = np.array(out, dtype=np.float32)
            out = out / 255
            mask = out > threshold
            mask = np.array(mask, dtype=np.float32)
            result = Image.fromarray((mask * 255).astype(np.uint8))
            result.save(params["infer_dir"] + img_name[:-4].replace('ps_', 'ms_') + '.png')


def predict_img(model, img, crop_size=512, crop_stride=512, use_gpu=True):
    patch_size = crop_size
    patch_stride = crop_stride
    model.eval()
    img = img.convert('RGB')
    img = np.array(img, dtype=np.uint8)
    if img.max() <= 1:
        img = img * 255

    img = np.transpose((img / 255), (2, 0, 1))
    img = np.float32(img)
    img = torch.from_numpy(img).unsqueeze(dim=0)

    if use_gpu:
        img = img.cuda()

    with torch.no_grad():
        B, C, H, W = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
        nb_patches_h = int((H - patch_size) / patch_stride + 1)
        nb_patches_w = int((W - patch_size) / patch_stride + 1)
        if nb_patches_h <= H:
            h_label = True
        if nb_patches_w <= W:
            w_label = True
        num = 0
        hn = nb_patches_h + int(h_label)
        wn = nb_patches_w + int(w_label)

        patches_out = torch.zeros(B, hn, wn, patch_size, patch_size)
        poses = torch.zeros(B, hn, wn, 4)

        for i in range(hn):
            for j in range(wn):
                num = num + 1
                h1 = i * patch_stride
                h2 = i * patch_stride + patch_size
                w1 = j * patch_stride
                w2 = j * patch_stride + patch_size
                if i == nb_patches_h:
                    h1 = H - patch_size
                    h2 = H
                if j == nb_patches_w:
                    w1 = W - patch_size
                    w2 = W

                patch = img[:, :, h1:h2, w1:w2]
                patch_out = model(patch).squeeze().cpu()
                patch_out = torch.sigmoid(patch_out)
                patches_out[:, i, j, :, :] = patch_out
                poses[:, i, j, :] = torch.tensor([h1, h2, w1, w2])

        out = torch.zeros(B, H, W)
        times = np.zeros([B, H, W])
        for i in range(hn):
            for j in range(wn):
                patch_out = patches_out[:, i, j, :, :]
                h1, h2, w1, w2 = int(poses[:, i, j, 0]), int(poses[:, i, j, 1]), int(poses[:, i, j, 2]), int(poses[:, i, j, 3])
                out[:, h1:h2, w1:w2] = out[:, h1:h2, w1:w2] + patch_out
                times[:, h1:h2, w1:w2] = times[:, h1:h2, w1:w2] + 1
        out = out.squeeze().cpu().numpy()
        mask = np.divide(out, times[0])

    return mask


if __name__ == '__main__':
    random.seed(42)
    parent_dir = os.getcwd()
    params = {
        # model_name: DFCN | senet | rrunet | mvss | CFLNet | MantraNet
        "model_name": 'CFLNet',
        # mode: train | infer | val
        "mode": "train",
        "lr": 0.0001,
        "batch_size": 2,
        "test_batch_size": 1,
        "num_workers": 4,
        "epochs": 200,
        "non_blocking_": True,
        "dataset_name": 'FCTM'
    }

    #
    params["load_model_path"] = os.path.join(parent_dir, 'checkpoint_mine', 'mvss_FCTM_best.pth')
    params["save_dir"] = os.path.join(parent_dir, 'checkpoint_mine')

    # =============================Dataset===================================
    # train dataset
    img_dirs = ['/data/lisongze/data/FCTM/tr_val_te_811/train/tamper/']
    gt_mask_dirs = ['/data/lisongze/data/FCTM/tr_val_te_811/train/masks/']

    # val or test dataset
    val_img_dirs = ['/data/lisongze/data/FCTM/tr_val_te_811/val/tamper/']

    val_gt_mask_dirs = ['/data/lisongze/data/FCTM/tr_val_te_811/val/tamper/']

    # =============================Dataset===================================

    infer_img_dir = '/data/lisongze/data/FCTM/tr_val_te_811/test/tamper/'
    infer_mask_dir = '/data/lisongze/data/FCTM/tr_val_te_811/test/masks/'
    params["infer_dir"] = '/data/lisongze/data/FCTM/tr_val_te_811/predict_results_infer/'
    save_dir = '/data/lisongze/data/FCTM/tr_val_te_811/predict_results/'

    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)

    train_dataset = UNetDataset(img_dirs, gt_mask_dirs, mode='train', data_type=0)
    val_dataset = UNetDataset(val_img_dirs, val_gt_mask_dirs, mode='predict', data_type=0)

    model = create_model(params['model_name'])
    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-5)
    # start_epoch, best_acc = load_model(model, params['load_model_path'], optimizer)
    start_epoch, best_acc = 1, 0

    train_and_validate(model, optimizer, train_dataset, val_dataset, infer_img_dir, params, start_epoch, best_acc)