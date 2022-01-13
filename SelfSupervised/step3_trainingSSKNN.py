import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
from Mesh_dataset import *
from meshsegnetSSKNN import *
from losses_and_metrics_for_mesh import *
import utils
import pandas as pd
import random
import dgcnn
import provider

if __name__ == '__main__':

    torch.cuda.set_device(utils.get_avail_gpu()) # assign which gpu will be used (only linux works)
    use_visdom = False # if you don't use visdom, please set to False

    train_list = './Supervised Train registered_vtp/train_list_3.csv'
    val_list = './Supervised Train registered_vtp/val_list_3.csv'

    KNN_ROOT = './Selfsupervised Clustered VTP/'
    model_path = '/home/ammar15/scratch/semisupervisedTraining/models/'
    model_name = 'semisupervisedTrainingModel' # need to define
    checkpoint_name = 'latest_checkpoint.tar'

    num_classes = 15
    num_channels = 15 # number of features
    num_epochs = 60
    num_workers = 0
    train_batch_size = 8 # on compute Canada should be 10
    val_batch_size = 8
    num_batches_to_print = 16

    if use_visdom:
        # set plotter
        global plotter
        plotter = utils.VisdomLinePlotter(env_name=model_name)

    # mkdir 'models'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # set dataset
    training_dataset = Mesh_Dataset(data_list_path=train_list,
                                    num_classes=num_classes,
                                    patch_size=9000)

    val_dataset = Mesh_Dataset(data_list_path=val_list,
                               num_classes=num_classes,
                               patch_size=9000)


    train_loader = DataLoader(dataset=training_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=num_workers)


    SELFSUP_DATASET = KNNSelfSupDataset(root=KNN_ROOT, num_classes=60, patch_size=9000,
                                        npoints=200,
                                        class_choice=None,
                                        normal_channel=False,
                                        k_shot=14,
                                        use_val=False,
                                        exclude_fns=[])

    selfsupDataLoader = torch.utils.data.DataLoader(SELFSUP_DATASET,
                                                    batch_size=2,
                                                    shuffle=True,
                                                    num_workers=4)


# set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MeshSegNet(num_classes=num_classes, num_channels=num_channels, with_dropout=True, dropout_p=0.5).to(device, dtype=torch.float)
    opt = optim.Adam(model.parameters(), amsgrad=True)

    selfsupCriterion = dgcnn.get_selfsup_loss(margin=0.5).cuda()

    losses, mdsc, msen, mppv = [], [], [], []
    val_losses, val_mdsc, val_msen, val_mppv = [], [], [], []

    best_val_dsc = 0.0

    #cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    print('Training model...')
    class_weights = torch.ones(num_classes).to(device, dtype=torch.float)
    for epoch in range(num_epochs):

        # training
        model.train()
        running_loss = 0.0
        running_mdsc = 0.0
        running_msen = 0.0
        running_mppv = 0.0
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0
        for i_batch, batched_sample in enumerate(train_loader):

            # send mini-batch to device
            inputs = batched_sample['cells'].to(device, dtype=torch.float)
            labels = batched_sample['labels'].to(device, dtype=torch.long)
            A_S = batched_sample['A_S'].to(device, dtype=torch.float)
            A_L = batched_sample['A_L'].to(device, dtype=torch.float)
            one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs, _ = model(inputs, A_S, A_L)
            loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
            dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
            sen = weighting_SEN(outputs, one_hot_labels, class_weights)
            ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
            loss.backward()
            opt.step()

            # Applying self-supervised loss
            '''applying supervised cross-entropy loss'''
            selfsupIterator = next(iter(selfsupDataLoader))
            sample = selfsupIterator
            inputsKNN = sample['cells'].to(device, dtype=torch.float)
            target = (torch.squeeze(sample['labels'])).to(device, dtype=torch.long)
            A_S = sample['A_S'].to(device, dtype=torch.float)
            A_L = sample['A_L'].to(device, dtype=torch.float)
            opt.zero_grad()
            model = model.train()
            '''applying self-supervised constrastive (pairwise) loss'''
            _, feat = model(inputsKNN, A_S, A_L)
            ss_loss = selfsupCriterion(feat, target) * 10.0
            ss_loss.backward()
            opt.step()

            # print statistics
            running_loss += loss.item()
            running_mdsc += dsc.item()
            running_msen += sen.item()
            running_mppv += ppv.item()
            loss_epoch += loss.item()
            mdsc_epoch += dsc.item()
            msen_epoch += sen.item()
            mppv_epoch += ppv.item()
            if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
                print('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}, sen: {6}, ppv: {7}'.format(epoch+1, num_epochs, i_batch+1, len(train_loader), running_loss/num_batches_to_print, running_mdsc/num_batches_to_print, running_msen/num_batches_to_print, running_mppv/num_batches_to_print))
                if use_visdom:
                    plotter.plot('loss', 'train', 'Loss', epoch+(i_batch+1)/len(train_loader), running_loss/num_batches_to_print)
                    plotter.plot('DSC', 'train', 'DSC', epoch+(i_batch+1)/len(train_loader), running_mdsc/num_batches_to_print)
                    plotter.plot('SEN', 'train', 'SEN', epoch+(i_batch+1)/len(train_loader), running_msen/num_batches_to_print)
                    plotter.plot('PPV', 'train', 'PPV', epoch+(i_batch+1)/len(train_loader), running_mppv/num_batches_to_print)
                running_loss = 0.0
                running_mdsc = 0.0
                running_msen = 0.0
                running_mppv = 0.0

        # record losses and metrics
        losses.append(loss_epoch/len(train_loader))
        mdsc.append(mdsc_epoch/len(train_loader))
        msen.append(msen_epoch/len(train_loader))
        mppv.append(mppv_epoch/len(train_loader))

        #reset
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0

        # validation
        model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            running_val_mdsc = 0.0
            running_val_msen = 0.0
            running_val_mppv = 0.0
            val_loss_epoch = 0.0
            val_mdsc_epoch = 0.0
            val_msen_epoch = 0.0
            val_mppv_epoch = 0.0
            for i_batch, batched_val_sample in enumerate(val_loader):

                # send mini-batch to device
                inputs = batched_val_sample['cells'].to(device, dtype=torch.float)
                labels = batched_val_sample['labels'].to(device, dtype=torch.long)
                A_S = batched_val_sample['A_S'].to(device, dtype=torch.float)
                A_L = batched_val_sample['A_L'].to(device, dtype=torch.float)
                one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

                outputs, _ = model(inputs, A_S, A_L)
                loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
                dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
                sen = weighting_SEN(outputs, one_hot_labels, class_weights)
                ppv = weighting_PPV(outputs, one_hot_labels, class_weights)

                running_val_loss += loss.item()
                running_val_mdsc += dsc.item()
                running_val_msen += sen.item()
                running_val_mppv += ppv.item()
                val_loss_epoch += loss.item()
                val_mdsc_epoch += dsc.item()
                val_msen_epoch += sen.item()
                val_mppv_epoch += ppv.item()

                if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
                    print('[Epoch: {0}/{1}, Val batch: {2}/{3}] val_loss: {4}, val_dsc: {5}, val_sen: {6}, val_ppv: {7}'.format(epoch+1, num_epochs, i_batch+1, len(val_loader), running_val_loss/num_batches_to_print, running_val_mdsc/num_batches_to_print, running_val_msen/num_batches_to_print, running_val_mppv/num_batches_to_print))
                    running_val_loss = 0.0
                    running_val_mdsc = 0.0
                    running_val_msen = 0.0
                    running_val_mppv = 0.0

            # record losses and metrics
            val_losses.append(val_loss_epoch/len(val_loader))
            val_mdsc.append(val_mdsc_epoch/len(val_loader))
            val_msen.append(val_msen_epoch/len(val_loader))
            val_mppv.append(val_mppv_epoch/len(val_loader))

            # reset
            val_loss_epoch = 0.0
            val_mdsc_epoch = 0.0
            val_msen_epoch = 0.0
            val_mppv_epoch = 0.0

            # output current status
            print('*****\nEpoch: {}/{}, loss: {}, dsc: {}, sen: {}, ppv: {}\n         val_loss: {}, val_dsc: {}, val_sen: {}, val_ppv: {}\n*****'.format(epoch+1, num_epochs, losses[-1], mdsc[-1], msen[-1], mppv[-1], val_losses[-1], val_mdsc[-1], val_msen[-1], val_mppv[-1]))
            if use_visdom:
                plotter.plot('loss', 'train', 'Loss', epoch+1, losses[-1])
                plotter.plot('DSC', 'train', 'DSC', epoch+1, mdsc[-1])
                plotter.plot('SEN', 'train', 'SEN', epoch+1, msen[-1])
                plotter.plot('PPV', 'train', 'PPV', epoch+1, mppv[-1])
                plotter.plot('loss', 'val', 'Loss', epoch+1, val_losses[-1])
                plotter.plot('DSC', 'val', 'DSC', epoch+1, val_mdsc[-1])
                plotter.plot('SEN', 'val', 'SEN', epoch+1, val_msen[-1])
                plotter.plot('PPV', 'val', 'PPV', epoch+1, val_mppv[-1])

        # save the checkpoint
        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'losses': losses,
                    'mdsc': mdsc,
                    'msen': msen,
                    'mppv': mppv,
                    'val_losses': val_losses,
                    'val_mdsc': val_mdsc,
                    'val_msen': val_msen,
                    'val_mppv': val_mppv},
                    model_path+checkpoint_name)

        # save the best model
        if best_val_dsc < val_mdsc[-1]:
            best_val_dsc = val_mdsc[-1]
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'losses': losses,
                        'mdsc': mdsc,
                        'msen': msen,
                        'mppv': mppv,
                        'val_losses': val_losses,
                        'val_mdsc': val_mdsc,
                        'val_msen': val_msen,
                        'val_mppv': val_mppv},
                        model_path+'{}_best.tar'.format(model_name))

        # save all losses and metrics data
        pd_dict = {'loss': losses, 'DSC': mdsc, 'SEN': msen, 'PPV': mppv, 'val_loss': val_losses, 'val_DSC': val_mdsc, 'val_SEN': val_msen, 'val_PPV': val_mppv}
        stat = pd.DataFrame(pd_dict)
        stat.to_csv(r'/home/ammar15/scratch/semisupervisedTraining/losses_metrics_vs_epoch.csv')