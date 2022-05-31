# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author axiu mao
"""

import os
import csv
import argparse
import time
import numpy as np
import seaborn as sns
import platform
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import torch.utils.model_zoo as model_zoo
# import matplotlib.font_manager as font_manager

import torch
import torch.nn as nn
import torch.optim as optim
 
from conf import settings
from Regularization import Regularization
from utils import get_network, get_mydataloader
from sklearn.metrics import f1_score, classification_report, confusion_matrix, cohen_kappa_score, recall_score, precision_score


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(args, optimizer, i_iter, len_train):
    steps = (len_train/args.b)*150
    lr = lr_poly(args.lr, i_iter, steps, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def train(train_loader, network, optimizer, epoch, loss_function):

    start = time.time()
    network.train()
    train_acc_process = []
    train_loss_process = []
    for batch_index, (images, labels) in enumerate(train_loader):
        
        len_train = len(train_loader.dataset)
        step = (len_train/args.b)*(epoch-1)+batch_index #当前的iteration;
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            loss_function = loss_function.cuda()

        optimizer.zero_grad() # clear gradients for this training step
        lr = adjust_learning_rate(args, optimizer, step, len_train)
        outputs = network(images)
        out = outputs.squeeze(dim=-1)
        loss = loss_function(out, labels.float())
        if args.weight_d > 0:
            loss = loss + reg_loss(net)
        
        loss.backward() # backpropogation, compute gradients
        optimizer.step() # apply gradients

        preds = torch.zeros_like(out)
        for i in range(len(out)):
            if out[i] <= 0.5:
                preds[i] = 0
            else:
                preds[i] = 1
        # print("prediction",preds)
        correct_n = preds.eq(labels).sum()
        accuracy_iter = correct_n.float() / len(labels)
        
        if args.gpu:
            accuracy_iter = accuracy_iter.cpu()
        
        train_acc_process.append(accuracy_iter.numpy().tolist())
        train_loss_process.append(loss.item())

    print('Training Epoch: {epoch} [{total_samples}]\tTrain_accuracy: {:.4f}\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            np.mean(train_acc_process),
            np.mean(train_loss_process),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            total_samples=len(train_loader.dataset)
    ))
    
    Train_Accuracy.append(np.mean(train_acc_process))
    Train_Loss.append(np.mean(train_loss_process))
    
    finish = time.time()
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    
    return network


@torch.no_grad()
def eval_training(valid_loader, network,loss_function, epoch=0):

    start = time.time()
    network.eval()
    
    n = 0
    valid_loss = 0.0 # cost function error
    correct = 0.0
    class_target =[]
    class_predict = []

    for (images, labels) in valid_loader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
            loss_function = loss_function.cuda()

        outputs = network(images)
        out = outputs.squeeze(dim=-1)
        loss = loss_function(out, labels.float())
        valid_loss += loss.item()
        
        preds = torch.zeros_like(out)
        for i in range(len(out)):
            if out[i] <= 0.5:
                preds[i] = 0
            else:
                preds[i] = 1
        correct += preds.eq(labels).sum()
        
        if args.gpu:
            labels = labels.cpu()
            preds = preds.cpu()
        
        class_target.extend(labels.numpy().tolist())
        class_predict.extend(preds.numpy().tolist())
        
        n +=1
    finish = time.time()
    
    print('Evaluating Network.....')
    print('Valid set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        valid_loss / n, #总的平均loss
        correct.float() / len(valid_loader.dataset),
        finish - start
    ))
    
    #Obtain f1_score of the prediction
    fs = f1_score(class_target, class_predict, average='macro')
    print('f1 score = {}'.format(fs))
    
    #Output the classification report
    print('------------')
    print('Classification Report')
    print(classification_report(class_target, class_predict))
    
    f1_s.append(fs)
    Valid_Loss.append(valid_loss / n)
    Valid_Accuracy.append(correct.float() / len(valid_loader.dataset))
    
    print('Setting: Epoch: {}, Batch size: {}, Learning rate: {:.6f}, gpu:{}, seed:{}'.format(args.epoch, args.b, args.lr, args.gpu, args.seed))

    return correct.float() / len(valid_loader.dataset), valid_loss / len(valid_loader.dataset), fs


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--net', type=str, default='canet', help='net type')
    parser.add_argument('--net', type=str, default='vgg11', help='net type')
    parser.add_argument('--gpu', type = int, default=0, help='use gpu or not')
    parser.add_argument('--b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--epoch',type=int, default=100, help='total training epoches')
    parser.add_argument('--seed',type=int, default=1, help='seed')
    parser.add_argument('--weight_d',type=float, default=0, help='weight decay for regularization')
    parser.add_argument('--save_path',type=str, default='setting0', help='saved path of each setting')
    parser.add_argument('--data_path',type=str, default='..\\1_new_separate_normal_myTensor_Log_power_88_bird_acoustic.pt', help='saved path of input data')
    args = parser.parse_args()

    if args.gpu:
        torch.cuda.manual_seed(args.seed)
      
    else:
        torch.manual_seed(args.seed)
       
    model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
    
    #Create the network
    net = get_network(args)
    print(net)
    
    # #Load pretrained weights
    pretrained_dict = model_zoo.load_url(model_urls['vgg11_bn'])
    net_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape == net_dict[k].shape)}
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)
    print(len(net_dict))
    print(len(pretrained_dict))
    print("Having loaded imagenet-pretrained successfully!")

    print('Setting: Epoch: {}, Batch size: {}, Learning rate: {:.6f}, gpu:{}'.format(args.epoch, args.b, args.lr, args.gpu))

    sysstr = platform.system()
    if(sysstr =="Windows"):
        num_workers = 0
    else:
        num_workers = 8

    pathway = args.data_path
    if sysstr=='Linux': 
        pathway = args.data_path
    
    train_loader = get_mydataloader(pathway, data_id=0, batch_size=args.b, num_workers=num_workers, shuffle=True)
    valid_loader = get_mydataloader(pathway, data_id=1, batch_size=args.b, num_workers=num_workers, shuffle=True)
    test_loader = get_mydataloader(pathway, data_id=2, batch_size=args.b, num_workers=num_workers, shuffle=False)

    
    #'natural barn sounds':0; 'distressCall':1
    for i, (x, y) in enumerate(train_loader):
        
        print("batch index {}, 0/1: {}/{}".format(i, (y == 0).sum(), (y == 1).sum()))
    print('------------------------------------------')
    for i, (x, y) in enumerate(valid_loader):
        print("batch index {}, 0/1: {}/{}".format(i, (y == 0).sum(), (y == 1).sum()))
    
    if args.weight_d > 0:
        reg_loss=Regularization(net, args.weight_d, p=2)
    else:
        print("no regularization")
    
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, args.save_path, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
  
    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path_pth = os.path.join(checkpoint_path, '{net}-{type}.pth')

    best_acc = 0.0
    Train_Loss = []
    Train_Accuracy = []
    Valid_Loss = []
    Valid_Accuracy = []
    f1_s = []
    best_epoch = 1
    best_weights_path = checkpoint_path_pth.format(net=args.net, type='best')
    for epoch in range(1, args.epoch + 1):
            
        net = train(train_loader, net, optimizer, epoch, loss_function)
        acc, validation_loss, fs_valid = eval_training(valid_loader, net, loss_function, epoch)
        
        #start to save best performance model (according to the accuracy on validation dataset) after learning rate decay to 0.01
        if epoch > settings.MILESTONES[0] and best_acc < acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(net.state_dict(), best_weights_path)
    print('best epoch is {}'.format(best_epoch))
    
    #####Output results
    #plot train loss and accuracy vary over time
    font_1 = {'weight' : 'normal', 'size'   : 20}
    fig1=plt.figure(figsize=(12,9))
    plt.title('Accuracy')
    index_train = list(range(1,len(Train_Accuracy)+1))
    plt.plot(index_train,Train_Accuracy,color='skyblue',label='train_accuracy')
    plt.plot(index_train,Valid_Accuracy,color='red',label='valid_accuracy')
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0,100)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('Accuracy',font_1)
    
    acc_figuresavedpath = os.path.join(checkpoint_path,'Accuracy_curve.png')
    plt.savefig(acc_figuresavedpath)
    # plt.show()
    
    #plot valid loss and accuracy vary over time
    fig2=plt.figure(figsize=(12,9))
    plt.title('Loss')
    index_valid = list(range(1,len(Valid_Loss)+1))
    plt.plot(index_valid,Train_Loss,color='skyblue', label='train_loss')
    plt.plot(index_valid,Valid_Loss,color='red', label='valid_loss')
    # plt.legend(prop=font_2)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0,100)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('Loss',font_1)
    
    loss_figuresavedpath = os.path.join(checkpoint_path,'Loss_curve.png')
    plt.savefig(loss_figuresavedpath)
    # plt.show()
    
    #plot f1 score varying over time
    fig3=plt.figure(figsize=(12,9))
    plt.title('F1-score',font_1)
    index_fs = list(range(1,len(f1_s)+1))
    plt.plot(index_fs,f1_s,color='skyblue')
    # plt.legend(prop=font_2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.xlim(0,100)
    plt.xlabel('n_iter',font_1)
    plt.ylabel('Loss',font_1)

    fs_figuresavedpath = os.path.join(checkpoint_path,'F1-score.png')
    plt.savefig(fs_figuresavedpath)
    # plt.show()
    
    out_txtsavedpath = os.path.join(checkpoint_path,'output.txt')
    f = open(out_txtsavedpath, 'w+')
    
    print('Setting: Seed:{}, Epoch: {}, Batch size: {}, Learning rate: {:.6f}, Weight decay: {}, gpu:{}, Saved path: {}'.format(
        args.seed, args.epoch, args.b, args.lr, args.weight_d, args.gpu, args.save_path),
        file=f)
    
    print('index: {}; maximum value of validation accuracy: {}.'.format(Valid_Accuracy.index(max(Valid_Accuracy))+1, max(Valid_Accuracy)), file=f)
    print('index: {}; maximum value of validation f1-score: {}.'.format(f1_s.index(max(f1_s))+1, max(f1_s)), file=f)
    print('--------------------------------------------------')
    print('Validation accuracy: {}'.format(Valid_Accuracy), file=f)
    print('Validation F1-score: {}'.format(f1_s), file=f)
    
    #Load the best trained model and test testing data
    best_net = get_network(args)
    # print(best_weights_path)
    best_net.load_state_dict(torch.load(best_weights_path))
    
    total_num_paras, trainable_num_paras = get_parameter_number(best_net)
    print('The total number of network parameters = {}'.format(total_num_paras), file=f)
    print('The trainable number of network parameters = {}'.format(trainable_num_paras), file=f)
    
    best_net.eval()
    number = 0
    correct_test = 0.0
    test_target =[]
    test_predict = []
    
    with torch.no_grad():
        
        start = time.time()
        
        for n_iter, (image, labels) in enumerate(test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            if args.gpu:
                image = image.cuda()
                labels = labels.cuda()

            output = best_net(image)
            out = output.squeeze(dim=-1)
            preds = torch.zeros_like(out)
            for i in range(len(out)):
                if out[i] <= 0.5:
                    preds[i] = 0
                else:
                    preds[i] = 1

            correct_test += preds.eq(labels).sum()
            
            if args.gpu:
                labels = labels.cpu()
                preds = preds.cpu()
        
            test_target.extend(labels.numpy().tolist())
            test_predict.extend(preds.numpy().tolist())
        
            number +=1
        
        print('Label values: {}'.format(test_target), file=f)
        print('Predicted values: {}'.format(test_predict), file=f)

        finish = time.time()
        accuracy_test = correct_test.float() / len(test_loader.dataset)
        print('Testing network......', file=f)
        print('Test set: Accuracy: {:.5f}, Time consumed: {:.5f}s'.format(
            accuracy_test,
            finish - start
            ), file=f)
        
        #Obtain f1_score of the prediction
        fs_test = f1_score(test_target, test_predict, average='macro')
        print('f1 score = {:.5f}'.format(fs_test), file=f)
        
        kappa_value = cohen_kappa_score(test_target, test_predict)
        print("kappa value = {:.5f}".format(kappa_value), file=f)
        
        precision_test = precision_score(test_target, test_predict, average='macro')
        print('precision = {:.5f}'.format(precision_test), file=f)
        
        recall_test = recall_score(test_target, test_predict, average='macro')
        print('recall = {:.5f}'.format(recall_test), file=f)
        
        #Output the classification report
        print('------------', file=f)
        print('Classification Report', file=f)
        print(classification_report(test_target, test_predict), file=f)
        
        if not os.path.exists('./results.csv'):
            with open("./results.csv", 'w+') as csvfile:
                writer_csv = csv.writer(csvfile)
                writer_csv.writerow(['index','accuracy','f1-score','precision','recall','kappa','time_consumed'])
        
        with open("./results.csv", 'a+') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow([args.seed, accuracy_test, fs_test, precision_test, recall_test, kappa_value, finish-start])
        
        Class_labels = ['natural barn sounds','distress']
        #Show the confusion matrix so that it can help us observe the results more intuitively
        def show_confusion_matrix(validations, predictions):
            matrix = confusion_matrix(validations, predictions) #No one-hot
            #matrix = confusion_matrix(validations.argmax(axis=1), predictions.argmax(axis=1)) #One-hot
            plt.figure(figsize=(6, 4))
            sns.heatmap(matrix,
                  cmap="coolwarm",
                  linecolor='white',
                  linewidths=1,
                  xticklabels=Class_labels,
                  yticklabels=Class_labels,
                  annot=True,
                  fmt="d")
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            cm_figuresavedpath = os.path.join(checkpoint_path,'Confusion_matrix.png')
            plt.savefig(cm_figuresavedpath)

        show_confusion_matrix(test_target, test_predict)
    
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
