from get_data import Data
from _model import HRM
from matplotlib.pyplot import *
import Const
import torch
import torch.optim as optim
from time import time
import numpy as np
import helper
from helper import Helper
from torch.autograd import Variable
import torch.nn.functional as F

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# train the model
def training(model, train_loader, epoch_id, lr):
    # optimizer 正则化参数？？
    optimizer = optim.Adam(model.parameters(), lr)

    losses = []
    for user_input, L_input, S_input, pos_item_input, neg_item_input in train_loader:
        # Forward
        pos_prediction = model(user_input, L_input, S_input, pos_item_input)
        neg_prediction = model(user_input, L_input, S_input, neg_item_input)
        # print('pos', pos_prediction)
        # print('neg', neg_prediction)
        temp = torch.cat((pos_prediction, neg_prediction), dim=1)
        # print(temp)
        # print(temp.size())
        # print(F.softmax(temp, dim=1))
        # Zero_grad
        model.zero_grad()

        # Loss
        loss = torch.mean(torch.neg(torch.log(torch.sigmoid(pos_prediction)) + torch.log(torch.sigmoid(torch.neg(neg_prediction)))))

        # record loss history
        losses.append(float(loss))

        # Backward
        loss.backward()
        optimizer.step()

    mean_loss = 0
    for loss in losses:
        mean_loss += loss
    mean_loss /= losses.__len__()
    print("epoch_id", epoch_id)
    print("mean_loss", mean_loss)
    # print('Iteration %d, loss is [%.4f ]' % (epoch_id, losses ))
    return mean_loss


def evaluation(model, Helper):
    model.eval()
    (hits, AUCs) = Helper.evaluate_model(model)

    # Recall
    count = 0.0
    for num in hits:
        if num == 1:
            count = count + 1
    Recall = count / hits.__len__()

    # AUC
    count = 0.0
    for num in AUCs:
        count = count + num
    AUC = count / AUCs.__len__()

    return Recall, AUC


if __name__ == '__main__':
    embedding_size = Const.embedding_size
    drop_ratio = Const.drop_ratio
    epoch = Const.epoch
    batch_size = Const.batch_size

    data = Data()
    h = Helper()
    num_users = data.get_user_size()
    num_items = data.get_item_size()
    hrm = HRM(num_users, num_items, embedding_size, drop_ratio)
    # print(hrm)

    lr_flag = True
    pre_mean_loss = 999
    lr = Const.lr
    for i in range(0, epoch):
        hrm.train()
        # 开始训练时间
        t1 = time()
        if lr_flag:
            lr *= 1.1
        else:
            lr *= 0.5
        mean_loss = training(hrm, data.get_dataloader(batch_size), i, lr)
        if mean_loss < pre_mean_loss:
            lr_flag = True
        else:
            lr_flag = False
        pre_mean_loss = mean_loss
        print("learning rate is: ", lr)
        print("Training time is: [%.1f s]" % (time() - t1))

        # evaluating
        t2 = time()
        Recall, AUC = evaluation(hrm, h)
        print("Recall", Recall)
        print("AUC", AUC)
        print("Evalulating time is: [%.1f s]" % (time() - t2))
        print("\n")

    # helper.draw_loss()
    print("Done!")
