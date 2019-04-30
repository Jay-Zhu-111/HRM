import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class HRM(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, drop_ratio):
        super(HRM, self).__init__()
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)
        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim)
        self.max_pooling = MaxPoolingLayer(2)
        self.embedding_dim = embedding_dim
        # self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        # self.predictlayer = PredictLayer(embedding_dim, drop_ratio)
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                w_range = np.sqrt(3 / embedding_dim)
                nn.init.uniform_(m.weight, -w_range, w_range)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean = 0, std = 0.01)

    def forward(self, user_inputs, L_inputs, S_inputs, item_inputs):
        # item_embeds_full = self.itemembeds(Variable(torch.LongTensor(item_inputs), requires_grad=False))
        re = torch.Tensor()
        for user, L_, S_, item in zip(user_inputs, L_inputs, S_inputs, item_inputs):
            L = []
            for i in range(0, L_.__len__()):
                if L_[i] != -1:
                    L.append(L_[i])
            S = []
            for i in range(0, S_.__len__()):
                if S_[i] != -1:
                    S.append(S_[i])
            item_set = L + S

            user = [user]
            item = [item]

            # 嵌入层
            item_set_embed = self.itemembeds(torch.LongTensor(item_set))
            user_embed = self.userembeds(torch.LongTensor(user))
            item_embed = self.itemembeds(torch.LongTensor(item))

            # 第一聚集层
            item_pooling = torch.reshape(torch.transpose(item_set_embed, 0, 1),
                                           (1, self.embedding_dim, item_set.__len__()))
            while item_pooling.size(2) != 1:
                item_pooling = self.max_pooling(item_pooling)
            item_pooling = torch.reshape(item_pooling, (1, self.embedding_dim))

            # 第二聚集层
            user_item_pooling = torch.cat((user_embed, item_pooling), dim=0)
            hybrid_pooling = torch.reshape(torch.transpose(user_item_pooling, 0, 1),
                                           (1, self.embedding_dim, 2))
            hybrid_pooling = self.max_pooling(hybrid_pooling)
            hybrid_pooling = torch.reshape(hybrid_pooling, (1, self.embedding_dim))
            # print("hybrid_pooling", hybrid_pooling)
            # print("hybrid_pooling", hybrid_pooling.size())
            # 得到项目评分
            mul_result = torch.matmul(item_embed, torch.transpose(hybrid_pooling, 0, 1))
            score = mul_result
            # print(mul_result)
            # print(mul_result)
            # score = F.softmax(mul_result, dim=0)
            # print(score.size())
            # print("score", score)

            if re.size(0) == 0:
                re = score
            else:
                re = torch.cat((re, score), 0)
        return re


class UserEmbeddingLayer(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddingLayer, self).__init__()
        self.userEmbedding = nn.Embedding(num_users, embedding_dim)
        # torch.nn.init.normal(self.userEmbedding.weight)

    def forward(self, user_inputs):
        user_embeds = self.userEmbedding(user_inputs)
        return user_embeds


class ItemEmbeddingLayer(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super(ItemEmbeddingLayer, self).__init__()
        self.itemEmbedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, item_inputs):
        item_embeds = self.itemEmbedding(item_inputs)
        return item_embeds


class MaxPoolingLayer(nn.Module):
    def __init__(self, size):
        super(MaxPoolingLayer, self).__init__()
        self.max_pooling = nn.MaxPool1d(size)

    def forward(self, item_inputs):
        item_pooling = self.max_pooling(item_inputs)
        return item_pooling


# class AttentionLayer(nn.Module):
#     def __init__(self, embedding_dim, drop_ratio=0):
#         super(AttentionLayer, self).__init__()
#         self.linear = nn.Sequential(
#             nn.Linear(embedding_dim, 16),
#             nn.ReLU(),
#             nn.Dropout(drop_ratio),
#             nn.Linear(16, 1)
#         )
#
#     def forward(self, x):
#         out = self.linear(x)
#         weight = F.softmax(out.view(1, -1), dim=1)
#         return weight


# class PredictLayer(nn.Module):
#     def __init__(self, embedding_dim, drop_ratio=0):
#         super(PredictLayer, self).__init__()
#         self.linear = nn.Sequential(
#             nn.Linear(embedding_dim, 8),
#             nn.ReLU(),
#             nn.Dropout(drop_ratio),
#             nn.Linear(8, 1)
#         )
#
#     def forward(self, x):
#         out = self.linear(x)
#         return out

