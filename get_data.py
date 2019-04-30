import Const
import random
import torch
from torch.utils.data import TensorDataset, DataLoader


class Data:
    def __init__(self):
        with open('data/user_dict.txt', 'r', encoding='UTF-8') as user_dict_train_f:
            self.user_dict = eval(user_dict_train_f.read())

        with open('data/item_set.txt', 'r', encoding='UTF-8') as item_set_f:
            self.item_set = eval(item_set_f.read())

        with open('data/user_set.txt', 'r', encoding='UTF-8') as user_set_f:
            self.user_set = eval(user_set_f.read())

        with open('data/user_input_train.txt', 'r', encoding='UTF-8') as f:
            self.user = eval(f.read())
        with open('data/L_input_train.txt', 'r', encoding='UTF-8') as f:
            self.L = eval(f.read())
        with open('data/S_input_train.txt', 'r', encoding='UTF-8') as f:
            self.S = eval(f.read())
        with open('data/pos_item_input_train.txt', 'r', encoding='UTF-8') as f:
            self.j = eval(f.read())
        with open('data/neg_item_input_train.txt', 'r', encoding='UTF-8') as f:
            self.k = eval(f.read())
        print("训练数据长度", self.user.__len__())
        print("用户数", self.get_user_size())
        print("项目数", self.get_item_size())
        # self.device = device
        # # user = input("输入用户ID:")
        # self.u = '6756'
        # for i in range(0, self.user_set.__len__()):
        #     if self.user_set[i] == self.u:
        #         self.u_id = i
        #         break
        #
        # self.user_info = self.user_dict[self.u]
        #
        # self.S = []
        # self.L = []
        # for item in self.user_info.items():  # 分别存入用户长期和短期项目集
        #     if int(item[0]) > currentTime - timeStep:
        #         self.S.append(item[1])
        #     else:
        #         self.L.append(item[1])

    # 总用户数
    def get_user_size(self):
        return self.user_set.__len__()

    # 总项目数
    def get_item_size(self):
        return self.item_set.__len__()

    # 返回物品
    def get_single_item(self, item_id):
        return self.item_set[item_id]

    # 返回物品序号
    def get_single_item_id(self, item):
        re = 0
        for i in range(0, self.item_set.__len__()):
            if item == self.item_set[i]:
                re = i
                break
        return re

    # 返回物品序号集
    def get_item_id(self, input_set):
        item_list = []
        for item in input_set:
            for i in range(0, self.item_set.__len__()):
                if item == self.item_set[i]:
                    item_list.append(i)
                    break
        return item_list

    # # 返回用户ID
    # def get_u(self):
    #     return self.u_id
    #
    # # 返回长期项目集
    # def get_L(self):
    #     return self.L
    #
    # # 返回短期项目集
    # def get_S(self):
    #     return self.S

    # # 获取观察样本集
    # def get_ob_set(self):
    #     ob_set = []
    #     for item in self.user_info.items():
    #         (u, L, S, j) = self.get_ob(int(item[0]))
    #         ob_set.append((u, L, S, j))
    #     return ob_set

    # 获取一个观察样本
    def get_ob(self, size):
        user_input, L_input, S_input, pos_item_input, neg_item_input = [], [], [], [], []
        for i in range(0, size):
            u = self.user_set[i]
            u_id = i
            user_info = self.user_dict[u]
            for time in user_info.keys():
                L = []
                S = []
                j = 0
                for item in user_info.items():  # 分别存入用户长期和短期项目集
                    if int(item[0]) >= int(time):
                        j = self.get_single_item_id(item[1])
                    elif int(item[0]) > int(time) - timeStep:
                        S.append(self.get_single_item_id(item[1]))
                    else:
                        L.append(self.get_single_item_id(item[1]))
                if L.__len__() == 0 and S.__len__() == 0:
                    continue
                k = self.get_unob(L, S)
                user_input.append(u_id)
                L_input.append(L)
                S_input.append(S)
                pos_item_input.append(j)
                neg_item_input.append(k)
        # L_S_pi_ni = [[L, S, pi, ni] for L, S, pi, ni in zip(L_input, S_input, pos_item_input, neg_item_input)]
        # L_S_pi_ni111 = torch.LongTensor(L_S_pi_ni)
        # print("user_long_tensor", L_S_pi_ni111)
        # print(type(L_S_pi_ni111))

        # 填充L_input
        L_max = 0
        for i in range(0, L_input.__len__()):
            if L_input[i].__len__() > L_max:
                L_max = L_input[i].__len__()
        for i in range(0, L_input.__len__()):
            while L_input[i].__len__() < L_max:
                L_input[i].append(-1)

        # 填充S_input
        S_max = 0
        for i in range(0, S_input.__len__()):
            if S_input[i].__len__() > S_max:
                S_max = S_input[i].__len__()
        for i in range(0, S_input.__len__()):
            while S_input[i].__len__() < S_max:
                S_input[i].append(-1)
        # print(user_input.__len__())
        # tensorU = torch.LongTensor(L_input)
        # print('tensorU')
        # print(tensorU)
        # print(tensorU.size())
        return user_input, L_input, S_input, pos_item_input, neg_item_input

    # 随机获取一个负样本
    def get_unob(self, L, S):
        ob_item = []
        for item in L:
            ob_item.append(item)
        for item in S:
            ob_item.append(item)
        while True:
            k = random.randint(0, self.item_set.__len__() - 1)
            if k not in ob_item:
                break
        return k

    # 返回训练dataloader
    def get_dataloader(self, batch_size):
        # if is_training:
        # with open('data/user_input_train.txt', 'r', encoding='UTF-8') as f:
        #     user = eval(f.read())
        # with open('data/L_input_train.txt', 'r', encoding='UTF-8') as f:
        #     L = eval(f.read())
        # with open('data/S_input_train.txt', 'r', encoding='UTF-8') as f:
        #     S = eval(f.read())
        # with open('data/pos_item_input_train.txt', 'r', encoding='UTF-8') as f:
        #     j = eval(f.read())
        # with open('data/neg_item_input_train.txt', 'r', encoding='UTF-8') as f:
        #     k = eval(f.read())
        # else:
        #     with open('data/user_input_test.txt', 'r', encoding='UTF-8') as f:
        #         user = eval(f.read())
        #     with open('data/L_input_test.txt', 'r', encoding='UTF-8') as f:
        #         L = eval(f.read())
        #     with open('data/S_input_test.txt', 'r', encoding='UTF-8') as f:
        #         S = eval(f.read())
        #     with open('data/pos_item_input_test.txt', 'r', encoding='UTF-8') as f:
        #         j = eval(f.read())
        #     with open('data/neg_item_input_test.txt', 'r', encoding='UTF-8') as f:
        #         k = eval(f.read())
        data = TensorDataset(
            torch.LongTensor(self.user),
            torch.LongTensor(self.L),
            torch.LongTensor(self.S),
            torch.LongTensor(self.j),
            torch.LongTensor(self.k)
        )
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        return data_loader

    def get_dataloader_with_size(self, batch_size, size):
        # user, L, S, j, k = self.get_ob(size)
        # print(L)
        data = TensorDataset(
            torch.LongTensor(self.user[0: size]),
            torch.LongTensor(self.L[0: size]),
            torch.LongTensor(self.S[0: size]),
            torch.LongTensor(self.j[0: size]),
            torch.LongTensor(self.k[0: size])
        )
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        return data_loader

    # # 返回dataframe
    # def get_dataframe(self):
    #     data = pd.DataFrame.from_dict(self.user_info, orient='index')
    #     data.rename(columns = {'user', 'item'})
    #     return data
    #
    # # 返回dataset
    # def read_data(self, batch_size, is_training):
    #     user_dict = {}
    #     user_dict['time'] = np.array( list(self.user_info.keys()) )
    #     user_dict['item'] = np.array(list(self.user_info.values()))
    #     dataset = tf.data.Dataset.from_tensor_slices(user_dict)
    #     if is_training:
    #         dataset = dataset.shuffle(10000)
    #     dataset = dataset.batch(batch_size)
    #
    #     return dataset

# data = Data()
