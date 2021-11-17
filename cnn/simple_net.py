import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim



class MyCNN(nn.Module):
    def __init__(self):
        # 创建一个pytorch神经网络模型
        super(MyCNN, self).__init__()
        # 卷积层1，32通道输出，卷积核大小3*3，步长1*1，padding为1
        self.Conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        # 最大值池化，核大小2*2，步长2*2
        self.pool1 = nn.MaxPool2d(2, 2)
        self.Conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.Conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 768)
        self.fc2 = nn.Linear(768, 10)

    def forward(self, x):
        x = self.pool1(func.relu(self.Conv1(x)))
        x = self.pool2(func.relu(self.Conv2(x)))
        x = self.pool3(func.relu(self.Conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        # softmax层，输出预测结果
        x = func.softmax(x, dim=1)  # 注意这里将给出BATCH_SIZE*10的矩阵
        return x

class poly_MyCNN(nn.Module):
    def __init__(self):
        # 创建一个pytorch神经网络模型
        super(poly_MyCNN, self).__init__()
        # 卷积层1，32通道输出，卷积核大小3*3，步长1*1，padding为1
        self.Conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        # 最大值池化，核大小2*2，步长2*2
        self.pool1 = nn.MaxPool2d(2, 2)
        self.Conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.Conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 768)
        self.fc2 = nn.Linear(768, 10)

    def forward(self, x):
        list_num= [0.0, 0.49999999999999994, 0.0016330266955266947, -1.5104317985218285e-21, -2.7533034336419635e-09, -1.869600345373798e-26, 2.3990207248264032e-15, -5.257186895069533e-33, -9.844542811156595e-22, 2.5053871295601977e-39, 1.8396495301046406e-28, 7.3811070233906075e-47, -1.254306497798623e-35]
        b=torch.Tensor(list_num).cuda()

        x = self.Conv1(x)

        c=torch.zeros(x.size()).cuda()
        for i in range(b.view(1,-1).size()[1]):
            c = c + b[i]*(x**i)
        x=c
        #x = func.relu(x)
        x = self.pool1(x)

        x = self.Conv2(x)
        c=torch.zeros(x.size()).cuda()
        #print(c.size())
        for i in range(b.view(1,-1).size()[1]):
            c = c + b[i]*(x**i)
        x=c

        #print(x.size())
        #input("eeeeeeeeeeeeeeeeeeeeeeeee")
        #x = func.relu(x)
        x = self.pool2(x)

        x = self.Conv3(x)
        c=torch.zeros(x.size()).cuda()
        #print(c.size())
        for i in range(b.view(1,-1).size()[1]):
            c = c + b[i]*(x**i)
        x=c

        #print(x.size())
        #input("eeeeeeeeeeeeeeeeeeeeeeeee")
        #x = func.relu(x)
        x = self.pool3(x)
        x = x.view(-1, 128 * 3 * 3)

        x = self.fc1(x)
        
        ''''''
        c=torch.zeros(x.size()).cuda()
        for i in range(b.view(1,-1).size()[1]):
            c += b[i]*(x**i)
        x=c
        #x = func.relu(x)
        
        x = self.fc2(x)
        # softmax层，输出预测结果
        x = func.softmax(x, dim=1)  # 注意这里将给出BATCH_SIZE*10的矩阵
        return x

class pre_poly_MyCNN(nn.Module):
    def __init__(self):
        # 创建一个pytorch神经网络模型
        super(pre_poly_MyCNN, self).__init__()
        # 卷积层1，32通道输出，卷积核大小3*3，步长1*1，padding为1
        self.Conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        # 最大值池化，核大小2*2，步长2*2
        self.pool1 = nn.MaxPool2d(2, 2)
        self.Conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.Conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.sign=1
        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 768)
        self.fc2 = nn.Linear(768, 10)

    def forward(self, x):
        if self.sign:
            #print("eeeee")
            x = self.pool1(func.relu(self.Conv1(x)))
            x = self.pool2(func.relu(self.Conv2(x)))
            x = self.pool3(func.relu(self.Conv3(x)))
            x = x.view(-1, 128 * 3 * 3)
            x = func.relu(self.fc1(x))
            x = self.fc2(x)
            # softmax层，输出预测结果
            x = func.softmax(x, dim=1)  # 注意这里将给出BATCH_SIZE*10的矩阵
        else:
            #print("eedddddddddddseee")
            list_num= [0.0, 0.49999999999999994, 0.0016330266955266947, -1.5104317985218285e-21, -2.7533034336419635e-09, -1.869600345373798e-26, 2.3990207248264032e-15, -5.257186895069533e-33, -9.844542811156595e-22, 2.5053871295601977e-39, 1.8396495301046406e-28, 7.3811070233906075e-47, -1.254306497798623e-35]
            b=torch.Tensor(list_num).cuda()

            x = self.Conv1(x)

            c=torch.zeros(x.size()).cuda()
            for i in range(b.view(1,-1).size()[1]):
                c = c + b[i]*(x**i)
            x=c
            #x = func.relu(x)
            x = self.pool1(x)

            x = self.Conv2(x)
            c=torch.zeros(x.size()).cuda()
            #print(c.size())
            for i in range(b.view(1,-1).size()[1]):
                c = c + b[i]*(x**i)
            x=c

            #print(x.size())
            #input("eeeeeeeeeeeeeeeeeeeeeeeee")
            #x = func.relu(x)
            x = self.pool2(x)

            x = self.Conv3(x)
            c=torch.zeros(x.size()).cuda()
            #print(c.size())
            for i in range(b.view(1,-1).size()[1]):
                c = c + b[i]*(x**i)
            x=c

            #print(x.size())
            #input("eeeeeeeeeeeeeeeeeeeeeeeee")
            #x = func.relu(x)
            x = self.pool3(x)
            x = x.view(-1, 128 * 3 * 3)

            x = self.fc1(x)
            
            ''''''
            c=torch.zeros(x.size()).cuda()
            for i in range(b.view(1,-1).size()[1]):
                c += b[i]*(x**i)
            x=c
            #x = func.relu(x)
            
            x = self.fc2(x)
            # softmax层，输出预测结果
            x = func.softmax(x, dim=1)  # 注意这里将给出BATCH_SIZE*10的矩阵
        return x