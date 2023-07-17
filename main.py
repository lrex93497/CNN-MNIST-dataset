#####################
##fisnished @ 14:05 25/10/2022
#####################
import torch
from torch import nn, optim
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=12, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=64, kernel_size=5, stride=1, padding=2)  #padding to 4 slide = 2
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(5*5*64, 1024)      ##5*5*64 = 1600
        self.fc2 = nn.Linear(1024, 10)    ##1024 ->10

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.flatten(x)     #flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def acc_test(test_loader, model, batch_size):
    model.eval()
    right_no = 0
    test_sample_amount = 0
    correct_no_this_loop = 0
    with torch.no_grad():
        for i, v in test_loader:
            i = i.to(device = device_to_use)
            v = v.to(device = device_to_use)
            
            result = model(i)
            _, predict = result.max(1)

            predict = predict.tolist()
            actual = v.tolist()

            for j,u in zip(predict,actual):
                if int(j) == int(u):
                    correct_no_this_loop = correct_no_this_loop + 1

            right_no = right_no + correct_no_this_loop
            test_sample_amount = test_sample_amount + batch_size
            correct_no_this_loop = 0
        percentage_correct = right_no/test_sample_amount *100
        print("accuracy -> correct precentage = " + str(percentage_correct) + "%/100%, " + str(right_no) + "/" + str(test_sample_amount))

def show_first_c_layer_filter(model):
    target_kernels = model.conv1.weight.detach().cpu().clone()      ## need cpu as i am using cuda, plt cannot use cuda

    target_kernels = target_kernels - target_kernels.min()
    target_kernels = target_kernels / target_kernels.max()
    kernel_to_show = torchvision.utils.make_grid(target_kernels, nrow = 5)  ##5x5 =25 filter
    ##plt.imshow(kernel_to_show.permute(1,2,0))
    torchvision.utils.save_image(kernel_to_show, 'output_filters.png' ,nrow = 5)
    print("25 filters of first convolutional layer were saved to output_filters.png at root")


if __name__ == "__main__":
    device_to_use = torch.device("cuda" if torch.cuda.is_available() else "cpu")     # I use 3060m

    learning_rate = 0.0001
    batch_size = 50         ##60000/50 = 1200  ,60k data point
    number_epoch = 4        ##1200x4 = 4.8k   , 4.8k iteration


    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

    train_set_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)      #shuffle data
    test_set_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)       #shuffle data

    model = CNN().to(device_to_use)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)        #10e-4 = 0.0001

    for i in range(number_epoch):
        for v, (image, label) in enumerate(train_set_loader):       ##ima
            #push to gpu cuda
            image = image.to(device_to_use)
            label = label.to(device_to_use)

            #forward
            output = model(image)
            loss = loss_function(output, label)

            #Adam optimizer and set gradient to zero
            optimizer.zero_grad()     
            loss.backward()
            optimizer.step()

            if v % 100 == 0 or v == 1199:
                print(str(v) + ", Loss: " + str(loss.item()))
        print("Epoch: "+ str(i+1) + "/" + str(number_epoch) + "completed\n")
        
    print("*******training complete*******\n")

    acc_test(test_set_loader, model, batch_size)

    show_first_c_layer_filter(model)