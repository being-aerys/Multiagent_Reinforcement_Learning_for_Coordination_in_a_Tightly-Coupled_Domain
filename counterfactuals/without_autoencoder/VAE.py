import os
#---------------------------------------works good
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import numpy as np
import torch.utils.data as utils
import matplotlib.pyplot as plt


num_epochs = 20
batch_size = 64
learning_rate = 1e-3
img_transform = transforms.Compose([transforms.ToTensor()])    #-------------------------------------we do not want to normalize
training_loss_per_batch = 0
testing_loss_per_batch = 0
epoch_list_for_the_plot = []
training_loss_list = []
testing_loss_list = []
no_of_training_batches = 0
no_of_testing_batches = 0
myfilename = "data_4_10_20_new.txt"
f=open(myfilename,'r')
line = f.readline()
line_cnt = 1
states = []
temp_line=[]
temp_line_for_1_state = []
updating_string = []
count = 0

#---------------------------for ninja technique, we need this demo string
demo_string = "a"
updating_string.append(demo_string)


while line:

    demo_string = demo_string + line
    if ("]" in line):

        demo_string = demo_string.replace("[","") #-----------
        demo_string = demo_string.replace("\n","")
        demo_string = demo_string.replace("]","")
        demo_string = demo_string.replace("a","")

        updating_string[count] = demo_string.split()

        count = count + 1
        demo_string = "a"
        updating_string.append(demo_string)

    line = f.readline()

del updating_string[-1]     #removing the "a" added at the end of the loop above
states = updating_string
length_of_total_states = len(states)#-----------584980

training_states = states[:int(length_of_total_states * 0.9)]#----------526482
testing_states = states[int(length_of_total_states * 0.9):]#-----------58498

#------------------------------------Check if the data is inconsistent
for i in range(len(training_states)):

    if(len(training_states[i]) != 40):
        #print(len(states[i]))
        print("Training data error: Not all joint states have the same number of elements. Position ", i," has inconsistent data.")

for i in range(len(testing_states)):

    if(len(testing_states[i]) != 40):
        #print(len(states[i]))
        print("Testing data error: Not all joint states have the same number of elements. Position ", i," has inconsistent data.")

#--------------------------------------convert a list of lists into a list of numpy arrays to build a custom dataset for pytorch


def states_list(states_input):
    states_list_x = []
    states_list_y = []

    for item in range(len(states_input)):

        states_list_x.append(np.asarray(states_input[item], dtype=np.float32))

    for item in range(len(states_input)):#-------------------------------------------------works as lable, need to provide to make a data set
        states_list_y.append(np.asarray([0], dtype=np.float32))
    return states_list_x, states_list_y

training_states_list_x, training_states_list_y = states_list(training_states)
testing_states_list_x, testing_states_list_y = states_list(testing_states)
# print(length_of_total_states)

# no_of_training_batches = len(training_states_list_x)/batch_size
# no_of_testing_batches = len(testing_states_list_x)/batch_size
# print(no_of_training_batches)
# print(no_of_testing_batches)
#
# time.sleep(333)

training_tensor_x = torch.stack([torch.Tensor(i) for i in training_states_list_x]) # transform to torch tensors
training_tensor_y = torch.stack([torch.Tensor(i) for i in training_states_list_y])
testing_tensor_x = torch.stack([torch.Tensor(i) for i in testing_states_list_x]) # transform to torch tensors
testing_tensor_y = torch.stack([torch.Tensor(i) for i in testing_states_list_y]) # transform to torch tensors
#tensor_x is a list of lists.
#tensor_x[0] has a length of 40

custom_training_dataset = utils.TensorDataset(training_tensor_x, training_tensor_y) # create your datset
custom_training_dataloader = utils.DataLoader(dataset=custom_training_dataset,batch_size = batch_size,shuffle = True) # create your dataloader

custom_testing_dataset = utils.TensorDataset(testing_tensor_x, testing_tensor_y) # create your datset
custom_testing_dataloader = utils.DataLoader(dataset=custom_testing_dataset,batch_size = batch_size,shuffle = True)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # self.encoder = nn.Sequential(nn.Linear(len(states[0]),50),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(50, 30),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(30, 20),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(20, 10),
        #                              nn.LeakyReLU(),
        #                              # nn.Linear(64, 32),
        #                              # nn.LeakyReLU()  # ------------------------------------------Encoder ends here
        #
        #                              )

        # The encoder part of the auto-encoder. nn.Sequential joins several layers end to end
        # self.encoder = nn.Sequential(nn.Linear(len(training_states[0]), 1024),
        #                              nn.LeakyReLU(),
        #
        #                              nn.LeakyReLU(),
        #                              nn.Linear(256, 64),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(64, 10),
        #                              nn.LeakyReLU())


        #----------------------------Encoder stuff
        self.linear1 = nn.Linear(len(training_states[0]), 1024)
        self.relu1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(1024, 256)
        self.relu2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(256,64)
        self.relu3 = nn.LeakyReLU()
        self.linear4 = nn.Linear(64,10)
        self.relu4 = nn.LeakyReLU()


        # The decoder part of the auto-encoder. nn.Sequential joins several layers end to end
        # self.decoder = nn.Sequential(nn.Linear(10, 64),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(64, 256),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(256, 1024),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(1024, len(training_states[0])))

        #-----------------------------decoder stuff
        self.linear5 = nn.Linear(10, 64)
        self.relu5 = nn.LeakyReLU()
        self.linear6 = nn.Linear(64,256)
        self.relu6 = nn.LeakyReLU()
        self.linear7 = nn.Linear(256, 1024)
        self.relu7 = nn.LeakyReLU()
        self.linear8 = nn.Linear(1024,len(training_states[0]))




        # self.decoder = nn.Sequential(nn.Linear(10, 20),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(20, 30),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(30, 50),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(50, len(states[0])),
        #                              nn.LeakyReLU(),
        #                              # nn.Linear(512, len(states[0]))
        #                              # # ------------------------------------------Decoder ends here
        #
        #                              )

    def forward(self, x):

        x = self.linear1(x)
        x= self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        x = self.linear6(x)
        x = self.relu6(x)
        x = self.linear7(x)
        x = self.relu7(x)
        x = self.linear8(x)




        #
        # x = self.encoder(x)
        #                         #------------------------may be we can put tolerance stuff on latent dimensions
        # x = self.decoder(x)
        return x


model = AutoEncoder().cuda()
criterion = nn.L1Loss()#--------------Which loss to use, L1 or MSE, L1 loss makes more sense because we do not have outliers in our data
#by default reduce = None for Loss class so batch ko harek 40dim array ko loss ko sum matra aauxa
# tara for each element chai MAE across each element ko sum garera mean nai nikalera dinxa
#so loss for a batch = loss returned divided by batch size

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay=1e-5)

#print(len(custom_dataloader.dataset))#-----------------102464


print("\nTraining Starts here...")



for epoch in range(num_epochs):

    running_training_loss_cumulative_for_epoch = 0
    running_testing_loss_cumulative_for_epoch = 0

    for index,data in enumerate(custom_training_dataloader):
        features_of_a_training_batch, training_labels = data


        #print(len(features)) gives 64 which is the batch size
        #print(len(features[0])) gives 40 which is the dimension of the state space
        features_of_a_training_batch = Variable(features_of_a_training_batch).cuda()

        # ----------------------------------------------forward
        training_output = model(features_of_a_training_batch)
        training_output = training_output.cpu()



        #check how good the model was at the end of every epoch
        if(index == len(custom_training_dataloader)-1):
            print("For Training Data: \n")
            print("In the original data, 1st element of the last batch: ",features_of_a_training_batch[1])
            print("Predicted Values to compare: ",training_output[1])

        training_loss_per_batch = criterion(training_output, features_of_a_training_batch.cpu())#by default, gives the mean of each elements within a sample
        #------------------take care of the order of the output and the sample
        running_training_loss_cumulative_for_epoch = running_training_loss_cumulative_for_epoch + training_loss_per_batch

        # ----------------------------------------------backward
        optimizer.zero_grad()
        training_loss_per_batch.backward()
        optimizer.step()

    #-----------------------------Check how good you are doing after every epoch with testing data
    for index,data in enumerate(custom_testing_dataloader):
        features_of_a_testing_batch, labels = data

        features_of_a_testing_batch = Variable(features_of_a_testing_batch).cuda()
        # ----------------------------------------------forward
        testing_output = model(features_of_a_testing_batch)
        testing_output = testing_output.cpu()

        #check how good the model was at the end of every epoch
        if(index == len(custom_testing_dataloader)-1):
            print("For Testing Data:\n")
            print("In the original data, 1st element of the last batch: ",features_of_a_testing_batch[1])
            print("Predicted Values to compare: ",testing_output[1])


        testing_loss_per_batch = criterion(testing_output, features_of_a_testing_batch.cpu())#------------------take care of the order of the output and the sample
        running_testing_loss_cumulative_for_epoch = running_testing_loss_cumulative_for_epoch + testing_loss_per_batch

    epoch_training_loss = (running_training_loss_cumulative_for_epoch/len(training_states))
    epoch_testing_loss = (running_testing_loss_cumulative_for_epoch/len(testing_states))
    # print("epoch training loss",epoch_training_loss)
    # print("epoch testing loss",epoch_testing_loss)
    # print("epoch training loss", training_loss_list)
    # print("epoch testing loss", testing_loss_list)
    # # time.sleep(111)
    training_loss_list.append(epoch_training_loss)
    testing_loss_list.append(epoch_testing_loss)
    epoch_list_for_the_plot.append(epoch)


    print("epoch [{}/{}], ".format(epoch+1,num_epochs))
    print('Training loss:{:.4f}'.format(epoch_training_loss))
    print('Testing loss:{:.4f}'.format(epoch_testing_loss))

    # ----------------------------Append the values to the lists


    # ----------------------------Plot the results for each epoch
    plt.figure(1)  # ------------------------------------------------------------Mode = figure(1) for plt

    plt.plot(epoch_list_for_the_plot, training_loss_list, 'g')  # pass array or list
    plt.plot(epoch_list_for_the_plot, testing_loss_list, 'r')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.gca().legend(('Training Loss', 'Testing Loss'))
    plt.grid()

    plt.title("Number of Epochs VS Loss")


#torch.save(model.state_dict(), './autoencoder.pth')