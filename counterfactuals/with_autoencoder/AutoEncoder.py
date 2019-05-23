# Created by Aashish. Modified by Ashwin Vinoo
# This file implements the auto-encoder that can compresses the state data of the neural network
# Directions:
# Press and hold 's' and 'd' simultaneously to save the model in the location specified by 'model_name_to_save'
# Press and hold 'x' and 'c' simultaneously to save the model in a custom location

# Importing all the necessary modules
import torch
from matplotlib.ticker import MaxNLocator
from torch import nn
from torch.autograd import Variable
import torch.utils.data as utils
import matplotlib.pyplot as plot
import keyboard
import time
import numpy as np
import math

# ----------- Hyper Parameters -----------
# The number of epochs to train
num_epochs = 250
# The batch size is 64
batch_size = 64
# The learning rate that we are using for training
learning_rate = 0.001
# The weight decay to be used for the optimizer
weight_decay = 0.00001
# The momentum to be used during backward propagation
momentum = 0.5
# The training dataset
dataset_txt_file = '../data/ddpg_1_1_20.txt'
# This specifies how to split the dataset into testing and training datasets
train_test_ratio = 0.8
# The optimizers to use ('ADAM', 'ADAGRAD', 'RMSPROP', 'SGD')
optimizer_to_use = 'ADAM'
# The loss criterion to use ('L1, 'MSE')
criterion_to_use = 'L1'
# The name of the trained model to be saved
model_name_to_save = 'trained_network_1.pth'
# The model to load if load_trained_model is true
model_name_to_load = 'trained_network_1.pth'
# Controls whether we should load the weights and biases of the current neural network from a .pth file
load_trained_model = False
# The data type of the tensor to use
tensor_data_type = np.float32
# ----------------------------------------


# ----------- AutoEncoder class definition -----------
# The auto-encoder class definition. Inherits from nn.Module
class AutoEncoder(nn.Module):

    # The class constructor
    def __init__(self, rover_state_size):
        # We call the class constructor of the parent class Module
        super(AutoEncoder, self).__init__()
        # The encoder part of the auto-encoder. nn.Sequential joins several layers end to end
        self.encoder = nn.Sequential(nn.Linear(rover_state_size, 1024),
                                     nn.LeakyReLU(),
                                     nn.Linear(1024, 256),
                                     nn.LeakyReLU(),
                                     nn.Linear(256, 64),
                                     nn.LeakyReLU(),
                                     nn.Linear(64, 10),
                                     nn.LeakyReLU())

        # The decoder part of the auto-encoder. nn.Sequential joins several layers end to end
        self.decoder = nn.Sequential(nn.Linear(10, 64),
                                     nn.LeakyReLU(),
                                     nn.Linear(64, 256),
                                     nn.LeakyReLU(),
                                     nn.Linear(256, 1024),
                                     nn.LeakyReLU(),
                                     nn.Linear(1024, rover_state_size))

    # This function is the forward pass of the auto-encoder neural network
    def forward(self, x):
        # This is the encoder section
        x = self.encoder(x)
        # This is the decoder section
        x = self.decoder(x)
        # Returns the decoded result
        return x

    # This function can be used to load a model
    def load_model(self, model_location):
        # Loads the pretrained dictionary
        pretrained_dict = torch.load(model_location)
        # The dictionary for the current neural network
        current_net_dict = self.state_dict()
        # Iterating through the key-value pairs in the dictionary
        for key, value in pretrained_dict.items():
            # Copies the values from the pretrained dictionary to the current model
            current_net_dict[key] = pretrained_dict[key]
        # Loads the data in the current network dictionary
        self.load_state_dict(current_net_dict)

    # This is a function to evaluate the neural network
    def evaluate_network(self, data_loader, loss_criterion):
        # initializing the total evaluation loss to be zero
        total_evaluation_loss = 0
        #count the number of total samples in the whole dataloader
        total_samples = 0
        # Making the network to be in evaluation mode so that losses won't be accumulated, thereby saving memory
        self.eval()


        #-------------------------------------
        #list of loss  or each batch for the plot
        list_loss_of_batch = []

        #----------------------------------


        # Iterating through the data is data_loader
        for index, data in enumerate(data_loader):
            # Splitting into states and labels
            states, labels = data
            samples_per_batch = (labels.size(0))
            total_samples = total_samples + samples_per_batch
            # Converting to nn variable. cuda() creates another Variable that isn’t a leaf node in the computation graph
            states, labels = Variable(states).cuda(), Variable(labels).cuda()
            # The output obtained from the neural network
            outputs = self(states)
            # The loss is evaluated based on the cross entropy loss criteria
            sum_of_loss_of_all_elements_in_a_batch = loss_criterion(outputs, states)

            #-----------------------append the loss of the batch for mean and std dev
            list_loss_of_batch.append(sum_of_loss_of_all_elements_in_a_batch.data.item())

            # We get the loss for the current batch
            total_evaluation_loss += sum_of_loss_of_all_elements_in_a_batch.data.item()


            # In case we are at the last batch
            if index == len(data_loader) - 1:
                # We print the first state and outputs of the last batch for comparison
                print("In the original data, 1st element of the last batch: ", states[0])
                print("Predicted Values to compare: ", outputs[0])
        # Converting the network back into training mode
        self.train()
        # Returns the average loss across the evaluation dataset
        #print(total_evaluation_loss,total_samples)
        #time.sleep(334)
        list_loss_of_batch = [x / batch_size for x in list_loss_of_batch]
        #----------------------


        mean_loss_of_the_batches = np.mean(np.array(list_loss_of_batch))
        std_loss_of_the_batches = np.std(np.array(list_loss_of_batch))

        return total_evaluation_loss / total_samples, mean_loss_of_the_batches, std_loss_of_the_batches

    @staticmethod
    # This function reads a text file containing the states information
    def states_from_text_file(txt_file, data_type):
        # We use with statement while opening the file as it automatically handles file closing
        with open(txt_file, 'r') as file_read_handle:
            # We initialize the states to an empty list
            states_list = []
            # The states string is initialized to ''
            states_string = ''
            # The while loop continues as long as there a line to read
            for line in file_read_handle:
                # We add the line to the states string
                states_string += line
                # In case a left facing closing bracket is detected
                if ']' in line:
                    # We replace the characters '[', '\n', ']' with '' to facilitate splitting
                    states_string = states_string.replace('[', '')
                    states_string = states_string.replace('\n', '')
                    states_string = states_string.replace(']', '')
                    # We append the split up strings into the states
                    states_list.append(states_string.split())
                    # We initialize the states string back to ''
                    states_string = ''
        # We obtain the rover state size
        rover_state_size = len(states_list[0])
        print("Total states in the data: ",len(states_list))
        # Returns the states array converted to the desired data type
        return np.array(states_list, dtype=data_type), rover_state_size

    @staticmethod
    # This function can be used to split a dataset into train and test data loaders
    def create_train_test_datasets(states_list, data_type=np.float32, train_test_ratios=0.8, batch_size_for_loader=64):
        # We obtain the size of the overall dataset
        size_dataset = len(states_list)
        # We obtain the training dataset size
        training_set_size = math.floor(size_dataset * train_test_ratios)
        # We obtain the test dataset size
        testing_set_size = size_dataset - training_set_size

        # We obtain the states and labels of the training dataset
        states_train = np.array(states_list[0:training_set_size], dtype=data_type)
        labels_train = np.zeros((training_set_size, 1), dtype=data_type)

        # We obtain the states and labels of the testing dataset
        states_test = np.array(states_list[training_set_size:size_dataset], dtype=data_type)
        labels_test = np.zeros((testing_set_size, 1), dtype=data_type)

        # We obtain the tensors for the train dataset
        tensor_states_train = torch.tensor(states_train)
        tensor_labels_train = torch.tensor(labels_train)

        # We obtain the tensors for the test dataset
        tensor_states_test = torch.tensor(states_test)
        tensor_labels_test = torch.tensor(labels_test)

        # We create the training dataset
        train_dataset = utils.TensorDataset(tensor_states_train, tensor_labels_train)
        # We create the training dataset
        test_dataset = utils.TensorDataset(tensor_states_test, tensor_labels_test)
        # We create a data loader to load the shuffled training dataset in batches
        train_dataset_loader = utils.DataLoader(dataset=train_dataset, batch_size=batch_size_for_loader, shuffle=True)
        # We create a data loader to load the shuffled training dataset in batches
        test_dataset_loader = utils.DataLoader(dataset=test_dataset, batch_size=batch_size_for_loader, shuffle=True)

        # We return the train and test data loaders
        return train_dataset_loader, test_dataset_loader



    def load_weights(self):
        ##### loading weights from pretrained model ##########
        pretrained_dict = torch.load('DDPG_1_agent_250_epochs.pth')  # filename
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # 3. load the new state dict
        self.load_state_dict(model_dict)


# If this file is the main one called for execution
if __name__ == "__main__":



    '''
    # ---------- Read the data from the text file and obtain the data loaders ----------
    # We obtain the states by reading the text file
    states, state_size = AutoEncoder.states_from_text_file(dataset_txt_file, tensor_data_type)
    # We split the above states into that for training and testing. Data loaders for both of them are returned
    train_data_loader, test_data_loader = AutoEncoder.create_train_test_datasets(states, tensor_data_type,
                                                                            train_test_ratio, batch_size)
                                                                            '''

    # ---------- Creates the model, loads the loss function and optimizer ----------
    # We create a model
    state_size = length(state)
    model = AutoEncoder(state_size).cuda()
    # We convert the auto-encoder to training mode

    model.load_weights() #loads the pretrained model



    epoch_testing_loss, epoch_testing_batch_mean, epoch_testing_batch_std = model.evaluate_network(test_data_loader,
                                                                                                   criterion)

    #model.train()
    # Check if we wanted to load the trained model
    if load_trained_model:
        # Loads the pretrained dictionary
        model.load_model(model_name_to_load)

    # Choosing the different types of loss criterion available
    if criterion_to_use == 'L1':
        # We use L1 loss as the loss criterion (absolute difference between the predicted value and actual value)
        criterion = nn.L1Loss(size_average=False)#size_average overrides reduction
    else:
        # We use the mean square error as the loss
        criterion = nn.MSELoss(size_average=False)

    # Choosing the different types of optimizers available
    if optimizer_to_use == 'ADAM':
        # The optimizer to use during training is Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_to_use == 'RMSPROP':
        # The optimizer to use during training is RMSprop
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum,
                                        weight_decay=weight_decay)
    elif optimizer_to_use == 'ADAGRAD':
        # The optimizer to use during training is Adagrad
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        # The optimizer to use during training is stochastic gradient descent
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # ----------- We train the neural network here ----------
    # We initialize a flag to help regulate saving of the model
    save_flag = True
     #To plot the training loss over epoch graph, we need a list to store the loss at each epoch
    #training_loss_list = []
    # To plot the testing loss over epoch graph, we need a list to store the loss at each epoch
    testing_loss_list = []


    training_batch_mean = []
    training_batch_std = []
    testing_batch_mean = []
    testing_batch_std = []
     We iterate through the specified number of epochs
    for epoch in range(num_epochs):
        # We enumerate through the data loader for the training data
        for index, data in enumerate(train_data_loader):
            # We obtain the states and labels of the batch
            batch_states, batch_labels = data
            # cuda() creates another Variable that isn’t a leaf node in the computation graph
            batch_states = Variable(batch_states).cuda()
            # The forward pass of the auto-encoder
            output = model(batch_states)
            # The loss is measured between the output of the decoder and the input to the encoder
            loss = criterion(output, batch_states)
            # Zeros the gradients calculated previously
            optimizer.zero_grad()
            # Computes the gradients through back-propagation
            loss.backward()
            # proceeds with the gradient descent and changes the weights and biases
            optimizer.step()

            # In case keyboard 's' is pressed
            if keyboard.is_pressed('s') and keyboard.is_pressed('d') and save_flag:
                # Save flag is marked as False
                save_flag = False
                # We save the the auto-encoder model
                torch.save(model.state_dict(), model_name_to_save)
                # We display that we saved the model
                print('model saved to \'' + model_name_to_save + '\'')

            # In case keyboard 's' is pressed
            if keyboard.is_pressed('x') and keyboard.is_pressed('c') and save_flag:
                # Save flag is marked as False
                save_flag = False
                # We ask the user to enter the location to save the model
                custom_save_location = input('Enter the location to save the model: ')
                # We save the the auto-encoder model
                torch.save(model.state_dict(), custom_save_location)
                # We display that we saved the model
                print('model saved to \'' + custom_save_location + '\'')

            # This helps to prevent saving the model multiple times accidentally
            if (not save_flag and not (keyboard.is_pressed('s') and keyboard.is_pressed('d')) and not
                    (keyboard.is_pressed('x') and keyboard.is_pressed('c'))):
                # Save flag is marked as True
                save_flag = True

        # we get the loss on the training dataset for the current epoch
        epoch_training_loss, epoch_training_batch_mean, epoch_training_batch_std = model.evaluate_network(train_data_loader, criterion)
        # we get the loss on the testing dataset for the current epoch
        epoch_testing_loss, epoch_testing_batch_mean, epoch_testing_batch_std = model.evaluate_network(test_data_loader, criterion)
        # Append the training loss for this epoch to the training loss list
        training_loss_list.append(epoch_training_loss)
        training_batch_mean.append(epoch_testing_batch_mean)
        training_batch_std.append(epoch_training_batch_std)
        testing_batch_mean.append(epoch_testing_batch_mean)
        testing_batch_std.append(epoch_training_batch_std)
        # Append the testing loss for this epoch to the testing loss list
        testing_loss_list.append(epoch_testing_loss)

        # ----------- Display relevant information ----------
        # We print the epoch number and the loss on the test dataset after each epoch
        # print('epoch [{}/{}], loss on test dataset:{:.4f}, loss on train dataset:{:.4f}'.format(epoch+1, num_epochs,
        #     model.evaluate_network(test_data_loader, criterion), model.evaluate_network(train_data_loader, criterion)))

        print("Epoch [{}/{}], ".format(epoch + 1, num_epochs))
        print('Training loss:{:.4f}'.format(epoch_training_loss))
        print('Testing loss:{:.4f}'.format(epoch_testing_loss))

    # We do a line plot of the training dataset loss over epochs of training
    plot.plot(list(range(1, num_epochs+1)), training_loss_list, color='green')
    # We do a line plot of the testing dataset loss over epochs of training
    plot.plot(list(range(1, num_epochs+1)), testing_loss_list, color='red')

    ax = plot.gca()


    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


    training_loss_list = np.array(training_loss_list)
    testing_loss_list = np.array(testing_loss_list)
    training_batch_std = np.array(training_batch_std)
    training_batch_mean = np.array(training_batch_mean)
    testing_batch_std = np.array(testing_batch_std)
    testing_batch_mean = np.array(testing_batch_mean)


    training_loss_list = training_loss_list.tolist()
    testing_loss_list = testing_loss_list.tolist()
    training_batch_mean = training_batch_mean.tolist()
    training_batch_std = training_batch_std.tolist()
    testing_batch_mean = testing_batch_mean.tolist()
    testing_batch_std = testing_batch_std.tolist()



    # Plot accurancy bands for training and test sets
    plot.fill_between(list(range(1, num_epochs+1)), np.subtract(training_loss_list,training_batch_std), np.add(training_batch_mean, training_batch_std), facecolor='green', alpha=0.1)
    plot.fill_between(list(range(1, num_epochs+1)), np.subtract(testing_loss_list, testing_batch_std), np.add(testing_loss_list, testing_batch_std), facecolor='red', alpha=0.1)

    # We mark the label on the x-axis
    plot.xlabel("Number of Epochs")
    # We mark the label on the y-axis
    plot.ylabel(criterion_to_use + ' Loss')
    # The legend is marked
    plot.gca().legend(('Training Loss', 'Testing Loss'))
    # The plot grid is highlighted
    plot.grid()
    # The plot title is displayed
    plot.title("Training and Testing Losses across Epochs")
    # We plot the image
    plot.show()

    # We save the the auto-encoder model

    torch.save(model.state_dict(), model_name_to_save)
    
# ----------- End of Program ----------
