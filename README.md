# PyTorch_learn
Practical Deep Learning with PyTorch



# Key points or useful commands


    torch.tensor(arr)  # Pass in an array or list of lists(matrix) to convert it into a float tensor
    torch.rand(2,2)  # Get random values in 2X2 matrix
    torch.manual_seed(42)  # For cpu
    
    
    if torch.cuda.is_available():  # For gpu
        torch.cuda.manual_seed_all(42)


# Torch to numpy bridge


Torch cannot convert all types of numpy arrays to tensor. The only supported datatypes are 
* double
* float
* int64(Long)
* int32(Int)
* uint8(byte)
So if the dtype of your numpy array is np.int8, then you'll get an error  


        torch.from_numpy(np_array)  # This will convert the nd array into torch.DoubleTensor by default

        torch_tensor.numpy() # Converts the torch tensor to a numpy nd array

# CPU to GPU

When you use gpu to accelerate your model, every tensor must be on gpu

        tensor_cpu.cuda() # Moves to gpu
        tensor_cpu.cpu()  # Moves to cpu. This is the default 
      
      
# Basic math

Reshaping

        a = torch.ones(2,2)
        a.view(4)  # makes it a torch tensor of size 4. This can be used to reshape the tensor
        
  Element-wise addition and sub
        
        c=a+b
        torch.add(a,b)  # Both are similar
        c.add_(a)  #  This is inplace addition
        
        a-b
        a.sub(b)  # Both are similar
        a.sub_(a)  #  This is inplace subtraction and will modify the original tensor
        
        c = a*b
        c = torch.mul(a,b)
        a.mul_(b)
        
        c = a/b
        c = torch.div(a,b)
        a.div_(b)
        
        a= torch.Tensor([1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10])
        a.mean(dim=0)
        a.mean(dim=1)
        a.std(dim=0)
        
        
  **_ after the operation means an inplace modification  **     



# Variables and Gradients

Variable behavior is almost the same as the Tensor, like you read above. Instead of passing torch tensors we can pass variables to the methods. The only difference is how we accumulate gradients in Variable. 

* **Backward should only be called on scalar(one element tensor) or with gradient w.r.t variable**. So we usually take the mean or sum the expression to create a scalar and use backward on that. Like cost function.

        from torch.autograd import Variable
        a = Variable(torch.ones(2,2),requires_grad = True)
    
    

    
 # Useful code tips
 
 
* **Create Model Class:** It needs to inherit from the nn.Module, so we use super. Each model needs a forward method
 
         import torch.nn as nn
         class LinearRegressionModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(LinearRegressionModel, self).__init__()
                self.linear = nn.Linear(input_dim, output_dim)  

            def forward(self, x):
                out = self.linear(x)
                return out
    
* After instantiating this class as model set loss and optimizer

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
* **Make sure to re-initialize the gradients each epoch during training**

        optimizer.zero_grad() 
    
* In the training loop : Variables are passed. The values in variables are accessed using **.data** and we do **loss.backward** to get the gradients and also update the step in optimizer for each epoch
    
        # Calculate Loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        print('epoch {}, loss {}'.format(epoch, loss.data[0]))
        
 * **Save and load :** To be memory efficient we save the parameters of the parameters
 
        save_model = False
        if save_model is True:
            # Saves only parameters
            # alpha & beta
            torch.save(model.state_dict(), 'awesome_model.pkl')
  
        load_model = False
        if load_model is True:
            model.load_state_dict(torch.load('awesome_model.pkl'))
            
            

* **To use GPU for model** : If you have a variable in your code put it on gpu

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        or simply 
        
        if torch.cuda.is_available():
            model.cuda()

*Check if cuda is available and also add all tensors inside the variable on cuda using the .cuda() as both model and Variables need to be on gpu*


* **Aim: make the dataset iterable**
    - **totaldata**: 60000
    - **minibatch**: 100
        - Number of examples in 1 iteration
    - **iterations**: 3000
        - 1 iteration: one mini-batch forward & backward pass
    - **epochs**
        - 1 epoch: running through the whole dataset once
        - num_epochs = iterations / (totaldata/minibatch) = 3000/{60000}/{100} = 5 


                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                       batch_size=batch_size, 
                                                       shuffle=True)
                # Check if it is actaully an iterable
                import collections
                isinstance(train_loader, collections.Iterable)



# CNN

Padding Summary:

- Valid Padding (Zero Padding) : Output size < Input Size  
- Same Padding : Output size = Input Size  

Dimension Calculations

    O=((W−K+2P)/S)+1
    
O : output height/length   
W : input height/length   
K : filter size (kernel size)   
P : padding  
P= (K−1)/2  
S : stride  

For pooling

    O=W/K


# RNN

An RNN is just the same as a feed forward nn with a linear function between them . You'll need to create separate variables in the forward part of the model for the cell state and hidden layers. 28 timesteps means our feed forward nn repeats itself 28 times and each time our input dimension is 28. Unlike cnn where we give 28* 28 here at each time step we only feed in 28 pixels. so input_dim is 28  

RNN Input: (1, 28)  
CNN Input: (1, 28, 28)  
FNN Input: (1, 28*28)   

         # batch_first=True causes input/output tensors to be of shape
         # Output of RNN cell (batch_dim, seq_dim, input_dim)
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
         
         # Initialize hidden state with zeros
         # (layer_dim, batch_size, hidden_dim) where layer_dim is number of hidden layers
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
         # The out would have all 28(MNIST) timesteps
        out, hn = self.rnn(x, h0)
        
         # Index hidden state of last time step
         # out.size() --> 100, 28, 10
         # out[:, -1, :] --> 100, 10 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
         # out.size() --> 100, 10
