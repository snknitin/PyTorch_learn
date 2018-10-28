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
        
        
        c = a+b
        c = torch.add(a,b)  # Both are similar
        a.add_(b)  #  This is inplace addition
        
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

Variable behavior is almost hte same as the Tensor, like you read above.Instead of passing torch tensors we can pass vvariables to the methods. The only difference is how we accumulate graddients in Variable

    from torch.autograd import Variable
    a = Variable(torch.ones(2,2),requires_grad = True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
