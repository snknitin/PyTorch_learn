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
