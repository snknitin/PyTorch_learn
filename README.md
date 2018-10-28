# PyTorch_learn
Practical Deep Learning with PyTorch



# Key points or useful commands


    torch.tensor(arr)  # Pass in an array or list of lists(matrix) to convert it into a float tensor
    torch.rand(2,2)  # Get random values in 2X2 matrix
    torch.manual_seed(42)  # For cpu
    
    
    if torch.cuda.is_available():  # For gpu
        torch.cuda.manual_seed_all(42)
