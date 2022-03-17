import torch 
if __name__ == '__main__':
    num_gpu = torch.cuda.device_count()
    if num_gpu > 1:
        model = torch.nn.DataParallel(model)
        print("make DataParallel")
    else:
        print(f'Number of GPUs avaialble: {num_gpu}')