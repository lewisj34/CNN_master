
# following two functions return the exact same amount of number of total params 

def netParams(model):
    '''
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p
        # print(total_paramters)
    print(f'Total Parameters: {total_paramters}')
    return total_paramters

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    # print(table)
    print(f"Total Trainable Params: {total_params/1000000}M")
    return total_params
    
if __name__ == '__main__':
    from z import EffNet_B3, EffNet_B4, EffNet_B7
    from seg.model.CNN.CNN import CNN_BRANCH
    count_parameters(EffNet_B3())
    count_parameters(EffNet_B4())
    count_parameters(EffNet_B7())
    count_parameters(CNN_BRANCH(3, 1, 16))