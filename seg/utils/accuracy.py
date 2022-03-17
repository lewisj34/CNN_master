import numpy as np

def accuracy(gt, pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(pred)): 
        if gt[i]==pred[i]==1:
           TP += 1
        if pred[i]==1 and gt[i]!=pred[i]:
           FP += 1
        if gt[i]==pred[i]==0:
           TN += 1
        if pred[i]==0 and gt[i]!=pred[i]:
           FN += 1

    ACC = (TP + TN) / (TP + TN + FP + FN)

    return(ACC, TP, FP, TN, FN)

if __name__ == "__main__":
    masks = np.random.rand(3,3)
    res = np.random.rand(3,3)
    masks = 1*(masks > 0.5)            
    res = 1*(res > 0.5)

    print(masks, "\n", res)
    print(accuracy(res.flatten(), masks.flatten()))