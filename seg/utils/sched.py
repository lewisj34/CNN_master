class WarmupPoly(object):
    '''
    CLass that defines cyclic learning rate that decays the learning rate linearly till the end of cycle and then restarts
    at the maximum value.
    '''
    def __init__(self, init_lr, total_ep, warmup_ratio=0.05, poly_pow = 0.98):
        super(WarmupPoly, self).__init__()
        self.init_lr = init_lr
        self.total_ep = total_ep
        self.warmup_ep = int(warmup_ratio*total_ep)
        print("warup unitl " + str(self.warmup_ep))
        self.poly_pow = poly_pow

    def get_lr(self, epoch):
        #
        if epoch < self.warmup_ep:
            curr_lr =  self.init_lr*pow((((epoch+1) / self.warmup_ep)), self.poly_pow)

        else:
            curr_lr = self.init_lr*pow((1 - ((epoch- self.warmup_ep)  / (self.total_ep-self.warmup_ep))), self.poly_pow)

        return curr_lr

