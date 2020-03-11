from workNet import R_net
import train

if __name__ == '__main__':
    net=R_net()
    trainner=train.Trainer(net,"./module/rnet.pt",r"F:\24")
    trainner.train()