from workNet import O_net
import train

if __name__ == '__main__':
    net=O_net()
    trainner=train.Trainer(net,"./module/onet1.pt",r"F:\48")
    trainner.train()
