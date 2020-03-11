from workNet import Pnet
import train

if __name__ == '__main__':
    net=Pnet()
    trainner=train.Trainer(net,"./module/pnet.pt",r"F:\12\12")
    trainner.train()