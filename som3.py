from ENPM808Y_HW3_Main import *


def genNet(row,col):
    return np.random.rand(row, col)

def euclidiean(selected,net):
    return np.linalg.norm(net - selected, axis=1)

def localize(center,sigma,domain):
    if sigma < 1:
        sigma = 1

    d= np.minimum( np.absolute(center - np.arange(domain)), domain - np.absolute(center - np.arange(domain)))
    Bij=np.array([np.exp(-(d*d) / (2 * (sigma*sigma)))]).T

    return Bij




def som3Main(input, epoch, closeMethod,lr,pivot,sigma):

    n = input.shape[0]*pivot
    print('Number of Neurons: ',n)
    net=genNet(n,input.shape[1])



    for i in range(epoch):

        selected=input[np.random.randint(0,input.shape[0])]

        if closeMethod=='euclidiean':
             eucl=euclidiean(selected,net)
             minVal=min(eucl)
             minNdx = eucl.argmin()


        localGuass=localize(minNdx,n//sigma,net.shape[0])

        net += lr*localGuass * (selected - net)

        lr = lr * 0.9999
        n = n * 0.999

        if n < 1:
            print("Radius decayed")
            break

        if lr < .00001:
            print("Learning Rate decayed")
            break

    print('Number of times trained before fully decayed: ',i)

    return net


