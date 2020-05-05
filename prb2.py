from ENPM808Y_HW3_Main import *
import re
import ast
np.set_printoptions(suppress=True)

def prepData(data):
    f = open(data, "r")

    count = 0
    dataOut = []
    for row in f:

        if count == 0:
            count = count + 1

        else:
            row = re.sub('\s+', ',', row)

            row = list(ast.literal_eval(row))

            count = count + 1

            dataOut.append(row)

    dataOut=np.asarray(dataOut)

    return dataOut


def prb2main():
    wineDesired = prepData('Wine Data/Wine Desired.asc')


    wineInput = prepData('Wine Data/Wine Input.asc')


    PercentTrain=.8


    indexLin = np.arange(0, len(wineInput), 1)
    indexshuf = np.random.shuffle(indexLin)

    trainNdx = indexLin[:int(len(indexLin) * PercentTrain)]
    trainI = wineInput[trainNdx]
    trainO = wineDesired[trainNdx]

    testNdx = indexLin[int(len(indexLin) * PercentTrain):]
    testI = wineInput[testNdx]
    testO = wineDesired[testNdx]


    #   IRIS data for comparaison
    data = np.genfromtxt('iris.csv', delimiter=',', usecols=(0, 1, 2, 3))
    # data normalization
    data = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, data)

    data=data
    # fig, ax = plt.subplots(7, 2,sharey=True,sharex=True,squeeze=False)

    # for i in range(trainI.shape[0]):
    # cluster,net = som2Main(input=trainI, epoch=10000, eta=.5,pivot=4,sigma=5)
    # eta = .5, pivot = 2, sigma = 5

    cluster,net=som2MainB(input=trainI, epoch=10000, lr=.7, pivot=5, sigma=7)
    # input = data, epoch = 10000, lr = .7, pivot = 2, sigma = 5 IRIS
    # input = trainI, epoch = 10000, lr = .7, pivot = 5, sigma = 20
    norm = np.linalg.norm(cluster)
    cluster = cluster / norm

    # # print(clusterNorm)
    plt.figure(figsize=(7, 7))
    plt.imshow(cluster, cmap='plasma',interpolation='nearest')
    plt.show()

    dmap=neuronDiff(net)

    plt.figure(figsize=(7, 7))
    plt.pcolor(dmap.T, cmap='bone_r')


    # target = np.genfromtxt('iris.csv', delimiter=',', usecols=(4), dtype=str)
    # t = np.zeros(len(target), dtype=int)
    # t[target == 'Setosa'] = 0
    # t[target == 'Versicolor'] = 1
    # t[target == 'Virginica'] = 2


    # use different colors and markers for each label
    # markers = ['d', 's', 'D']
    # colors = ['C0', 'C1', 'C2']
    # for cnt, data in enumerate(data.T):
    #     y,x,_ = activate(data,net)  # getting the winner
    #     # place a marker on the winning position for the sample xx
    #
    #     plt.plot(x + .5, y + .5, markers[t[cnt]], markerfacecolor='None',
    #                           markeredgecolor=colors[t[cnt]], markersize=12, markeredgewidth=2)
    # plt.axis([0, 8, 0, 8])

    markers = ['d', 's', 'D']
    colors = ['C0', 'C1', 'C2']
    for cnt, data in enumerate(trainI):
        y,x,_ = activate(data,net)  # getting the winner
        # place a marker on the winning position for the sample xx
        if trainO[cnt][2]==1:
            DESG=0
        if trainO[cnt][1]==1:
            DESG=1
        if trainO[cnt][0]==1:
            DESG=2
        plt.plot(x + .5, y + .5, markers[DESG], markerfacecolor='None',
                              markeredgecolor=colors[DESG], markersize=12, markeredgewidth=2)
    plt.axis([0, net.shape[0], 0, net.shape[1]])
    plt.show()

    # plt.figure(figsize=(7, 7))
    # plt.pcolor(dmap.T, cmap='bone_r')
    # markers = ['d', 's', 'D']
    # colors = ['C0', 'C1', 'C2']
    # for cnt, data in enumerate(testI):
    #     y, x, _ = activate(data, net)  # getting the winner
    #     # place a marker on the winning position for the sample xx
    #     if testI[cnt][2] == 1:
    #         DESG = 0
    #     if trainO[cnt][1] == 1:
    #         DESG = 1
    #     if trainO[cnt][0] == 1:
    #         DESG = 2
    #     plt.plot(x + .5, y + .5, markers[DESG], markerfacecolor='None',
    #              markeredgecolor=colors[DESG], markersize=12, markeredgewidth=2)
    # plt.axis([0, net.shape[0], 0, net.shape[1]])
    # plt.show()
