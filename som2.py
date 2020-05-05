from ENPM808Y_HW3_Main import *
import math


def genNet(row, col, z):
    return np.random.rand(row, col, z)


def euclidiean(selected, net):
    return np.linalg.norm(net - selected, axis=2)


def localize2D(y, x, sigma, nX, nY):
    if sigma < 1:
        sigma = 1

    d = 2 * 3.14 * sigma * sigma

    ax = np.exp(-np.power(nX - x, 2) / d)
    ay = np.exp(-np.power(nY - y, 2) / d)
    Bij = np.outer(ax, ay)

    # d= np.minimum( np.absolute(center - np.arange(domain)), domain -  np.absolute(center - np.arange(domain)))
    # Bij=np.array([np.exp(-(d*d) / (2 * (sigma*sigma)))]).T

    return Bij


def localize(center, sigma, domain, net):
    if sigma < 1:
        sigma = 1

    delta_array = np.empty((15, 15))
    for i in range(15):
        for j in range(15):
            delta_array[j][i] = math.sqrt((center[0] - i) ** 2 + (center[1] - j) ** 2)
    d = delta_array
    this = -(d * d) / (2 * (sigma * sigma))
    return np.exp(this)


def activate(selected, net):
    eucl = euclidiean(selected, net)
    # minVal=min(eucl)
    # minNdx = eucl.argmin()
    minNdx = np.where(eucl == np.min(eucl))

    y = minNdx[0][0]
    x = minNdx[1][0]

    return y, x, eucl


def deactivate(y, x, sigma, neighborsX, neighborsY, eta, net, selected):
    localGuass = localize2D(y, x, sigma, neighborsX, neighborsY)
    LGR = eta * localGuass
    net += np.einsum('ij, ijk->ijk', LGR, selected - net)
    return net


def som2MainB(input, epoch, lr, pivot, sigma):
    n = int(input.shape[1] * pivot)
    print('Number of Neurons: ', n)

    neighborsX = np.arange(n)
    neighborsY = np.arange(n)
    net = genNet(n, n, input.shape[1])
    # activationMap = np.zeros((n, n))

    for i in range(epoch):
        if i % 1000 == 0:
            print("Epoch: ",i)
            print('Lr',lr)
            print('n',n)
        r = np.random.randint(0, input.shape[0])
        selected = input[r]

        y, x, eucl = activate(selected, net)
        # activationMap[y, x] += 1

        net = deactivate(y, x, n // sigma, neighborsX, neighborsY, lr, net, selected)

        lr = lr * 0.9999
        n = n * 0.999

        if n < 1:
            print("\nRadius decayed")
            break

        if lr < .00001:
            print("\nLearning Rate decayed")
            break
    print('Number of times trained before fully decayed: ',i)

    return eucl,net



def fast_norm(var):
    return np.sqrt(np.dot(var, var.T))


def neuronDiff(net):
    'Modified Distance map function'
    'Credit: Giuseppe Vettigli JustGlowing'
    'https: // github.com / JustGlowing / minisom / blob / master / minisom.py'

    dmap = np.zeros([net.shape[0], net.shape[1]])

    it = np.nditer(dmap, flags=['multi_index'])
    while not it.finished:
        for ii in range(it.multi_index[0] - 1, it.multi_index[0] + 2):
            for jj in range(it.multi_index[1] - 1, it.multi_index[1] + 2):
                if (ii >= 0 and ii < net.shape[0] and jj >= 0 and jj < net.shape[1]):
                    w_1 = net[ii, jj, :]
                    w_2 = net[it.multi_index]
                    dmap[it.multi_index] += fast_norm(w_1 - w_2)
        it.iternext()

    dmap = dmap / np.max(dmap)
    return dmap
