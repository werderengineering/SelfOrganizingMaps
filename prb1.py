from ENPM808Y_HW3_Main import *

from som1 import *
def prb1main():
    tspPoints = np.array([
        [0.2, 0.1],
        [0.15, 0.2],
        [0.4, 0.45],
        [0.2, 0.77],
        [0.5, 0.9],
        [0.83, 0.65],
        [0.7, 0.5],
        [0.82, 0.35],
        [0.65, 0.23],
        [0.6, 0.28]
    ])
    plt.scatter(tspPoints[:, 0], tspPoints[:, 1])

    route = som1Main(input=tspPoints, epoch=1000, closeMethod='euclidiean', lr=.8, pivot=10, sigma=6)
    # lr = .8, pivot = 10, sigma = 6 Best

    for i in range(route.shape[0]):
        plt.scatter(route[i, 0], route[i, 1], color='r', marker='.')
        plt.draw()
        plt.pause(.00001)

    plt.clf()

    plt.scatter(tspPoints[:, 0], tspPoints[:, 1])
    plt.plot(route[:, 0], route[:, 1], color='r', markersize=2)
    plt.scatter(route[0, 0], route[0, 1], color='cyan')
    plt.show()

    print(route)
