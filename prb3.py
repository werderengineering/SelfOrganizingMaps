from ENPM808Y_HW3_Main import *
from matplotlib.path import Path


def prb3main():
    HPoints = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [2, 1],
        [2, 0],
        [3, 0],
        [3, 3],
        [2, 3],
        [2, 2],
        [1, 2],
        [1, 3],
        [0, 3],
        [0, 0]

    ])

    x, y = np.meshgrid(np.arange(0, 3, .1), np.arange(0, 3, .1))  # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T

    p = Path(HPoints)
    grid = p.contains_points(points)

    HinsideX = []
    HinsideY = []

    i = 0
    for Keep in grid:
        # print(Keep)
        if Keep:
            # print(points[i, :][0])
            HinsideX.append(points[i, :][0])
            HinsideY.append(points[i, :][1])
        i += 1

    HinsideX = np.array(HinsideX)
    HinsideY = np.array(HinsideY)

    Hinside = np.array([HinsideX, HinsideY]).T

    plt.plot(HPoints[:, 0], HPoints[:, 1])
    plt.scatter(Hinside[:, 0], Hinside[:, 1])
    plt.show()

    print('Mapping Out')
    # pivot = 14, sigma = 16
    route = som3Main(input=Hinside, epoch=10000, closeMethod='euclidiean', lr=.8, pivot=14, sigma=16)

    print("Mapping Complete")

    # plt.plot(HPoints[:, 0], HPoints[:, 1])
    # plt.scatter(Hinside[:, 0], Hinside[:, 1])
    # for i in range(route.shape[0]):
    #     plt.scatter(route[i, 0], route[i, 1], color='r', marker='.')
    #     plt.draw()
    #     plt.pause(.0001)

    # plt.clf()

    plt.plot(HPoints[:, 0], HPoints[:, 1], markersize=5)
    plt.scatter(HPoints[:, 0], HPoints[:, 1])
    plt.plot(route[:, 0], route[:, 1], color='r', markersize=2)
    plt.scatter(route[0, 0], route[0, 1], color='cyan')
    plt.show()

    print(route)
