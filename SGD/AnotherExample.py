import numpy as np


def phi(x):
    return np.array(x)

def phi_1(x):
    return np.array([1, x])

def initializeFunction(d):
    return np.zeros(d)

# Stochastic GD


def trainLoss(w, points):
    return 1/len(points) * sum((w.dot(phi_1(x)) - y)**2 for x, y in points)


def dTrainLoss(w, points):
    return 1/len(points) * sum(2*(w.dot(phi_1(x)) - y) * phi(x) for x, y in points)

# Hinge Lost


def hingeLost(w, points):
    return 1/len(points) * sum(max(1-w.dot(phi(x)) * y, 0) for x, y in points)


def dHingeLost(w, points):
    return 1/len(points) * sum(-phi(x)*y if 1 - w.dot(phi(x)) * y > 0 else 0 for x, y in points)


def gradientDescent(initialize, f, df, eta, points, d):
    w = initialize(d)
    for t in range(500):
        value = f(w, points)
        gradient = df(w, points)
        w = w - eta * gradient
        print(f"t = {t}; value = {value}; w = {w}")


def main():
    trainExample = [
        (1, 1),
        (2, 3),
        (4, 3),
    ]

    HingeTrainExample = [
        ((0, 2), 1),
        ((-2, 0), 1),
        ((1, -1), -1),
    ]
    step = 0.1
    dimension = 2
    ##gradientDescent(initializeFunction, trainLoss, dTrainLoss, step, trainExample, 2)
    # Hinge GD
    gradientDescent(initializeFunction, hingeLost,
                    dHingeLost, step, HingeTrainExample, dimension)


main()
