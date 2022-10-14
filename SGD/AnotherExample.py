import numpy as np


def phi(x):
    return np.array([1, x])


def initializeFunction(d):
    return np.zeros(d)


def trainLoss(w, points):
    return 1/len(points) * sum((w.dot(phi(x)) - y)**2 for x, y in points)


def dTrainLoss(w, points):
    return 1/len(points) * sum(2*(w.dot(phi(x)) - y) * phi(x) for x, y in points)


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
    step = 0.1
    gradientDescent(initializeFunction, trainLoss, dTrainLoss,
                    step, trainExample, 2)


main()
