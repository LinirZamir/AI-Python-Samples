import numpy as np

true_w = np.array([1,2,3,4,5])
d = len(true_w)
points = []
####################################################
## Initialize points dimension-d
for i in range(1000):
    x = np.random.randn(d)
    y = true_w.dot(x) + np.random.randn() ## Plus some noise
    points.append((x,y))

####################################################
## Helper functions
def F(w):
    return sum((w.dot(x) - y)**2 for x,y in points) / len(points)
    
def dF(w):
    return sum(2*(w.dot(x) - y)*x for x,y in points) / len(points)

####################################################
## Gradient Descent
def gradientDecent(step):
    w = np.zeros(d)
    for t in range(1000):
        value = F(w)
        gradient = dF(w)
        w = w - step * gradient
        print("Iteration: {}, w = {}".format(t, w))

gradientDecent(0.01)
