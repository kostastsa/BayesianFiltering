import numpy as np

## 1
f1 = lambda x, p: (1 + np.dot(x,x))**(p/2)
J1 = lambda x, p: p * (1 + np.dot(x,x))**(p/2-1) * x
H1 = lambda x, p: 2 * p*(p/2-1)*(1 + np.dot(x,x))**(p/2-2)*np.outer(x,x) + np.eye(dx)*p* (1 + np.dot(x,x))**(p/2-1)

## 2 Sinc
f2 = lambda x: np.sin(np.dot(x,x))/np.dot(x,x)
J2 = lambda x: 2 * (np.dot(x,x)*np.cos(np.dot(x,x)) - np.sin(np.dot(x,x))) / (np.dot(x,x))**2 * x
H2 = lambda x: -4*(np.sin(np.dot(x,x))/np.dot(x,x) + \
                  2*(np.cos(np.dot(x,x))*np.dot(x,x) - \
                     np.sin(np.dot(x,x))) / np.dot(x,x)**3 ) * \
                     np.outer(x,x) + 2 * (np.dot(x,x)*np.cos(np.dot(x,x)) - \
                                          np.sin(np.dot(x,x))) / \
                                          (np.dot(x,x))**2 * np.eye(dx)

## 3 Linear-Nonlinear product <------------------- Hard one
f3 = lambda x: x[0] * np.sin(x[1])
J3 = lambda x: np.array([np.sin(x[1]), x[0]*np.cos(x[1])])
H3 = lambda x: np.array([[0, np.cos(x[1])],[np.cos(x[1]), -x[0]*np.sin(x[1])]])

## 4 Linear-Nonlinear sum
f4 = lambda x: x[0] + np.sin(x[1])
J4 = lambda x: np.array([1, np.cos(x[1])])
H4 = lambda x: np.array([[0, 0], [0, -np.sin(x[1])]])

## 5 Quadratic
a = 1
b = 1
A = np.array([[a, 0], [0, b]])
f5 = lambda x: np.dot(x, np.matmul(A, x))/2
J5 = lambda x: np.matmul(A, x)
H5 = lambda x: A