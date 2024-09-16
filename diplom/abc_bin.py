#работа с тестовыми функциями

import numpy as np
from numpy import array


def function(x):#f1
    f = 0
    for i in range(0, len(x)):
        f = f + x[i]**2
        #print("f ", f)

    return f

def function2(x):#f2
    sum = 0
    com = 1
    for i in range(0, len(x)):
        sum = sum + x[i]
        com = com * x[i]
        #print("f ", f)
    f = sum + com
    return f

def function3(x):#f3
    f = 0
    for i in range(0, len(x)-1):
        f = f + 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
        #print("f ", f)

    return f

def function4(x):#f4
    f = 0
    for i in range(0, len(x)-1):
        f = f + x[i]**2 - 10 * np.cos(2 * np.pi * x[i]) + 10
        #print("f ", f)

    return f

def function5(x):#f5
    sum1 = 0
    sum2 = 0
    f = 0
    for i in range(0, len(x)):
        sum1 = sum1 + x[i]**2
        sum2 = sum2 + np.sin(x[i])**2
        f = f + np.sin(x[i])**2
        #print("f ", f)
    f = (f - np.exp(-sum1)) * np.exp(-sum2)
    return f


def transfer_func(x): #s1
    x_new = []
    for i in range(0, len(x)):
        t = 1 / (1 + np.exp(-2*x[i]))
        x_new.append(t)

    return x_new

"""def transfer_func(x): #s2
    x_new = []
    for i in range(0, len(x)):
        t = 1 / (1 + np.exp(-x[i]))
        x_new.append(t)

    return x_new"""

"""def transfer_func(x): #s3
    x_new = []
    for i in range(0, len(x)):
        t = 1 / (1 + np.exp(-x[i]/2))
        x_new.append(t)

    return x_new"""

"""def transfer_func(x): #s4
    x_new = []
    for i in range(0, len(x)):
        t = 1 / (1 + np.exp(-x[i]/3))
        x_new.append(t)

    return x_new"""


"""def transfer_func(x): #v1
    x_new = []
    for i in range(0, len(x)):
        if x[i] <= 0:
            t = 2 / (1 + np.exp(-2 * x[i]))
        else:
            t = 2 / (1 + np.exp(-2 * x[i])) - 1
        x_new.append(t)

    return x_new"""

"""def transfer_func(x): #v2
    x_new = []
    for i in range(0, len(x)):
        if x[i] <= 0:
            t = 2 / (1 + np.exp(x[i]))
        else:
            t = 2 / (1 + np.exp(x[i])) - 1
        x_new.append(t)

    return x_new"""

"""def transfer_func(x): #v3
    x_new = []
    for i in range(0, len(x)):
        if x[i] <= 0:
            t = 2 / (1 + np.exp(x[i] / 2))
        else:
            t = 2 / (1 + np.exp(x[i] / 2)) - 1
        x_new.append(t)

    return x_new"""

"""def transfer_func(x): #v4
    x_new = []
    for i in range(0, len(x)):
        if x[i] <= 0:
            t = 2 / (1 + np.exp(x[i] / 3))
        else:
            t = 2 / (1 + np.exp(x[i] / 3)) - 1
        x_new.append(t)

    return x_new"""


def binarization(x):
    x_bin = []

    for i in range(0, len(x)):
        rand = np.random.uniform(0, 1)
        if x[i] >= rand:
            x_bin.append(1)
        else:
            x_bin.append(0)

    return x_bin

def init(s, border, n, threshold):
    points = [[0]*2]*s
    """x1 = []
    x2 = []
    for i in range(0, n): #крайние точки
        x1.append(-a)
        x2.append(a)"""

    points[0] = np.random.uniform(-border, border, n)
    for i in range(1, s):
        param = 0
        while param == 0:
            x_new = np.random.uniform(-border, border, n)
            for j in range(0, i):
                dist = 0
                for k in range(0, n):
                    dist += (x_new[k] - points[j][k])**2
                dist = np.sqrt(dist)
                if dist < threshold:
                    param = 0
                    print("dist!! ", dist)
                else:
                    param = 1
        points[i] = x_new

    #print("points^ ", points)

    return points

def step(points, b, B, border, P, n, delta):
    results = []
    for i in range(0, len(points)):
        point_t = transfer_func(points[i])
        point_bin = binarization(point_t)
        results.append([function(point_bin), points[i], point_bin])
    #print("result ", results)
    results_sort = sorted(results, key=lambda x: x[0])
    #print("sorted ", results_sort)
    results = []
    results = results_sort #вектор значений функции (первые b- наилучшие, следующие p- перспективные)
    #for i in range(0, s):
    #    points[i] = results[i][1]

    points = []
    border_count = 0
    for i in range(0, b): #генерирование новых решений в наилучших областях
        points.append(results[i][1])
        x1 = np.zeros(n)
        x2 = np. zeros(n)
        for k in range(0, n):
            x1[k] = results[i][1][k] - delta
            if x1[k] < -border:
                x1[k] = -border
                border_count += 1
            x2[k] = results[i][1][k] + delta
            if x1[k] > border:
                x1[k] = border
                border_count += 1
        for j in range(1, B):
            points.append(np.random.uniform(x1, x2, n))

    for i in range(b, s): #генерирование новых решений в перспективных областях
        points.append(results[i][1])
        x1 = np.zeros(n)
        x2 = np. zeros(n)
        for k in range(0, n):
            x1[k] = results[i][1][k] - delta
            if x1[k] < -border:
                x1[k] = -border
                border_count += 1
            x2[k] = results[i][1][k] + delta
            if x1[k] > border:
                x1[k] = border
                border_count += 1
        for j in range(1, P):
            points.append(np.random.uniform(x1, x2, n))
    #if border_count > 0:
    #    print("border!! ", border_count)

    results = []
    points_bin = []
    for i in range(0, len(points)):
        point_t = transfer_func(points[i])
        point_bin = binarization(point_t)
        #points_bin.append(point_bin)
        results.append([function(point_bin), points[i], point_bin])
    results_sort = sorted(results, key=lambda x: x[0])
    results = []
    results = results_sort
    #results.sort()


    points = [[0]*2]*s
    for i in range(0, s):
        points[i] = results[i][1]
        #print(function(points[i]))
    #print("end of itteration ", results[0][0])


    return points, results


prog_count = 100
min_sum = 0
min_sum1 = 0
success = 0
sigma = 0
final_res = []

K = 300
s = 50
b = 20
p = s - b
B = 10
P = 10
R = 0.85
q = 1
delta = 0.85
treshold = 0.0001

border = 30 #граница области D
n = 5
k = 0

for j in range(0, prog_count):
    points = init(s, border, n, treshold)
    k = 0
    while k < K:
        points, results = step(points, b, B, border, P, n, delta)
        #delta = delta * R
        #print("Минимум функции достигается в точке: ", results[0][2], results[0][1])  # , results[0][1]
        #print("Наименьшее значение функции", results[0][0])
        # print("delta ", delta)
        k += 1
    min_sum1 += results[0][0]
    min_sum += results[0][0] + 0
    if results[0][0] == 0:
        success += 1
    final_res.append(results[0][0] + 0)
    print("Прогон ", j)
    print("Минимум функции достигается в точке: ", results[0][2])
    print("Наименьшее значение функции", results[0][0])

for i in range(0, prog_count):
    sigma += (final_res[i] - min_sum/prog_count)**2

print(min_sum1/prog_count)
print(min_sum/prog_count)
print(success)
print(sigma / (prog_count - 1))
#print(final_res)

"""points = init(s, border, n, treshold)
while k < K:
    points, results = step(points, b, B, border, P, n, delta*(1-k/K))
    #delta = delta * R
    print("Минимум функции достигается в точке: ", results[0][2], results[0][1]) #, results[0][1]
    print("Наименьшее значение функции", results[0][0])
    #print("delta ", delta)
    k += 1"""

"""while k < K:
    while delta > 9.9*np.e**(-18):
        points, results = step(points, b, B, border, P, n, delta)
        delta = delta * R**q
        print("Минимум функции достигается в точке: ", results[0][2])
        print("Наименьшее значение функции", results[0][0])
        #print("delta ", delta)
        k += 1
    q += 1
    delta = 0.85 * R**q"""

#print("Минимум функции достигается в точке: ", results[0][2])
#print("Наименьшее значение функции", results[0][0])
#print(q)
#print("Минимум функции достигается в точке: ", points[0])