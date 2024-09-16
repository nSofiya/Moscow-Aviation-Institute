#работа над практической задачей
import numpy as np
from numpy import array

#f = - (7*x[0] + 4*x[1] + 4*x[2] + 3*x[0]*x[1] + x[0]*x[2] + 2*x[1]*x[2])

def function(x):#пробная
    r = [-0.17, 0.089, 0.015, -0.25, -0.16, 0.448, 0.161, 0.27, 0.074, -0.008]
    f = 0
    for i in range(0, 10):
        f -= x[i] * r[i]

    return f

def condition(x):
    w = [0.1351, 7.7995, 0.5928, 1.1614, 0.7414, 0.2233, 0.7448, 3.0580, 0.0332, 0.4585]
    f = 0
    for i in range(0, 10):
        f += x[i] * w[i]

    return f

def condition2(x):
    sigma = [[0.00509, 0.00205, -0.00047, 0.00041, -0.00014, 0.00030, -0.00050, 0.00489, 0.00417, 0.00607],
             [0, 0.00237, -0.00028, -0.00013, 0.00017, 0.00289, -0.00152, 0.00452, 0.00469, 0.00500],
             [0, 0, 0.00046, -0.00029, 0.00043, -0.00177, 0.00068, -0.00151, -0.00024, -0.00102],
             [0, 0, 0, 0.00030, -0.00032, 0.00075, -0.00017, 0.00040, -0.00028, 0.00054],
             [0, 0, 0, 0, 0.00065, -0.00179, 0.00053, -0.00086, 0.00073, 0.00069],
             [0, 0, 0, 0, 0, 0.01514, -0.00524, 0.00812, 0.00547, 0.00545],
             [0, 0, 0, 0, 0, 0, 0.00206, -0.00392, -0.00265, -0.00278],
             [0, 0, 0, 0, 0, 0, 0, 0.01130, 0.00781, 0.00943],
             [0, 0, 0, 0, 0, 0, 0, 0, 0.01022, 0.01110],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01716]]
    f = 0
    for i in range(0, 10):
        for j in range(i, 10):
            f += sigma[i][j] * x[i] * x[j]

    return f

"""def transfer_func(x): #s1
    x_new = []
    for i in range(0, len(x)):
        t = 1 / (1 + np.exp(-2*x[i]))
        x_new.append(t)

    return x_new"""

def transfer_func(x): #s2
    x_new = []
    for i in range(0, len(x)):
        t = 1 / (1 + np.exp(-x[i]))
        x_new.append(t)

    return x_new

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
        if x[i] >= 0.5:
            x_bin.append(1)
        else:
            x_bin.append(0)

    return x_bin

def init(s, border, n, threshold):
    points = [[0]*2]*s
    #points_and_bin = []
    """x1 = []
    x2 = []
    for i in range(0, n): #крайние точки
        x1.append(-a)
        x2.append(a)"""

    points[0] = np.random.uniform(-border, border, n)
    point_bin = binarization(transfer_func(points[0]))
    while (condition(point_bin) > 10) or (condition2(point_bin) > 0.04):
        #print('oups ', point_bin)
        points[0] = np.random.uniform(-border, border, n)
        point_bin = binarization(transfer_func(points[0]))
    for i in range(1, s):
        param = 0
        while param == 0:
            x_new = np.random.uniform(-border, border, n)
            point_bin = binarization(transfer_func(x_new))
            while (condition(point_bin) > 10) or (condition2(point_bin) > 0.04):
                #print('oups ', point_bin)
                x_new = np.random.uniform(-border, border, n)
                point_bin = binarization(transfer_func(x_new))
            #print('huray ', point_bin)
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
        #points_and_bin.append([x_new, point_bin])

    #print("points^ ", points_and_bin)

    return points

def step(points, b, B, border, P, n, delta):
    results = []
    for i in range(0, len(points)):
        point_t = transfer_func(points[i])
        point_bin = binarization(point_t)
        results.append([function(point_bin), points[i], point_bin])
        #print("result ", results[i][2])
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
            point_new = np.random.uniform(x1, x2, n)
            point_bin = binarization(transfer_func(point_new))
            while (condition(point_bin) > 10) or (condition2(point_bin) > 0.04):
                #print('sorry ', point_bin)
                point_new = np.random.uniform(x1, x2, n)
                point_bin = binarization(transfer_func(point_new))
            points.append(point_new)

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
            point_new = np.random.uniform(x1, x2, n)
            point_bin = binarization(transfer_func(point_new))
            while (condition(point_bin) > 10) or (condition2(point_bin) > 0.04):
                #print('sorry ', point_bin)
                point_new = np.random.uniform(x1, x2, n)
                point_bin = binarization(transfer_func(point_new))
            points.append(point_new)
            #print('Huray ', point_bin)
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
    #print('end of itteration ', results[0][0])


    return points, results


prog_count = 10
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

border = 5 #граница области D
n = 10
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
    
    print("Прогон ", j)
    print("Минимум функции достигается в точке: ", results[0][2])
    print("Наименьшее значение функции", results[0][0])
    print("Risk ", condition2(results[0][2]))




#for i in range(0, prog_count):
#    sigma += (final_res[i] - min_sum/prog_count)**2

#print(min_sum1/prog_count)
#print(min_sum/prog_count)
#print(success)
#print(sigma / (prog_count - 1))
#print(final_res)

"""min_sum1 += results[0][0]
    min_sum += results[0][0] + 0
    if results[0][0] == 0:
        success += 1
    final_res.append(results[0][0] + 0)"""