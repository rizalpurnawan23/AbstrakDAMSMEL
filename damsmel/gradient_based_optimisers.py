import time
import pandas as pd
import numpy as np

# COMPARATIVE METHODS:
# -------------------------------------------------------------------
def gradient_descent(
        F, nabla_F, u_0,
        learning_rate= 0.01,
        M= 100,
        total_time = True,
        verbose= False
        ):
    u_k = np.array(u_0)
    eta = learning_rate
    k = 1
    loss = list()
    points = list()

    start = time.time()
    while k <= M:
        cycle_start = time.time()
        u_k = u_k - eta *nabla_F(u_k)
        points.append(u_k)
        loss.append(F(u_k))

        cycle_end = time.time()
        if verbose == True:
            print(f".....Loop {k}: {cycle_end - cycle_start} seconds")
        k += 1
    finish = time.time()
    if total_time == True:
        if verbose == True:
            print("\n")
        print(f">>> Total Time: {finish - start} seconds []")

    return pd.DataFrame({"solution": points, "loss": loss})


def stochastic_gradient_descent(
        F, gradQ, data, u_0,
        learning_rate= 0.01,
        M= 100,
        total_time = True,
        verbose= False
        ):
    u_k = np.array(u_0)
    eta = learning_rate
    k = 1
    loss = list()
    points = list()

    start = time.time()
    while k <= M:
        cycle_start = time.time()
        sample = data[np.random.choice(range(len(data)))]
        grad = lambda x: gradQ(x, *sample)
        u_k = u_k - eta *grad(u_k)
        points.append(u_k)
        loss.append(F(u_k))

        cycle_end = time.time()
        if verbose == True:
            print(f".....Loop {k}: {cycle_end - cycle_start} seconds")
        k += 1
    finish = time.time()
    if total_time == True:
        if verbose == True:
            print("\n")
        print(f">>> Total Time: {finish - start} seconds []")

    return pd.DataFrame({"solution": points, "loss": loss})


def adam(
        F, gradQ, data, u0,
        learning_rate= 0.001,
        beta1= 0.9, beta2= 0.999,
        eps= 10**(-8),
        M= 100,
        total_time= True,
        verbose= False
        ):
    u = np.array(u0)
    dim = len(u0)
    eta = learning_rate
    m = np.array([0] *dim)
    v = np.array([0] *dim)

    loss = list()
    points = list()

    t = 1
    
    start = time.time()
    while t <= M:
        cycle_start = time.time()
        sample = data[np.random.choice(range(len(data)))]
        grad = lambda x: gradQ(x, *sample)
        m = beta1 *m + (1 - beta1) *grad(u)
        v = beta2 *v + (1 - beta2) *grad(u)**2

        m_hat = m / (1 -beta1**t)
        v_hat = v / (1 -beta2**t)

        u = u - eta *(m_hat / (np.sqrt(v_hat) + eps))

        points.append(u)
        loss.append(F(u))

        cycle_end = time.time()
        if verbose == True:
            print(f".....Loop {t}: {cycle_end - cycle_start} seconds")

        t += 1
    finish = time.time()
    if total_time == True:
        if verbose == True:
            print("\n")
        print(f">>> Total Time: {finish - start} seconds []")

    return pd.DataFrame({"solution": points, "loss": loss})

def batch_adam(
        F, nabla_F, u_0,
        learning_rate= 0.01,
        eps= 10**(-8),
        beta1= 0.9, beta2= 0.999,
        M= 100,
        total_time = True,
        verbose= False
        ):
    u = np.array(u_0)
    dim = len(u)
    eta = learning_rate
    m = np.array([0] *dim)
    v = np.array([0] *dim)

    points = list()
    loss = list()

    start = time.time()
    for i in range(1, M + 1):
        loop_start = time.time()
        
        m = beta1 *m + (1 - beta1) *nabla_F(u)
        v = beta2 *v + (1 - beta2) *nabla_F(u)**2
        hat_m = m / (1 - beta1**i)
        hat_v = v / (1 - beta2**i)
        u = u - eta *hat_m / (np.sqrt(hat_v) + eps)

        points.append(u)
        loss.append(F(u))

        loop_end = time.time()
        if verbose == True:
            print(f".....Loop {i}: {loop_end - loop_start} seconds")
    finish = time.time()
    if total_time == True:
        if verbose == True:
            print("\n")
        print(f">>> Runtime: {finish - start} seconds []")
    return pd.DataFrame({"solution": points, "loss": loss})