from control.matlab import *
import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy.optimize import minimize, OptimizeResult, differential_evolution

wandb.init(project="control_optimizer")

s = tf([1, 0], [1])

overshoot = .1
simulation_time = 5
time_steps = 150
settling_time = 2
T = np.linspace(0, simulation_time, time_steps)


def find_settling_time(res):
    settled_q = (np.abs(res - 1) < 0.02)
    for i in range(1, len(res) + 1):
        if not settled_q[-i]:
            return T[-i]
    return 0


def plot_response(res):
    plt.plot(T, res, [0, simulation_time], [1 + overshoot] * 2, [settling_time, settling_time], [0, 1 - 0.02])
    plt.plot([0, simulation_time], [1.02, 1.02], [0, simulation_time], [1 - 0.02, 1 - 0.02])
    po = (np.max(res) - 1) * 100
    ts = find_settling_time(res)
    plt.title(f'P.O = {po:.2f}%,  $T_s$={ts:.2f}s');


def controller(k, p, z):
    return k * tf([1 / z, 1], [1 / p, 1])


sys = ((-10) * (-10 * (s + 1) * (s + 0.01))) / ((s + 10) * ((s ** 2 + 2 * s + 2) * (s ** 2 + 0.02 * s + 0.0101)))

ramp = lambda x: x if x > 0 else 0


def alpha_beta_cost(alpha, beta):
    def system_cost(params):
        res, _ = step(feedback(sys * controller(*params)), T)
        return alpha * ramp((np.max(res) - 1) - overshoot) + beta * ramp(find_settling_time(res) - settling_time)

    return system_cost


cost_fn = alpha_beta_cost(1, 1)


def callback(x):
    cost = cost_fn(x)
    print('Cost: ', cost)
    wandb.log({'Cost': cost, 'K': x[0], 'Pole': x[1], 'Zero': x[2]})
    return cost <= 0


def evolution_callback(x, convergence):
    cost = cost_fn(x)
    print('Cost: ', cost)
    wandb.log({'Cost': cost, 'K': x[0], 'Pole': x[1], 'Zero': x[2]})
    return cost <= 0


if __name__ == '__main__':
    # minimize(cost_fn, x0=np.array([1.529, 999.991, 1]), callback=callback, method='Nelder-Mead',
    #          options={'disp': True, 'maxiter': 100000, 'xatol': 0, 'fatol': 0}, tol = 0)

    differential_evolution(cost_fn, bounds=[(.01, 100), (2, 100000), (0, 10)], callback=evolution_callback, disp=True,
                           maxiter=1000000, tol=0)
