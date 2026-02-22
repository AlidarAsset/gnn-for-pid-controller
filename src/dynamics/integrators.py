import torch

def rk4_step(state, u, dt, f_func):
    x, v = state
    k1x, k1v = f_func((x, v), u)
    k2x, k2v = f_func((x + 0.5 * dt * k1x, v + 0.5 * dt * k1v), u)
    k3x, k3v = f_func((x + 0.5 * dt * k2x, v + 0.5 * dt * k2v), u)
    k4x, k4v = f_func((x + dt * k3x, v + dt * k3v), u)
    x_next = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    v_next = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return x_next, v_next

def euler_step(state, u, dt, f_func):
    x_dot, v_dot = f_func(state, u)
    x_next = state[0] + dt * x_dot
    v_next = state[1] + dt * v_dot
    return x_next, v_next