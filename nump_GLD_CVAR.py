#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 23:13:02 2024

@author: masiha97
"""
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt 
import jax 
from numpy import linalg as LA

def h(x, gamma):
    if x > gamma:
        return x
    elif -gamma <= x <= gamma:
        return (x**2) / (4 * gamma) + x / 2 + gamma / 4
    else:  # x < -gamma
        return 0

def g(x, y):
    if y > np.sin(x) and 0 < x < 3*np.pi:
        return (y - np.sin(x))**2
    elif x < 0 and y > 0:
        return (y - x)**2
    elif -3*np.pi < y < 0 and x < np.sin(y):
        return (x - np.sin(y))**2
    elif y < -3*np.pi and x < 0:
        return (y + x + 3*np.pi)**2
    elif y < -np.sin(x) - 3*np.pi and 0 < x<3*np.pi:
        return (y + np.sin(x) + 3*np.pi)**2
    elif y < -3*np.pi and x > 3*np.pi:
        return (y - x + 6*np.pi)**2
    elif -3*np.pi < y < 0 and x > -np.sin(y) + 3*np.pi:
        return (x + np.sin(y) - 3*np.pi)**2
    elif y > 0 and x > 3*np.pi:
        return (x - 3*np.pi + y)**2
    else:
        return np.nan

def gradient_g(x, y):
    # Calculating partial derivatives for x and y
    # The default gradients are set to zero
    dx = 0
    dy = 0

    if -np.sin(x) - 3*np.pi <= y <= np.sin(x) and np.sin(y) <= x <= -np.sin(y) + 3*np.pi:
        pass  # No need to modify dx, dy as they are already 0
    elif y > np.sin(x) and 0 < x < 3*np.pi:
        dx = -2 * (y - np.sin(x)) * np.cos(x)
        dy = 2 * (y - np.sin(x))
    elif x < 0 and y > 0:
        dx = -2 * (y - x)
        dy = 2 * (y - x)
    elif -3*np.pi < y < 0 and x < np.sin(x):
        dx = 2 * (x - np.sin(y))
        dy = -2 * np.cos(y) * (x - np.sin(y))
    elif y < -3*np.pi and x < 0:
        dx = 2 * (y + x + 3*np.pi)
        dy = 2 * (y + x + 3*np.pi)
    elif y < -np.sin(x) - 3*np.pi and 0 < x < 3*np.pi:
        dx = 2 * (y + np.sin(x) + 3*np.pi) * np.cos(x)
        dy = 2 * (y + np.sin(x) + 3*np.pi)
    elif y < -3*np.pi and x > 3*np.pi:
        dx = -2 * (y - x + 6*np.pi)
        dy = 2 * (y - x + 6*np.pi)
    elif -3*np.pi < y < 0 and x > -np.sin(y) + 3*np.pi:
        dx = 2 * (x + np.sin(y) - 3*np.pi)
        dy = 2 * np.cos(y) * (x + np.sin(y) - 3*np.pi)
    elif y > 0 and x > 3*np.pi:
        dx = 2 * (x - 3*np.pi + y)
        dy = 2 * (x - 3*np.pi + y)
    return dx, dy


# GLD algorithm
def gradient_langevin_dynamics_iteration(num_iterations, epsilon, step_size, noise_scale, init_x, init_y, theta):
    # Initialize x and y
    x, y = init_x, init_y

    # Sampling points

    for iteration in range(num_iterations):
        # Compute the gradient
        grad_x, grad_y = gradient_g(x-theta[0], y-theta[1])

        # Update x and y using Gradient Descent with Langevin noise
        x -= step_size * grad_x + np.sqrt(2*epsilon)*np.random.normal(scale=np.sqrt( step_size * noise_scale))
        y -= step_size * grad_y + np.sqrt(2*epsilon)*np.random.normal(scale=np.sqrt( step_size * noise_scale))

        # Store the sample at the target iteration

    return x,y


def tilde_g(theta, z):
    x, y = z
    return g(x - theta[0], y - theta[1])

def gradient_tilde_g(theta, z):
    x, y = z
    grad_g_x, grad_g_y = gradient_g(x +5- theta[0], y-2.5 - theta[1])
    return np.array([-grad_g_x, -grad_g_y])

def compute_expression(theta, x_sample, lambda_, gld_samples):
    # Compute nabla_theta_tilde_g for the specific x
    grad_theta_tilde_g_x = gradient_tilde_g(theta, x_sample)

    # Compute expected value of nabla_theta_tilde_g over GLD samples
    grad_theta_tilde_g_sum = np.array([0.0, 0.0])
    for z in gld_samples:
        grad_theta_tilde_g_sum += gradient_tilde_g(theta, z)

    expected_grad = grad_theta_tilde_g_sum / len(gld_samples)

    # Compute the final expression
    result = (expected_grad - grad_theta_tilde_g_x) / lambda_
    return result

def derivative_h(x, gamma):
    if x > gamma:
        return 1
    elif -gamma <= x <= gamma:
        return x / (2 * gamma) + 1 / 2
    else:  # x < -gamma
        return 0

#def derivative_h_wrt_beta(x, beta, gamma):
    # Placeholder for the derivative of h w.r.t. beta
    # This needs to be defined based on your specific h function
#    return -derivative_h(x - beta, gamma)

def f(theta, Y_sample):
    return theta[0]**2+theta[1]**2-(Y_sample[1])**2-(Y_sample[0])**2

def gradient_f(theta, Y_sample):
    grad_x0 = 2*theta[0]
    grad_x1 = 2*theta[1]
    return np.array([grad_x0, grad_x1])

def expected_gradient(theta, beta, Y_samples, lambda_, delta, gamma):
    gradient_sum_theta = 0
    gradient_sum_beta = 0
    for Y_sample in Y_samples:
        f_value = f(theta, Y_sample)  # Define this function
        h_prime = derivative_h(f_value - beta, gamma)
        grad_f = gradient_f(theta, Y_sample)  # Define this function
        compute_expression(theta, Y_sample, lambda_, Y_samples)
        #print('f_value:',f_value)
        #print('h_prime:',h_prime)
        #print('h:',h(f_value - beta, gamma))

        gradient_sum_theta += h_prime * grad_f + compute_expression(theta, Y_sample, lambda_, Y_samples) * h(f_value - beta, gamma)
        gradient_sum_beta += 1-h_prime

    expected_grad_theta = gradient_sum_theta / (delta*len(Y_samples))
    expected_grad_beta = gradient_sum_beta / (delta*len(Y_samples))
    return expected_grad_theta, expected_grad_beta

# Gradient descent algorithm
def gradient_descent(theta_init, beta_init, step_size, step_size_beta, Y_samples, lambda_, delta, gamma,beta_num_itera):
    theta = theta_init
    beta = beta_init
    #for _ in range(num_iterations):
    for i in range(beta_num_itera):
        grad_phi_theta, grad_phi_beta = expected_gradient(theta, beta, Y_samples, lambda_, delta, gamma)
        beta = beta - step_size_beta * grad_phi_beta
    print('beta:',beta)
    grad_phi_theta, grad_phi_beta = expected_gradient(theta, beta, Y_samples, lambda_, delta, gamma)
    theta = theta - step_size * grad_phi_theta
    return theta, 0

def CVAR_delta(theta, beta_init, step_size_beta, Y_samples, lambda_, delta, gamma,beta_num_itera):
    #return 0
    beta = beta_init
    h1_sum=0
    for i in range(beta_num_itera):
        grad_phi_theta, grad_phi_beta = expected_gradient(theta, beta, Y_samples, lambda_, delta, gamma)
        beta1 = beta - step_size_beta * grad_phi_beta
        beta=beta1
    print('beta:', beta)
    for Y_sample in Y_samples:
        f_value = f(theta, Y_sample)
        h1=h(f_value - beta, gamma)
        h1_sum+=h1
    print('h1_sum:', h1_sum)    
    expected_h=h1_sum/(delta*len(Y_samples))
    print('expected_h:', expected_h) 
    return beta + expected_h
# Adjusting epsilon as requested
epsilon = 0.001
delta=0.1
gamma=0.01

beta_init=1
beta_num_itera=100
num_iterations1=1000
num_iterations2=1
step_size=1
step_size_beta=0.001
step_size_gd=0.001
noise_scale=1
samples = []
theta=np.array([0,0])
f_value_sum=0
samples_phi_grad=[]
for num_iterations in range(10):
    for iteration in range(500):
        init_x=20*random.random()-10+5+theta[0]
        init_y=20*random.random()-10-2.5+theta[1]
        # Run the algorithm and get samples at iteration 100
        x1,y1 = gradient_langevin_dynamics_iteration(num_iterations1, epsilon, step_size, noise_scale, init_x, init_y, theta)
        samples.append((x1, y1))
    print(len(samples))
    theta_1,beta=gradient_descent(theta, beta_init, step_size_gd, step_size_beta, samples, epsilon, delta, gamma,beta_num_itera)    
    theta=theta_1
    print(CVAR_delta(theta, beta_init, step_size_beta, samples, epsilon, delta, gamma,beta_num_itera))
    grad_phi_theta1, grad_phi_beta1 =expected_gradient(theta,beta,samples,epsilon,delta,gamma)
    samples_phi_grad.append(LA.norm(grad_phi_theta1,2))
    print('norm of gradient:',LA.norm(grad_phi_theta1,2))
    #Plotting the samples
    x_vals, y_vals = zip(*samples)
    plt.scatter(x_vals, y_vals, color='blue')
    plt.title("Samples at Iteration 100 (Epsilon = 0.01)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()
    samples = []