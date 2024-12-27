import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Inverted pendulum system parameters
mass_pendulum = 0.15  # Pendulum mass (kg)
mass_wheel = 0.9    # Wheel mass (kg)
length_pendulum = 0.25  # Pendulum length (m)
gravity = 9.81       # Gravity acceleration (m/s^2)
inertia = 0.004      # Pendulum moment of inertia (kg.m^2)
damping = 0.015       # Damping coefficient for smoother motion

# Simulation time parameters
dt = 0.01  # Time step (s)
time = np.arange(0, 10, dt)  # Time range (10 seconds)

def simulate_pendulum(Kp, Ki, disturbance=0.0):
    """
    Simulate the inverted pendulum system with PI control.
    
    Args:
        Kp (float): Proportional gain
        Ki (float): Integral gain
        disturbance (float): Initial angular displacement (rad)
    
    Returns:
        numpy.array: Array of pendulum angles over time
    """
    theta = np.zeros_like(time)  # Pendulum angle (rad)
    theta_dot = np.zeros_like(time)  # Angular velocity (rad/s)
    integral_error = 0
    desired_theta = 0  # Target angle (upright position)
    
    # Apply initial disturbance
    theta[0] = disturbance
    
    for i in range(1, len(time)):
        # Calculate control signal
        error = desired_theta - theta[i - 1]
        integral_error += error * dt
        torque = Kp * error + Ki * integral_error

        # Calculate angular acceleration
        theta_ddot = (torque - mass_pendulum * gravity * length_pendulum * np.sin(theta[i - 1]) 
                     - damping * theta_dot[i - 1]) / inertia
        
        # Update state variables using numerical integration
        theta_dot[i] = theta_dot[i - 1] + theta_ddot * dt
        theta[i] = theta[i - 1] + theta_dot[i] * dt

    return theta

def cost_function(Kp, Ki, disturbance=0.0):
    """
    Calculate the cost (ITAE) for given PI parameters.
    
    Args:
        Kp (float): Proportional gain
        Ki (float): Integral gain
        disturbance (float): Initial disturbance angle
    
    Returns:
        float: ITAE (Integral Time Absolute Error) cost value
    """
    theta = simulate_pendulum(Kp, Ki, disturbance)
    error = np.abs(theta - 0)
    time_weighted_error = time * error
    itae = np.sum(time_weighted_error * dt)
    return itae

def artificial_bee_colony(population_size, max_iter, disturbance=0.0):
    """
    Optimize PI controller parameters using Artificial Bee Colony algorithm.
    
    Args:
        population_size (int): Number of solutions in the population
        max_iter (int): Maximum number of iterations
        disturbance (float): Initial disturbance angle
    
    Returns:
        tuple: Optimal Kp and Ki values
    """
    # Initialize random population
    population = np.random.uniform(0, 100, (population_size, 2))
    fitness = np.zeros(population_size)
    
    # Track optimization progress
    best_fitness_over_time = []
    best_solution_over_time = []

    # Evaluate initial population
    for i in range(population_size):
        fitness[i] = cost_function(population[i, 0], population[i, 1], disturbance)
    
    best_fitness = np.min(fitness)
    best_solution = population[np.argmin(fitness)]
    
    best_fitness_over_time.append(best_fitness)
    best_solution_over_time.append(best_solution)
    
    # Main ABC optimization loop
    for iteration in range(max_iter):
        # Employed bee phase
        for i in range(population_size):
            candidate = population[i] + np.random.uniform(-1, 1, 2)
            candidate = np.clip(candidate, 0, 100)
            
            candidate_fitness = cost_function(candidate[0], candidate[1], disturbance)
            
            if candidate_fitness < fitness[i]:
                population[i] = candidate
                fitness[i] = candidate_fitness
                
                if candidate_fitness < best_fitness:
                    best_fitness = candidate_fitness
                    best_solution = candidate
        
        best_fitness_over_time.append(best_fitness)
        best_solution_over_time.append(best_solution)
        
        print(f"Iteration {iteration+1}: Best Fitness = {best_fitness:.5f}, Best Kp = {best_solution[0]:.5f}, Best Ki = {best_solution[1]:.5f}")
    
    # Plot convergence
    plt.figure()
    plt.plot(best_fitness_over_time)
    plt.title('Grafik Konvergensi Optimasi dengan Algoritma ABC')
    plt.xlabel('Iterasi')
    plt.ylabel('Best Fitness (ITAE)')
    plt.grid(True)
    plt.show()
    
    return best_solution

# Run optimization
population_size = 100
max_iter = 100
disturbance = np.random.uniform(-np.pi / 4, np.pi / 4)  # Random disturbance between -45° and 45°
best_Kp, best_Ki = artificial_bee_colony(population_size, max_iter, disturbance)

print(f"Optimal Parameters: Kp = {best_Kp:.4f}, Ki = {best_Ki:.4f}")

# Simulate system with optimal parameters
theta_best = simulate_pendulum(best_Kp, best_Ki, disturbance)

# Create animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-1, 1)
ax.set_ylim(-0.5, 1)
ax.set_title("Simulasi Inverted Pendulum dengan Algoritma ABC Kontrol PI")
ax.set_xlabel("Posisi (m)")
ax.set_ylabel("Tinggi (m)")
ax.grid()

# Animation elements
line, = ax.plot([], [], 'o-', lw=2)
wheel_radius = 0.08
wheel = plt.Circle((0, 0), wheel_radius, color='red', fill=True)
ax.add_artist(wheel)

def init():
    """Initialize animation"""
    line.set_data([], [])
    wheel.set_center((0, 0))
    return line, wheel

def update(frame):
    """Update animation frame"""
    x_pendulum = length_pendulum * np.sin(theta_best[frame])
    y_pendulum = length_pendulum * np.cos(theta_best[frame])
    x_wheel = -x_pendulum
    
    line.set_data([x_wheel, x_pendulum], [0, y_pendulum])
    wheel.set_center((x_wheel, 0))
    
    return line, wheel

# Run animation
ani = FuncAnimation(fig, update, frames=len(time), init_func=init, blit=True, interval=dt * 1000)
plt.show()