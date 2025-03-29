import numpy as np
import matplotlib.pyplot as plt
import os

class RandomWalkEnv:
    """
    A class to simulate the 1000-state random walk environment.
    
    - States are numbered from 1 to 1000.
    - The agent starts at state 500 and randomly moves left or right by 1–100 states.
    - If the agent moves past state 1 or 1000, the episode terminates.
    - Reward is -1 for terminating on the left, +1 for terminating on the right, and 0 otherwise.
    """

    def __init__(self, num_states=1000, max_jump=100):
        """
        Initializes the environment.

        Args:
            num_states (int): Total number of states in the environment.
            max_jump (int): Maximum number of states the agent can jump left or right.
        """
        self.num_states = num_states
        self.max_jump = max_jump
        self.start_state = num_states // 2  # Start at the middle (state 500)
        self.reset()

    def reset(self):
        """
        Resets the environment to the starting state.

        Returns:
            int: The starting state.
        """
        self.state = self.start_state
        return self.state

    def step(self):
        """
        Takes one step in the environment based on random dynamics.

        Returns:
            tuple: (next_state, reward, done)
                - next_state (int): The next state after the jump.
                - reward (int): The reward received after the transition.
                - done (bool): True if the episode has terminated.
        """
        # Sample a jump size between 1 and max_jump
        step = np.random.randint(1, self.max_jump + 1)

        # Randomly decide direction: left or right
        if np.random.rand() < 0.5:
            step = -step
        next_state = self.state + step

        # Handle termination cases
        if next_state < 1:
            reward = -1
            done = True
        elif next_state > self.num_states:
            reward = +1
            done = True
        else:
            reward = 0
            done = False

        # Update the internal state
        self.state = next_state
        return next_state, reward, done

def estimate_true_value_function(env, num_states=1000, num_rollouts=1000):
    """
    Estimate the true value function v_pi(s) using Monte Carlo rollouts.

    Args:
        env (RandomWalkEnv): The environment instance.
        num_states (int): Number of states to evaluate.
        num_rollouts (int): Number of episodes to simulate per state.

    Returns:
        np.ndarray: Array of shape (num_states,) with estimated values.
    """
    values = np.zeros(num_states)

    for s in range(1, num_states + 1):
        returns = []

        for _ in range(num_rollouts):
            env.state = s
            done = False
            while not done:
                next_state, reward, done = env.step()
            returns.append(reward)  # final reward is the return

        values[s - 1] = np.mean(returns)

    return values

def state_to_feature(state, num_states=1000, num_groups=10):
    """
    Converts a state to a one-hot feature vector based on its group.
    Each group corresponds to 100 states (1000 / 10).
    """
    group_size = num_states // num_groups
    group_idx = (state - 1) // group_size  # zero-based indexing
    features = np.zeros(num_groups)
    features[group_idx] = 1.0
    return features

def gradient_mc_prediction(env, feature_fn, alpha, num_episodes=5000, num_states=1000):
    """
    Gradient Monte Carlo prediction using a custom feature function.

    Args:
        env: The RandomWalkEnv instance.
        feature_fn: Function that maps a state to a feature vector.
        alpha: Step-size (learning rate).
        num_episodes: Number of episodes to run.
        num_states: Total number of states.

    Returns:
        tuple:
            np.ndarray: Estimated value function for each state.
            np.ndarray: Final learned weights.
    """
    sample_phi = feature_fn(1)  # Get feature size from one example
    w = np.zeros(len(sample_phi))

    for ep in range(num_episodes):
        states_visited = []
        state = env.reset()

        done = False
        while not done:
            states_visited.append(state)
            state, reward, done = env.step()

        G = reward  # return is either +1 or -1 for all visited states

        for s in states_visited:
            phi_s = feature_fn(s)
            v_hat = np.dot(w, phi_s)
            w += alpha * (G - v_hat) * phi_s

    # Compute estimated value for each of the 1000 states
    estimated_values = np.zeros(num_states)
    for s in range(1, num_states + 1):
        phi_s = feature_fn(s)
        estimated_values[s - 1] = np.dot(w, phi_s)

    return estimated_values, w

def state_to_polynomial_features(state, order=5, num_states=1000):
    """
    Converts a state to a normalized polynomial feature vector.

    Args:
        state (int): The current state (1-indexed).
        order (int): The degree of the polynomial basis (default is 5).
        num_states (int): Total number of states (used for normalization).

    Returns:
        np.ndarray: Feature vector of shape (order + 1,) containing
                    [1, x, x^2, ..., x^order], where x is the normalized state.
    """
    x = (state - 1) / (num_states - 1)  # Normalize state to [0, 1]
    return np.array([x**i for i in range(order + 1)])

def state_to_fourier_features(state, order=5, num_states=1000):
    """
    Converts a state to a normalized Fourier feature vector.

    Args:
        state (int): The current state (1-indexed).
        order (int): The order of the Fourier basis (default is 5).
        num_states (int): Total number of states (used for normalization).

    Returns:
        np.ndarray: Feature vector of shape (order + 1,) containing
                    [cos(0 * pi * x), cos(1 * pi * x), ..., cos(order * pi * x)],
                    where x is the normalized state in [0, 1].
    """
    x = (state - 1) / (num_states - 1)
    return np.array([np.cos(np.pi * i * x) for i in range(order + 1)])

# Create environment
env = RandomWalkEnv()

# Variables
num_states = 1000
start_state = 1

# True value function
if os.path.exists("true_values.npy"):
    true_values = np.load("true_values.npy")
    print("Loaded true value function from file.")
else:
    print("File not found. Estimating true value function...")
    true_values = estimate_true_value_function(env, num_states, num_rollouts=1000)
    np.save("true_values.npy", true_values)
    print("True value function saved to file.")

plt.figure(figsize=(10, 4))
plt.plot(true_values, label='True Value Function')
plt.xlabel('State')
plt.ylabel('Value')
plt.title('Estimated True Value Function $v_\pi(s)$')
plt.grid(True)
plt.xlim(-50, num_states+50)
plt.ylim(-start_state-.2, start_state+.2)
plt.legend()
plt.show()

# State aggregation
agg_estimated_values, agg_weights = gradient_mc_prediction(
    env=env,
    feature_fn=state_to_feature,
    alpha=2e-5,
    num_episodes=5000,
    num_states=num_states
)

plt.figure(figsize=(10, 4))
plt.plot(true_values, label='True Value Function')
plt.plot(agg_estimated_values, label='Estimated (State Aggregation)')
plt.xlabel('State')
plt.ylabel('Value')
plt.title('Gradient Monte Carlo with State Aggregation')
plt.xlim(-50, num_states+50)
plt.ylim(-start_state-.2, start_state+.2)
plt.legend()
plt.grid(True)
plt.show()

# Polynomial basis
pol_estimated_values, pol_weights = gradient_mc_prediction(
    env=env,
    feature_fn=lambda s: state_to_polynomial_features(s, order=5, num_states=num_states),
    alpha=1e-4,
    num_episodes=5000,
    num_states=num_states
)

plt.figure(figsize=(10, 4))
plt.plot(true_values, label='True Value Function')
plt.plot(pol_estimated_values, label='Estimated (Polynomial Basis)')
plt.xlabel('State')
plt.ylabel('Value')
plt.title('Gradient Monte Carlo with Polynomial Basis')
plt.xlim(-50, num_states+50)
plt.ylim(-start_state-.2, start_state+.2)
plt.legend()
plt.grid(True)
plt.show()

# Fourier basis
fourier_estimated_values, fourier_weights = gradient_mc_prediction(
    env=env,
    feature_fn=lambda s: state_to_fourier_features(s, order=5, num_states=num_states),
    alpha=5e-5,
    num_episodes=5000,
    num_states=num_states
)

plt.figure(figsize=(10, 4))
plt.plot(true_values, label='True Value Function')
plt.plot(fourier_estimated_values, label='Estimated (Fourrier Basis)')
plt.xlabel('State')
plt.ylabel('Value')
plt.title('Gradient Monte Carlo with Fourrier Basis')
plt.xlim(-50, num_states+50)
plt.ylim(-start_state-.2, start_state+.2)
plt.legend()
plt.grid(True)
plt.show()

# Comparison
plt.figure(figsize=(10, 4))
plt.plot(true_values, label='True Value Function', linewidth=2)
plt.plot(agg_estimated_values, label='State Aggregation', linestyle='--')
plt.plot(pol_estimated_values, label='Polynomial Basis', linestyle='-.')
plt.plot(fourier_estimated_values, label='Fourier Basis', linestyle=':')

plt.xlabel('State')
plt.ylabel('Value')
plt.title('Comparison of Value Function Approximations')
plt.xlim(-50, num_states+50)
plt.ylim(-start_state-.2, start_state+.2)
plt.legend()
plt.grid(True)
plt.show()