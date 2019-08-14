import gym
import numpy as np
import math
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt


class CartPoleQAgent:
    def __init__(self, buckets=(1, 1, 6, 12), num_episodes=1000, min_lr=0.1, min_epsilon=0.1, discount=1.0, decay=25):
        self.buckets = buckets
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay

        self.env = gym.make('CartPole-v0')

        # [position, velocity, angle, angular velocity]
        self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2],
                             math.radians(50) / 1.]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2],
                             -math.radians(50) / 1.]

        self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))
        self.Q1_table = np.zeros(self.buckets + (self.env.action_space.n,))

    def discretize_state(self, obs):
        discretized = list()
        for i in range(len(obs)):
            scaling = (obs[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i])
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        elif np.random.random() < self.epsilon:
            return np.argmax(self.Q_table[state])
        else:
            return np.argmax(self.Q1_table[state])

    def choose_best_action(self, state):
        if np.random.random() < self.epsilon:
            return np.argmax(self.Q_table[state])
        else:
            return np.argmax(self.Q1_table[state])

    def update_q(self, state, action, reward, new_state):
        if np.random.random() < 0.5:
            self.Q_table[state][action] += self.learning_rate * (
            reward + self.discount * np.max(self.Q1_table[new_state]) - self.Q_table[state][action])
            return self.Q_table[state][action]
        else:
            self.Q1_table[state][action] += self.learning_rate * (
                reward + self.discount * np.max(self.Q_table[new_state]) - self.Q1_table[state][action])
            return self.Q1_table[state][action]

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_learning_rate(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def train(self):
        steps = []
        for e in range(self.num_episodes):
            # track the total time steps in this episode
            time = 0

            current_state = self.discretize_state(self.env.reset())

            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            done = False

            while not done:
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                time += self.update_q(current_state, action, reward, new_state)
                current_state = new_state

            steps.append(time)

        self.saveperformance(steps)

    def saveperformance(self, steps):
        steps = np.add.accumulate(steps)
        plt.plot(steps, np.arange(1, len(steps) + 1))
        plt.xlabel('Reward accumulated')
        plt.ylabel('Episodes')

        plt.savefig('figure_QQAgCartPolePerformance.png')
        plt.close()

    def run(self, render):
        #self.env = gym.wrappers.Monitor(self.env, 'cartpole')
        t = 0
        done = False
        current_state = self.discretize_state(self.env.reset())
        while not done:
            if render:
                self.env.render()
            t = t + 1
            action = self.choose_best_action(current_state)
            obs, reward, done, _ = self.env.step(action)
            new_state = self.discretize_state(obs)
            current_state = new_state
        return t

def solve_cart_pole_agent(min_lr=0.1, min_epsilon=0.1, discount=1.0, decay=25, render = bool(0)):
    agent = CartPoleQAgent((1, 1, 6, 12), 1000, min_lr, min_epsilon, discount, decay)
    agent.train()
    c = 0 #cantidad de veces que resuelve el problema
    total_res = 1;
    for i in range(0,total_res):
        t = agent.run(render)
        if t > 195:#si pudo pasar 195 steps sin caerse el pole resolvio el problema
           c = c + 1
    return c/total_res# taza de resolucion

def optimization():
    hypScope = {
        'decay': (15, 30),
        'discount': (0.5, 2),
        'min_epsilon': (0.01, 0.15),
        'min_lr': (0.01, 0.15),
    }

    bo = BayesianOptimization(solve_cart_pole_agent, hypScope)

    bo.maximize()

    print(bo.max)

if __name__ == "__main__":

   #optimization()

   #(min_lr=0.1, min_epsilon=0.1, discount=1.0, decay=25, render = bool(0))
   #taza_resolucion = (solve_cart_pole_agent(0.1, 0.1, 1.0, 25, bool(1)))
   taza_resolucion = (solve_cart_pole_agent(0.01, 0.01, 1.027,  24.27, bool(1)))
   print(taza_resolucion)
