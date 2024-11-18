# LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import csv
import random


class Bandit(ABC):
    # ==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass


# --------------------------------------#

class Visualization:
    @staticmethod
    def plot1(greedy_rewards, thompson_rewards):
        plt.plot(greedy_rewards, label="Epsilon Greedy Algorithm")
        plt.plot(thompson_rewards, label="Thompson Sampling")
        plt.title("Learning Process")
        plt.xlabel("Trials")
        plt.ylabel("Reward")
        plt.legend()
        plt.show()

    @staticmethod
    def plot2(e_greedy_rewards, thompson_rewards):
        cumulative_e_greedy_rewards = [
            sum(e_greedy_rewards[: i + 1]) for i in range(len(e_greedy_rewards))
        ]
        cumulative_thompson_rewards = [
            sum(thompson_rewards[: i + 1]) for i in range(len(thompson_rewards))
        ]
        plt.plot(cumulative_e_greedy_rewards, label="Epsilon Greedy Algorithm")
        plt.plot(cumulative_thompson_rewards, label="Thompson Sampling")
        plt.title("Cumulative Rewards")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.show()


class EpsilonGreedy(Bandit):
    NUM_TRIALS = 10000
    EPS = 0.1
    BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

    def __init__(self, m: float):
        self.m = m
        self.m_estimate = 0
        self.N = 0
        self.rewards = []  # Initialize rewards
        self.regrets = []  # Initialize regrets

    def pull(self):
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        self.m_estimate = ((self.N - 1) * self.m_estimate + x) / self.N

    def __repr__(self):
        return f"An Arm with {self.m} Reward"

    @classmethod
    def experiment(cls, BANDIT_PROBABILITIES=None, NUM_TRIALS=None, EPS=0.1, min_epsilon=0.02):
        logger.info("Initializing the Epsilon Greedy algorithm experiment.")

        BANDIT_PROBABILITIES = BANDIT_PROBABILITIES or cls.BANDIT_PROBABILITIES
        NUM_TRIALS = NUM_TRIALS or cls.NUM_TRIALS

        bandits = [EpsilonGreedy(p) for p in BANDIT_PROBABILITIES]
        rewards = []
        regrets = []

        for i in range(NUM_TRIALS):
            if np.random.random() < max(EPS / (i // 100 + 1), min_epsilon):
                chosen_bandit = np.random.randint(len(bandits))
            else:
                chosen_bandit = np.argmax([b.m_estimate for b in bandits])

            # Optimal reward calculation
            reward = bandits[chosen_bandit].pull()
            optimal_reward = max(b.m for b in bandits)
            regret = optimal_reward - reward

            bandits[chosen_bandit].update(reward)
            rewards.append(reward)
            regrets.append(regret)

        return bandits, rewards, regrets

    def report(self, rewards, regrets):
        mean_reward = np.mean(rewards)
        mean_regret = np.mean(regrets)

        logger.info(f"Average Reward for Epsilon-Greedy: {mean_reward}")
        logger.info(f"Average Regret for Epsilon-Greedy: {mean_regret}")

        with open("epsilon_greedy_report.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Trial", "Reward", "Regret"])  # Header
            for i, (reward, regret) in enumerate(zip(rewards, regrets), start=1):
                writer.writerow([i, reward, regret])

        logger.info("Data saved to 'epsilon_greedy_report.csv'.")


# --------------------------------------#


class ThompsonSampling(Bandit):
    def __init__(self, p):
        self.p = p
        self.alpha = np.ones(len(p))
        self.beta = np.ones(len(p))
        self.rewards = []
        self.regrets = []

    def __repr__(self):
        return f"ThompsonSampling(probabilities={self.p})"

    def pull(self):
        samples = [random.betavariate(float(self.alpha[i]), float(self.beta[i])) for i in range(len(self.p))]
        return np.argmax(samples)

    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward

    def experiment(self, n):
        for _ in range(n):
            arm = self.pull()
            reward = np.random.random() < self.p[arm]  # Generate reward (1 or 0)
            regret = max(self.p) - reward

            self.rewards.append(reward)
            self.regrets.append(regret)
            self.update(arm, reward)

    def report(self):
        mean_reward = np.mean(self.rewards)
        mean_regret = np.mean(self.regrets)

        logger.info(f"Average Reward for Thompson Sampling: {mean_reward}")
        logger.info(f"Average Regret for Thompson Sampling: {mean_regret}")

        with open("thompson_sampling_report.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Trial", "Reward", "Regret"])  # Header
            for i, (reward, regret) in enumerate(zip(self.rewards, self.regrets), start=1):
                writer.writerow([i, reward, regret])

        logger.info("Data saved to 'thompson_sampling_report.csv'.")


def comparison():
    e_bandits, e_rewards, e_regrets = EpsilonGreedy.experiment()
    t_bandits = ThompsonSampling([0.2, 0.5, 0.75])
    t_bandits.experiment(10000)

    Visualization.plot1(e_rewards, t_bandits.rewards)
    Visualization.plot2(e_rewards, t_bandits.rewards)


if __name__ == '__main__':
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

comparison()
