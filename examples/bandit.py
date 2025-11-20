import numpy as np
import time
import os
from tinygrad.tensor import Tensor

# -------- ENVIRONMENT --------
class MultiArmedBandit:
    def __init__(self, means):
        self.means = np.array(means)

    def pull(self, arm):
        return np.random.randn() + self.means[arm]


# -------- AGENT --------
class BanditAgent:
    def __init__(self, n_arms, lr=0.1, policy="epsilon_greedy", epsilon=0.1, temperature=0.5):
        self.Q = Tensor.zeros(n_arms)
        self.lr = lr
        self.policy = policy
        self.epsilon = epsilon
        self.temperature = temperature
        self.n_arms = n_arms

    def select_action(self):
        if self.policy == "epsilon_greedy":
            return self.select_action_epsilon_greedy()
        return self.select_action_softmax()

    def select_action_epsilon_greedy(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        return int(self.Q.numpy().argmax())

    def select_action_softmax(self):
        probs = (self.Q / self.temperature).softmax().numpy()
        return np.random.choice(self.n_arms, p=probs)

    def update(self, action, reward):
        q_vals = self.Q.numpy().copy()
        td_error = reward - q_vals[action]
        q_vals[action] += self.lr * td_error
        self.Q = Tensor(q_vals)


# -------- ASCII VISUALIZATION --------
def ascii_bar(value, max_len=40):
    """Convert a number to a text bar."""
    value = max(0.0, float(value))
    length = int(value * max_len)
    return "#" * length


def run_ascii(env, agent, steps=600):

    rewards = []
    for t in range(1, steps + 1):
        action = agent.select_action()
        reward = env.pull(action)
        agent.update(action, reward)

        rewards.append(reward)
        avg_reward = sum(rewards[-50:]) / min(len(rewards), 50)

        # Clear screen
        os.system("clear" if os.name == "posix" else "cls")

        print(f"Step {t}/{steps}")
        print(f"Chosen arm: {action}")
        print(f"Reward: {reward:.3f}")
        print(f"Rolling avg reward (last 50): {avg_reward:.3f}")
        print("\nQ-values:")

        Q = agent.Q.numpy()
        maxQ = max(Q) if len(Q) else 1

        for i, q in enumerate(Q):
            bar = ascii_bar(q / maxQ if maxQ > 0 else 0)
            prefix = ">" if i == action else " "
            suffix = "<" if i == action else " "
            print(f"{prefix} arm {i}: {q:.3f} {suffix} |{bar}")

        time.sleep(0.02)   # slow enough to watch, fast enough to finish

    # Finished learning
    best_arm = int(agent.Q.numpy().argmax())

    print("\n" + "=" * 50)
    print(f"🎉 Finished {steps} steps!")
    print(f"🏆 Best arm found: arm {best_arm} (Q={agent.Q.numpy()[best_arm]:.3f})")
    print("=" * 50)


# -------- DEMO --------
if __name__ == "__main__":
    env = MultiArmedBandit([0.9, 0.1, 1.01, 1.1, 0.99, 0.0, 0.3, 1.0, 1.2])  # best arm = 3
    agent = BanditAgent(9, lr=0.1, policy="softmax", epsilon=0.1)
    run_ascii(env, agent, steps=6000)
