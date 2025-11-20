import tkinter as tk
from tinygrad.tensor import Tensor
import numpy as np


# --------------------------------------
# Markov Chain Model (using tinygrad)
# --------------------------------------
class MarkovChain:
    def __init__(self, n_states=3):
        self.n = n_states

        # Store transition matrix as numpy (avoid tinygrad in UI loop)
        self.P = np.full((self.n, self.n), 1/self.n, dtype=np.float32)

        # Start in state 0
        self.state = np.array([1.0] + [0.0]*(self.n-1), dtype=np.float32)

    def step(self):
        """One probabilistic forward step (no sampling)."""
        self.state = self.state @ self.P
        return self.state

    def sample_step(self):
        """Sample a real state using probability distribution."""
        current_state_idx = int(np.argmax(self.state))
        probs = self.P[current_state_idx]
        probs = probs / probs.sum()  # ensure normalization
        nxt = np.random.choice(self.n, p=probs)
        new_state = np.zeros(self.n, dtype=np.float32)
        new_state[nxt] = 1.0
        self.state = new_state
        return nxt

    def reset(self):
        """Reset to state 0."""
        self.state = np.array([1.0] + [0.0]*(self.n-1), dtype=np.float32)

    def update_transition(self, i, j, val):
        # Update transition matrix
        self.P[i, j] = val
        # normalize row so probabilities sum to 1
        self.P[i] = self.P[i] / self.P[i].sum()


# --------------------------------------
# UI: Markov Chain Explorer
# --------------------------------------
class MarkovUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎲 Weather Game - Let's Play!")
        self.root.configure(bg="#FFF8DC")

        self.mc = MarkovChain(n_states=3)

        # --- Title ---
        title = tk.Label(root, text="☀️  ☁️  🌧️\nWeather Game!", 
                        font=("Arial", 24, "bold"), bg="#FFF8DC", fg="#C41E3A")
        title.pack(pady=10)

        # --- Emoji States ---
        self.state_emojis = ["☀️ Sunny", "☁️ Cloudy", "🌧️ Rainy"]

        # --- State display with emoji ---
        self.state_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#FFF8DC", fg="#000")
        self.state_label.pack(pady=10)

        # --- Action Buttons ---
        button_frame = tk.Frame(root, bg="#FFF8DC")
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="⏭️  Next Day", command=self.step, 
                 font=("Arial", 12), bg="#4ECDC4", fg="white", width=15, padx=10, pady=5).pack(pady=5)
        tk.Button(button_frame, text="🎲 Random Jump", command=self.sample_step, 
                 font=("Arial", 12), bg="#95E1D3", fg="white", width=15, padx=10, pady=5).pack(pady=5)
        tk.Button(button_frame, text="🔄 Start Over", command=self.reset, 
                 font=("Arial", 12), bg="#FFB6C1", fg="white", width=15, padx=10, pady=5).pack(pady=5)

        # --- Pattern Presets ---
        tk.Label(root, text="Try These Patterns:", font=("Arial", 12, "bold"), bg="#FFF8DC", fg="#000").pack(pady=(10, 5))

        preset_frame = tk.Frame(root, bg="#FFF8DC")
        preset_frame.pack(pady=5)

        tk.Button(preset_frame, text="Always Sunny ☀️", command=lambda: self.bias_transitions(0),
                 font=("Arial", 10), bg="#FFD93D", fg="#000", width=14).grid(row=0, column=0, padx=5, pady=2)
        tk.Button(preset_frame, text="Cloudy ☁️", command=lambda: self.bias_transitions(1),
                 font=("Arial", 10), bg="#95B8D1", fg="white", width=14).grid(row=0, column=1, padx=5, pady=2)
        tk.Button(preset_frame, text="Rainy Days 🌧️", command=lambda: self.bias_transitions(2),
                 font=("Arial", 10), bg="#4A90E2", fg="white", width=14).grid(row=1, column=0, padx=5, pady=2)
        tk.Button(preset_frame, text="Mixed 🎨", command=self.uniform_transitions,
                 font=("Arial", 10), bg="#9B59B6", fg="white", width=14).grid(row=1, column=1, padx=5, pady=2)

        # --- Advanced Matrix Editor (collapsible) ---
        tk.Label(root, text="Advanced: Edit Probabilities", font=("Arial", 10, "bold"), bg="#FFF8DC", fg="#000").pack(pady=(15, 5))

        matrix_label = tk.Label(root, text="How likely is each weather tomorrow?\n(each row must add to 1.0)", 
                              font=("Arial", 9), bg="#FFF8DC", fg="#333")
        matrix_label.pack()

        self.entries = []
        frame = tk.Frame(root, bg="#FFF8DC")
        frame.pack(pady=5)

        # Matrix header
        tk.Label(frame, text="From →", font=("Arial", 9, "bold"), bg="#FFF8DC", fg="#000").grid(row=0, column=0)
        for j, emoji in enumerate(self.state_emojis):
            tk.Label(frame, text=emoji, font=("Arial", 10), bg="#FFF8DC", fg="#000").grid(row=0, column=j+1, padx=5)

        for i in range(3):
            tk.Label(frame, text=self.state_emojis[i], font=("Arial", 10), bg="#FFF8DC").grid(row=i+1, column=0, padx=5)
            row = []
            for j in range(3):
                e = tk.Entry(frame, width=5, font=("Arial", 10), justify="center")
                e.insert(0, f"{1/3:.2f}")
                e.grid(row=i+1, column=j+1, padx=3, pady=3)
                row.append(e)
            self.entries.append(row)

        tk.Button(root, text="Apply Changes", command=self.update_matrix,
                 font=("Arial", 10), bg="#2ECC71", fg="white", padx=10, pady=5).pack(pady=10)

        self.update_display()

    def update_display(self):
        # Show current state distribution with emojis and visual bar
        st = self.mc.state
        text = "Today's weather chances:\n\n"
        for i, (emoji_name, prob) in enumerate(zip(self.state_emojis, st)):
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            text += f"{emoji_name}: {bar} {prob*100:.0f}%\n"
        self.state_label.config(text=text)

    def step(self):
        """Move to next day with deterministic progression."""
        self.mc.step()
        self.update_display()

    def sample_step(self):
        """Jump to a random weather state."""
        state_idx = self.mc.sample_step()
        self.update_display()

    def reset(self):
        """Start from sunny day again."""
        self.mc.reset()
        self.update_display()

    def update_matrix(self):
        # Read entries, update transition matrix
        for i in range(3):
            for j in range(3):
                try:
                    val = float(self.entries[i][j].get())
                except:
                    val = 0.0
                self.mc.P[i, j] = val

            # renormalize row
            self.mc.P[i] = self.mc.P[i] / self.mc.P[i].sum()

        self.update_entries()
        self.update_display()

    def bias_transitions(self, state):
        """Bias all transitions toward a specific state."""
        for i in range(3):
            self.mc.P[i] = np.array([0.1, 0.1, 0.1], dtype=np.float32)
            self.mc.P[i, state] = 0.7
            self.mc.P[i] = self.mc.P[i] / self.mc.P[i].sum()
        self.update_entries()
        self.update_display()

    def uniform_transitions(self):
        """Reset to uniform transitions (equal probability)."""
        self.mc.P = np.full((3, 3), 1/3, dtype=np.float32)
        self.update_entries()
        self.update_display()

    def update_entries(self):
        """Update entry widgets to reflect current transition matrix."""
        for i in range(3):
            for j in range(3):
                self.entries[i][j].delete(0, tk.END)
                self.entries[i][j].insert(0, f"{self.mc.P[i, j]:.2f}")


# --------------------------------------
# Run the UI
# How the buttons update the transition matrix:
#
# 1. "Next Day" - Uses matrix multiplication: state @ P
#    The transition matrix P tells us the probability of moving from
#    one weather state to another. Multiplying the current state by P
#    gives us tomorrow's probabilities!
#
# 2. "Random Jump" - Samples from current state's row in matrix
#    If we're sunny, the first row tells us: 70% chance sunny, 20% cloud, 10% rain.
#    We randomly pick one outcome based on these probabilities.
#
# 3. Preset buttons (Always Sunny, Cloudy, etc.) - Change the transition matrix:
#    These buttons modify P[i,j] values to bias toward a specific weather.
#    For example, "Always Sunny" makes P[i,0] = 0.7 for all rows i.
#    This means no matter what the weather is, there's a 70% chance of sunny!
#
# 4. "Apply Changes" - Reads the entry boxes and updates the matrix
#    Each entry P[i,j] = probability of going from state i to state j
#    The numbers in each row must add up to 1.0 (normalized)
# --------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = MarkovUI(root)
    root.geometry("650x900")
    root.mainloop()
