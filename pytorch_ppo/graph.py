import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file into a pandas DataFrame
file = pd.read_csv('./pytorch_ppo/BipedalWalker-v3.csv')

# Extract time_steps and rewards columns
time_steps = file['timestep']
rewards = file['reward']

# Create a scatter plot using Matplotlib
plt.scatter(time_steps, rewards)
plt.title('Rewards vs Time Steps')
plt.xlabel('Time Steps')
plt.ylabel('Rewards')
#plt.savefig('./pytorch_ppo/1.png')
plt.show()