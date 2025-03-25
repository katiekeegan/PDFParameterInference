import numpy as np
import matplotlib.pyplot as plt

# Load the loss values from the .npy file
losses = np.load('loss_values.npy')

# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Training Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('losses.png')