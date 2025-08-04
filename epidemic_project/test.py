import numpy as np
import pandas as pd
import matplotlib
# Fix the display issue by using a non-interactive backend
matplotlib.use('Agg')  # This line fixes the error!
import matplotlib.pyplot as plt

print("Setup successful!")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# Create a simple test plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o', linewidth=2, markersize=8)
plt.title("Test Plot - Setup Working!", fontsize=16)
plt.xlabel("X values", fontsize=12)
plt.ylabel("Y values", fontsize=12)
plt.grid(True, alpha=0.3)

# Save the plot instead of trying to display it
plt.savefig('test_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'test_plot.png' in your project folder!")
print("Open the file to see your first successful plot!")

# Also test some basic data operations
data = pd.DataFrame({
    'day': [1, 2, 3, 4, 5],
    'infected': [10, 25, 60, 120, 200]
})
print("\nSample epidemic data:")
print(data)

# Calculate growth rate
data['growth_rate'] = data['infected'].pct_change() * 100
print("\nWith growth rate calculated:")
print(data)

print("\n🎉 Congratulations! Your setup is working perfectly!")
print("Next step: We'll create your first epidemic simulation!")