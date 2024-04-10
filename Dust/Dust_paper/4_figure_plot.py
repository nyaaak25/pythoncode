import matplotlib.pyplot as plt

# Sample data
data = [1, 2, 3, 4, 5, 5, 6, 6, 6, 7, 8, 9, 9, 9, 10]

# Plot histogram
plt.hist(data, bins=10)

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')

# Show the plot
plt.show()