import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the reference image you just created
img = mpimg.imread('reference_map.jpg')

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(img)
plt.title("Hover mouse over the TOP-LEFT corner of the minimap")

# This enables the coordinate display
plt.show()
