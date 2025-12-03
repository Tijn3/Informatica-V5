import tensorflow as tf
import matplotlib.pyplot as plt
 
mnist = tf.keras.datasets.mnist
(train_images, train_labels) , (test_images, test_labels) = mnist.load_data()
 
# Printing the shapes
print("train_images shape: ", train_images.shape)
print("train_labels shape: ", train_labels.shape)
print("test_images shape: ", test_images.shape)
print("test_labels shape: ", test_labels.shape)
 
# Displaying first 9 images of dataset
fig = plt.figure(figsize=(10,10))
 
nrows=3
ncols=3
for i in range(9):
  fig.add_subplot(nrows, ncols, i+1)
  plt.imshow(train_images[i])
  plt.title("Digit: {}".format(train_labels[i]))
  plt.axis(False)
plt.show()
 
 
# Converting image pixel values to 0 - 1
train_images = train_images / 255
test_images = test_images / 255
 
print("First Label before conversion:")
print(train_labels[0])
 
# Converting labels to one-hot encoded vectors
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
 
print("First Label after conversion:")
print(train_labels[0])
 
 
# Defining Model
# Using Sequential() to build layers one after another
model = tf.keras.Sequential([
 
  # Flatten Layer that converts images to 1D array
  tf.keras.layers.Flatten(),
   
  # Hidden Layer with 512 units and relu activation
  tf.keras.layers.Dense(units=512, activation='relu'),
   
  # Output Layer with 10 units for 10 classes and softmax activation
  tf.keras.layers.Dense(units=10, activation='softmax')
])
 
model.compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = ['accuracy']
)
 
history = model.fit(
  x = train_images,
  y = train_labels,
  epochs = 10
)
 
 
# Creating training progress visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training Loss over epochs
axes[0].plot(history.history['loss'], linewidth=2, color='#FF6B6B')
axes[0].set_xlabel('Epochs', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training Loss Progress', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(range(0, len(history.history['loss'])))

# Plot 2: Training Accuracy over epochs
axes[1].plot(history.history['accuracy'], linewidth=2, color='#4ECDC4')
axes[1].set_xlabel('Epochs', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Training Accuracy Progress', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(range(0, len(history.history['accuracy'])))
axes[1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('training_progress.png', dpi=100, bbox_inches='tight')
print("Training progress visualization saved as 'training_progress.png'")
 
 
# Call evaluate to find the accuracy on test images
test_loss, test_accuracy = model.evaluate(
  x = test_images, 
  y = test_labels
)
 
print("Test Loss: %.4f"%test_loss)
print("Test Accuracy: %.4f"%test_accuracy)
 
# Creating visualization of test metrics
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Test Loss and Accuracy comparison
metrics = ['Test Loss', 'Test Accuracy']
values = [test_loss, test_accuracy]
colors = ['#FF6B6B', '#4ECDC4']
bars = axes[0].bar(metrics, values, color=colors, width=0.6)
axes[0].set_ylabel('Value', fontsize=12)
axes[0].set_title('Test Metrics', fontsize=14, fontweight='bold')
axes[0].set_ylim([0, 1])

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 2: Model Performance Summary
performance_data = {
    'Correct': test_accuracy * 100,
    'Incorrect': (1 - test_accuracy) * 100
}
colors_pie = ['#4ECDC4', '#FF6B6B']
wedges, texts, autotexts = axes[1].pie(performance_data.values(), labels=performance_data.keys(),
                                         autopct='%1.2f%%', colors=colors_pie, startangle=90,
                                         textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Model Accuracy Breakdown', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('test_metrics.png', dpi=100, bbox_inches='tight')
print("\nTest metrics visualization saved as 'test_metrics.png'")
 
# Making Predictions
predicted_probabilities = model.predict(test_images)
predicted_classes = tf.argmax(predicted_probabilities, axis=-1).numpy()
 
index=11
 
# Showing image
plt.imshow(test_images[index])
 
# Printing Probabilities
print("Probabilities predicted for image at index", index)
print(predicted_probabilities[index])
 
print()
 
# Printing Predicted Class
print("Probabilities class for image at index", index)
print(predicted_classes[index])
