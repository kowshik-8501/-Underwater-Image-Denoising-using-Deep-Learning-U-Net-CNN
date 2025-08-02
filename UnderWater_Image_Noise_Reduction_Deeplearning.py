# -----------------------------------------
# 1. Import Required Libraries
# -----------------------------------------
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, InputLayer
from tensorflow.keras.optimizers import Adam
from PIL import Image

# -----------------------------------------
# 2. Load and Display Images from Dataset
# -----------------------------------------

# Function to load and resize images
def load_images(path, img_size=(128, 128)):
    images = []
    image_names = []
    for file in os.listdir(path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(path, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size) / 255.0  # Normalize
            images.append(img)
            image_names.append(file)
    return np.array(images), image_names

# Dataset path
data_path = "/content/drive/MyDrive/raw-890"
raw_images, image_names = load_images(data_path)
print(f"Loaded {len(raw_images)} images from {data_path}")

# Displaying multiple images
num_samples = min(5, len(raw_images))  # Display 5 or fewer if dataset is smaller
fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

for i in range(num_samples):
    axes[i].imshow(raw_images[i])
    axes[i].set_title(f"Image {i+1}: {image_names[i]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# -----------------------------------------
# 3. Add Synthetic Noise to Images
# -----------------------------------------
def add_noise(images, noise_factor=0.05):
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    noisy_images = np.clip(noisy_images, 0.0, 1.0)  # Clip pixel values between 0 and 1
    return noisy_images

# Create noisy images
noisy_images = add_noise(raw_images)
print("Noisy images generated.")

# Display noisy vs. original images
fig, axes = plt.subplots(2, num_samples, figsize=(15, 5))

for i in range(num_samples):
    # Original images
    axes[0, i].imshow(raw_images[i])
    axes[0, i].set_title("Original")
    axes[0, i].axis('off')

    # Noisy images
    axes[1, i].imshow(noisy_images[i])
    axes[1, i].set_title("Noisy")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()

# -----------------------------------------
# 4. Train-Test Split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(noisy_images, raw_images, test_size=0.2, random_state=42)
print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

# -----------------------------------------
# 5. Define the Denoising Model
# -----------------------------------------
def build_denoising_model():
    model = Sequential()

    # Encoder
    model.add(InputLayer(input_shape=(128, 128, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    # Decoder
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))  # Output layer

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    return model

# -----------------------------------------
# 6. Train the Model
# -----------------------------------------
model = build_denoising_model()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# -----------------------------------------
# 7. Save the Trained Model
# -----------------------------------------
output_model_path = "output_model"
os.makedirs(output_model_path, exist_ok=True)

model.save(os.path.join(output_model_path, "underwater_denoising_model.h5"))
print(f"Model saved at: {os.path.join(output_model_path, 'underwater_denoising_model.h5')}")

# -----------------------------------------
# 8. Visualize the Denoising Results
# -----------------------------------------
num_samples = 5
denoised_images = model.predict(X_test[:num_samples])

plt.figure(figsize=(15, 5))

for i in range(num_samples):
    # Noisy Image
    plt.subplot(3, num_samples, i + 1)
    plt.imshow(X_test[i])
    plt.title("Noisy")
    plt.axis('off')

    # Original Clean Image
    plt.subplot(3, num_samples, i + num_samples + 1)
    plt.imshow(y_test[i])
    plt.title("Original")
    plt.axis('off')

    # Denoised Output
    plt.subplot(3, num_samples, i + 2 * num_samples + 1)
    plt.imshow(denoised_images[i])
    plt.title("Denoised")
    plt.axis('off')

plt.tight_layout()
plt.show()

# -----------------------------------------
# 9. Save Denoised Images (Optional)
# -----------------------------------------
save_images = True

if save_images:
    denoised_folder = "denoised_images"
    os.makedirs(denoised_folder, exist_ok=True)

    for i, img in enumerate(denoised_images):
        img = (img * 255).astype('uint8')  # Convert to uint8 format
        cv2.imwrite(os.path.join(denoised_folder, f"denoised_{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"Denoised images saved in {denoised_folder}")

# Print dataset path and images
print(f"Images in dataset: {data_path}")
