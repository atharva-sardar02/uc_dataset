import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Define constants
IMAGE_SIZE = (64, 64)


# Load images and labels
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Check if the image is loaded successfully
        if img is None:
            print(f"Error loading image: {img_path}")
            continue

        img = cv2.resize(img, IMAGE_SIZE)

        # Check resized image shape
        print(f"Resized Image Shape: {img.shape}")

        # Compute HOG features
        hog_features, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        hog_features = hog_features.flatten()

        images.append(hog_features)
        labels.append(1 if folder == 'full' else 0)  # 1 for full, 0 for free

    return np.array(images), np.array(labels)


full_images, full_labels = load_images_from_folder('full')
free_images, free_labels = load_images_from_folder('free')

# Combine data
X = np.concatenate([full_images, free_images], axis=0)
y = np.concatenate([full_labels, free_labels], axis=0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train SVM
svm_clf = SVC()
svm_clf.fit(X_train, y_train)

# Predictions on the test set
y_pred = svm_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

y_pred2 = svm_clf.predict()
