import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.preprocessing import label_binarize

# Define the genres
genres = [
    'Country',
    'Electronic',
    'HipHop',
    'Pop',
    'Rock',
]

# Set the path to the CSV file containing the file paths and genres
csv_file = 'dataset.csv'

# Set the path to the directory containing the genre folders
data_path = 'Album_Covers'

# Define the image dimensions
image_width, image_height = 300, 300


# Function to preprocess an image
def preprocess_image(image):
    # Resize the image to the desired dimensions
    resized_image = cv2.resize(image, (image_width, image_height))

    # Normalize pixel values to the range [0, 1]
    normalized_image = resized_image.astype(np.float32) / 255.0

    # Flatten the image into a feature vector
    flattened_image = normalized_image.flatten()

    return flattened_image




# Create empty lists for the image data and corresponding labels
data = []
labels = []

# Load file paths and genres from the CSV file
file_paths_df = pd.read_csv(csv_file)

# Iterate over the rows of the DataFrame
for _, row in file_paths_df.iterrows():
    # Get the file path and genre from the row
    file_path = row['File_path']
    genre = row['Genre']

    # Set the path to the image file
    image_path = file_path

    # Preprocess the image and add it to the data list
    preprocessed_image = preprocess_image(cv2.imread(image_path))
    data.append(preprocessed_image)

    # Add the corresponding label to the labels list
    labels.append(genre)

# Convert the data and labels lists to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Print the processed data
# print("Processed Image Data:")
# print(data[:5])  # Print the first 5 rows of preprocessed image data
# print("Encoded Labels:")
# print(labels[:5])  # Print the first 5 encoded labels

# Split the dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Perform label encoding on the target labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Create an instance of the Random Forest classifier
model = RandomForestClassifier()

# Train the model using the training data
model.fit(X_train, y_train_encoded)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Convert the predicted labels back to genre names
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_labels)
precision = precision_score(y_test, y_pred_labels, average='weighted')
recall = recall_score(y_test, y_pred_labels, average='weighted')
f1 = f1_score(y_test, y_pred_labels, average='weighted')

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# Function to predict the genre of a single input image
def predict_genre(image_url):
    # Download the image from the provided URL
    response = requests.get(image_url)
    image_bytes = BytesIO(response.content)

    # Load and preprocess the image using OpenCV
    image = cv2.imdecode(np.asarray(bytearray(image_bytes.read()), dtype=np.uint8), 1)
    preprocessed_image = preprocess_image(image)

    # Reshape the preprocessed image to match the input shape of the model
    preprocessed_image = preprocessed_image.reshape(1, -1)

    # Make a prediction using the trained model
    prediction = model.predict(preprocessed_image)

    # Convert the predicted label back to a genre name
    predicted_genre = label_encoder.inverse_transform(prediction)

    return predicted_genre[0]


# User input procedure
def user_input():
    while True:
        # Get the image URL from the user
        image_url = input("Enter the URL of an image: ")

        try:
            # Predict the genre of the input image
            predicted_genre = predict_genre(image_url)

            # Print the predicted genre
            print("Predicted genre:", predicted_genre)

        except Exception as e:
            # Handle any errors that may occur (e.g., invalid URL or image format)
            print("Error:", str(e))

        # Ask the user if they want to continue
        continue_input = input("Do you want to continue? (yes/no): ")

        # Check if the user wants to continue
        if continue_input.lower() != "yes":
            break


# Bar Plot for Genre Distribution
genre_counts = np.unique(labels, return_counts=True)
genres = genre_counts[0]
counts = genre_counts[1]

plt.figure(figsize=(8, 6))
plt.bar(genres, counts)
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Genre Distribution')
plt.show()

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=genres, yticklabels=genres)
plt.xlabel('Predicted Genre')
plt.ylabel('True Genre')
plt.title('Confusion Matrix')
plt.show()

# Convert the true labels to binary format
y_test_bin = label_binarize(y_test_encoded, classes=np.arange(len(genres)))
y_pred_bin = label_binarize(y_pred, classes=np.arange(len(genres)))

# Compute precision and recall for each class
precision = dict()
recall = dict()
for i in range(len(genres)):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_bin[:, i])

# Plot the precision-recall curves for each class
plt.figure(figsize=(8, 6))
for i in range(len(genres)):
    plt.plot(recall[i], precision[i], label=genres[i])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

user_input()
