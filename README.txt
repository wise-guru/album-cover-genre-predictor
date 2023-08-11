The objective of this project is to develop a machine learning solution that can accurately predict the genre of music based on the album cover artwork. By utilizing
deep learning techniques and neural networks, the program is trained to analyze visual patterns and features in album
covers to determine the most suitable genre.


To run:
Create your virtual environment (the venv) folder and download the packages in requirement.txt. If you're not automatically prompted by the IDE, attempt to run main.py and the option will appear.

Sample prompts:

Melodrama by Lorde (pop album) Cover:
https://upload.wikimedia.org/wikipedia/en/b/b2/Lorde_-_Melodrama.png

Room on Fire by The Strokes (rock album) Cover:
https://upload.wikimedia.org/wikipedia/en/9/9f/Room_on_Fire_cover.jpg

Straight Outta Compton by N.W.A. (hipHop album) Cover:
https://upload.wikimedia.org/wikipedia/en/8/84/StraightOuttaComptonN.W.A..jpg

Red by Taylor Swift (country album) Cover:
https://upload.wikimedia.org/wikipedia/en/e/e8/Taylor_Swift_-_Red.png

Discovery by Daft Punk (electronic album) Cover:
https://upload.wikimedia.org/wikipedia/en/2/27/Daft_Punk_-_Discovery.png

To find others like these, you can go to the wikipedia page for an album, right-click the album cover, and select "open image in new tab".








## Datasets

The raw data used in the machine learning solution is a collection of album cover images along with their corresponding genres. The raw data can be in the form of image files and a CSV file containing information about the file paths and genres.



To process the raw data, several steps are performed:



Loading the Dataset:

The CSV file is read using the pandas library to obtain the file paths and genres associated with each album cover image.





Preprocessing the Images:

Each image is loaded using OpenCV (cv2.imread) and then resized to the desired dimensions using cv2.resize. The pixel values of the resized image are normalized to the range [0, 1] by dividing by 255.0. Finally, the normalized image is flattened into a feature vector.



Label Encoding:

The genre labels are encoded into numerical format using scikit-learn's LabelEncoder. This conversion assigns a unique numerical value to each genre class, allowing the machine learning model to work with categorical data.



The processed data consists of two main components:

Preprocessed Image Data:

The preprocessed image data is a NumPy array where each row represents an album cover image that has been resized, normalized, and flattened into a feature vector. This data is used as input to train and evaluate the machine learning model.



Encoded Labels:

The genre labels are encoded into numerical format using LabelEncoder. The encoded labels are also represented as a NumPy array, corresponding to the genre classes associated with each album cover image.



Access to Datasets:

Album Cover Images:

The raw album cover images are organized by genre, which is a subset of a Kagglle.com dataset. They are stored locally in the application, but also are available online.

https://www.kaggle.com/datasets/michaeljkerr/20k-album-covers-within-20-genres

Dataset CSV:

The CSV file containing file paths and genres associated with the album cover images are found in the application. The file path should be relative to the location of the CSV file.

Data Product Code



Processing raw data:

The raw data consists of album cover images.

The code preprocesses the raw images by resizing them to a specific dimension, normalizing the pixel values, and flattening them into a feature vector.

This processing step is necessary to ensure consistent input dimensions for the machine learning model and to transform the image data into a suitable format for classification.



Descriptive methods and visualizations:

The code includes several descriptive methods and visualizations:

Bar Plot for Genre Distribution: Visualizes the distribution of genres in the dataset.

Confusion Matrix: Visualizes the performance of the classification model by showing the predicted versus true genre labels.

Precision-Recall Curve: Plots the precision-recall curves for each genre, providing insights into the model's performance for different classification thresholds.



Non-descriptive method(s):

The non-descriptive method applied is a Random Forest classifier.

Random Forest is an ensemble learning method that combines multiple decision trees to make predictions. It is appropriate for this project because it can handle multi-class classification tasks, handle high-dimensional feature vectors, and has the potential to capture complex relationships between album cover images and music genres.

The Random Forest classifier is trained using the preprocessed image data and the corresponding genre labels. It is then tested on the testing subset of the dataset.

The evaluation metrics (accuracy, precision, recall, F1 score) are calculated to assess the performance of the Random Forest model.



The data analysis supports the choice and improvement of descriptive and non-descriptive methods in the following ways:

Descriptive methods and visualizations provide insights into the dataset's characteristics and distribution of genres, helping to understand the data's structure and make informed decisions during model selection and evaluation.

The confusion matrix visualizes the model's performance, highlighting potential misclassifications and areas for improvement.

The precision-recall curve provides a more detailed analysis of the model's performance, particularly for imbalanced classes, and helps in setting an appropriate classification threshold.



Objective (or Hypothesis) Verification



The project's objective was to develop a music genre classification application based on album cover images and achieve a certain level of accuracy.

However, based on the statistics provided, the objective was not met. The accuracy achieved did not go above 25%, which indicates that the model correctly classified approximately a quarter or less of the album cover images. The precision, recall, and F1 score are also relatively low, indicating that the model's performance in classifying the different genres was not satisfactory.

Several factors, such as the complexity of the task, insufficient training data, feature representation, and model selection, may have contributed to the low performance.

 To improve the objective, steps such as data augmentation, acquiring more labeled data, exploring advanced feature extraction methods like CNNs, and optimizing model hyperparameters can be taken to enhance the accuracy and overall performance of the classification application.

Effective Visualization and Reporting



The descriptive methods and visualizations played a crucial role in supporting the development process of the non-descriptive methods. Here's how they supported various aspects (and the examples of each):

Data Exploration:

The bar plot for genre distribution provided insights into the distribution of album covers across different genres. This helped in understanding the class distribution and identifying any potential data imbalances.

This visualization showed the count of album covers for each genre, providing an overview of the dataset's genre distribution. It helped identify if there were any class imbalances that could potentially affect the model's training and performance.







Data Analysis:

The confusion matrix visualized the performance of the classification model by showing the true and predicted genre labels. It provided an overview of the model's accuracy and highlighted the areas where misclassifications were occurring.

The confusion matrix displayed a grid of true and predicted genre labels. The cells in the matrix represented the number of samples that were correctly or incorrectly classified for each genre. It allowed for a detailed analysis of the model's performance, highlighting the specific genres where misclassifications were occurring.