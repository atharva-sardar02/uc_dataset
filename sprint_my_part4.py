# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import warnings
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc

warnings.filterwarnings("ignore")


# Function to extract date taken from the metadata
def extract_date_taken(img_path):
    image = Image.open(img_path)
    exif_data = image._getexif()

    if exif_data is not None and 36867 in exif_data:
        return exif_data[36867]
    else:
        return None

# Function to create a DataFrame
def create_dataframe(folder, label):
    data = {'Image': [], 'Label': [], 'DateTaken': []}

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        date_taken = extract_date_taken(img_path)

        data['Image'].append(img_path)
        data['Label'].append(label)
        data['DateTaken'].append(date_taken)

    df = pd.DataFrame(data)
    return df

# Create DataFrames for full and free images
full_df = create_dataframe("full", "full")
free_df = create_dataframe("free", "free")

# Combine full and free DataFrames
df = pd.concat([full_df, free_df], ignore_index=True)


try:
    # Convert 'DateTaken' to datetime objects with the specified format
    df['DateTaken'] = pd.to_datetime(df['DateTaken'], format='%Y:%m:%d %H:%M:%S', errors='coerce')

    # Create a new column 'Date' with day of the year
    df['Date'] = df['DateTaken'].dt.dayofyear

    # Create a new column 'Time' with the time in minutes
    df['Time'] = df['DateTaken'].dt.hour * 60 + df['DateTaken'].dt.minute
except Exception as e:
    # Print the error and problematic values
    print(f"Error: {e}")
    problematic_values = df.loc[pd.to_datetime(df['DateTaken'], errors='coerce').isna(), 'DateTaken']
    print(f"Problematic values:\n{problematic_values}")

# Display the resulting DataFrame
print(df)
# Convert categorical variables to numerical using Label Encoding
le_label = LabelEncoder()
df['Label'] = le_label.fit_transform(df['Label'])

# Create categories for 'Time' and 'Date'
df['TimeCategory'] = pd.cut(df['Time'], bins=[0, 360, 720, 1080, 1440], labels=['0-6', '6-12', '12-18', '18-24'])
df['DayCategory'] = df['Date'].apply(lambda x: 'Weekend' if pd.to_datetime(x).weekday() >= 5 else 'Weekday')

# Convert categorical variables to numerical using Label Encoding for 'TimeCategory' and 'DayCategory'
le_time_category = LabelEncoder()
df['TimeCategory'] = le_time_category.fit_transform(df['TimeCategory'])
le_day_category = LabelEncoder()
df['DayCategory'] = le_day_category.fit_transform(df['DayCategory'])

# Separate features (X) and target variable (y)
X = df[['Date', 'TimeCategory', 'DayCategory']]
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

classification_table = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score',
                                             'Specificity', 'Cross Validation Mean Score'])

# Initialize the Decision Tree Classifier
dt_clf = DecisionTreeClassifier()
# Perform cross-validation
cv_scores_dt = cross_val_score(dt_clf, X, y, cv=5)
# Fit the model on the training data
dt_clf.fit(X_train, y_train)
# Make predictions on the test data
y_pred = dt_clf.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Decision Tree Accuracy: {accuracy:.2f}')
# Display classification report
print('Decision Tree Classification Report:')
print(classification_report(y_test, y_pred))
# Visualize the decision tree using matplotlib
plt.figure(figsize=(150,100))
plot_tree(dt_clf, filled=True, feature_names=X.columns,
          class_names=[str(label) for label in le_label.classes_], rounded=True, proportion=True)
# Save the plot to a larger image file (adjust the file format and resolution as needed)
plt.savefig('decision_tree_plot4.png', bbox_inches='tight', pad_inches=0.1)

# plt.show()
# Close the plot to prevent display issues
plt.close()
# plot_multiclass_roc_curve(dt_clf, X_test, y_test)
# classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Decision Tree',
#                                 'Accuracy': accuracy_score(y_test, y_pred).round(2),
#                                 'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
#                                 'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
#                                 'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
#                                 'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
#                                 'Cross Validation Mean Score': cv_scores_dt.mean().round(2)}, index=[0])], axis=0)

# Initialize the Neural Network Classifier
nn_clf = MLPClassifier(random_state=5805)
# Perform cross-validation for Neural Network
cv_scores_nn = cross_val_score(nn_clf, X, y, cv=5)
# Fit the Neural Network model on the training data
nn_clf.fit(X_train, y_train)
# Make predictions on the test data using Neural Network
y_pred_nn = nn_clf.predict(X_test)
# Evaluate the Neural Network model
accuracy_nn = accuracy_score(y_test, y_pred_nn)
print(f'Neural Network Accuracy: {accuracy_nn:.2f}')
# Display Neural Network classification report
print('Neural Network Classification Report:')
print(classification_report(y_test, y_pred_nn))
# classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Neural Network(MLP)',
#                                 'Accuracy': accuracy_score(y_test, y_pred).round(2),
#                                 'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
#                                 'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
#                                 'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
#                                 'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
#                                 'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)

# Initialize the Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=5805)
# Perform cross-validation for Random Forest
cv_scores_rf = cross_val_score(rf_clf, X, y, cv=5)
# Fit the Random Forest model on the training data
rf_clf.fit(X_train, y_train)
# Make predictions on the test data using Random Forest
y_pred_rf = rf_clf.predict(X_test)
# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf:.2f}')
# Display Random Forest classification report
print('Random Forest Classification Report:')
print(classification_report(y_test, y_pred_rf))
# classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Random Forest',
#                                 'Accuracy': accuracy_score(y_test, y_pred).round(2),
#                                 'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
#                                 'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
#                                 'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
#                                 'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
#                                 'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)

# Initialize the SVM Classifier
svm_clf = SVC(random_state=5805)
# Perform cross-validation for SVM
cv_scores_svm = cross_val_score(svm_clf, X, y, cv=5)
# Fit the SVM model on the training data
svm_clf.fit(X_train, y_train)
# Make predictions on the test data using SVM
y_pred_svm = svm_clf.predict(X_test)
# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'SVM Accuracy: {accuracy_svm:.2f}')
# Display SVM classification report
print('SVM Classification Report:')
print(classification_report(y_test, y_pred_svm))
# classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'SVM',
#                                 'Accuracy': accuracy_score(y_test, y_pred).round(2),
#                                 'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
#                                 'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
#                                 'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
#                                 'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
#                                 'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)

# Initialize the Logistic Regression Classifier
lr_clf = LogisticRegression(random_state=5805)
# Perform cross-validation for Logistic Regression
cv_scores_lr = cross_val_score(lr_clf, X, y, cv=5)
# Fit the Logistic Regression model on the training data
lr_clf.fit(X_train, y_train)
# Make predictions on the test data using Logistic Regression
y_pred_lr = lr_clf.predict(X_test)
# Evaluate the Logistic Regression model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Logistic Regression Accuracy: {accuracy_lr:.2f}')
# Display Logistic Regression classification report
print('Logistic Regression Classification Report:')
print(classification_report(y_test, y_pred_lr))
# classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Logistic Regression',
#                                 'Accuracy': accuracy_score(y_test, y_pred).round(2),
#                                 'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
#                                 'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
#                                 'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
#                                 'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
#                                 'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)

# Initialize the Naïve Bayes Classifier
nb_clf = GaussianNB()
# Perform cross-validation for Naïve Bayes
cv_scores_nb = cross_val_score(nb_clf, X, y, cv=5)
# Fit the Naïve Bayes model on the training data
nb_clf.fit(X_train, y_train)
# Make predictions on the test data using Naïve Bayes
y_pred_nb = nb_clf.predict(X_test)
# Evaluate the Naïve Bayes model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f'Naïve Bayes Accuracy: {accuracy_nb:.2f}')
# Display Naïve Bayes classification report
print('Naïve Bayes Classification Report:')
print(classification_report(y_test, y_pred_nb))
# classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Naive Bayes',
#                                 'Accuracy': accuracy_score(y_test, y_pred).round(2),
#                                 'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
#                                 'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
#                                 'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
#                                 'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
#                                 'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)

# Initialize the KNN Classifier
knn_clf = KNeighborsClassifier()
# Perform cross-validation for KNN
cv_scores_knn = cross_val_score(knn_clf, X, y, cv=5)
# Fit the KNN model on the training data
knn_clf.fit(X_train, y_train)
# Make predictions on the test data using KNN
y_pred_knn = knn_clf.predict(X_test)
# Evaluate the KNN model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'KNN Accuracy: {accuracy_knn:.2f}')
# Display KNN classification report
print('KNN Classification Report:')
print(classification_report(y_test, y_pred_knn))
# classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'KNN',
#                                 'Accuracy': accuracy_score(y_test, y_pred).round(2),
#                                 'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
#                                 'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
#                                 'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
#                                 'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
#                                 'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)


