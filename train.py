import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Dataset paths
train_path = "data/NEU-DET/train/images"

# Lists to store data
images = []
labels = []

# Image size
IMG_SIZE = 128

print("Loading images...\n")

for label in os.listdir(train_path):

    class_path = os.path.join(train_path, label)

    if not os.path.isdir(class_path):
        continue

    for image_name in os.listdir(class_path):

        image_path = os.path.join(class_path, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        images.append(image)

        labels.append(label)

print(f"Total Images Loaded : {len(images)}")
print(f"Total Labels Loaded : {len(labels)}")

print("\nClasses Found:")
print(set(labels))

print("\nExtracting HOG Features...")

features = []

for image in images:

    hog_features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    features.append(hog_features)

features = np.array(features)

print("Feature Matrix Shape:", features.shape)
print("Labels:", len(labels))
print("\nEncoding Labels...")

encoder = LabelEncoder()

y = encoder.fit_transform(labels)

X = features

print("Classes:")
print(encoder.classes_)
print("\nLoading Validation Images...\n")

val_path = "data/NEU-DET/validation/images"

val_images = []
val_labels = []

for label in os.listdir(val_path):

    class_path = os.path.join(val_path, label)

    if not os.path.isdir(class_path):
        continue

    for image_name in os.listdir(class_path):

        image_path = os.path.join(class_path, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        val_images.append(image)

        val_labels.append(label)

print(f"Validation Images Loaded: {len(val_images)}")
print("\nExtracting Validation HOG Features...\n")

val_features = []

for image in val_images:

    hog_features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm="L2-Hys"
    )

    val_features.append(hog_features)

val_features = np.array(val_features)

print("Validation Feature Shape:", val_features.shape)
val_labels = encoder.transform(val_labels)

print("Validation Labels:", len(val_labels))


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="linear"),
    "XGBoost": xgb.XGBClassifier(
        eval_metric="mlogloss",
        random_state=42
    )
}

print("\nModel Comparison\n")

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X, y)

    predictions = model.predict(val_features)

    accuracy = accuracy_score(val_labels, predictions)

    print(f"{name}: {accuracy:.4f}\n")
    
best_model = models["SVM"]

joblib.dump(best_model, "models/svm_model.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")

print("\nModel Saved Successfully!")    


# Use the trained SVM model
svm_model = models["SVM"]

predictions = svm_model.predict(val_features)

cm = confusion_matrix(val_labels, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap="Blues")
plt.title("Confusion Matrix - Support Vector Machine")
plt.tight_layout()

plt.savefig("screenshots/confusion_matrix.png")

plt.show()


results = {}

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X, y)

    predictions = model.predict(val_features)

    accuracy = accuracy_score(val_labels, predictions)

    results[name] = accuracy

    print(f"{name}: {accuracy:.4f}")


plt.figure(figsize=(8,5))

plt.bar(results.keys(), results.values())

plt.ylabel("Accuracy")

plt.title("Traditional ML Model Comparison")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig("screenshots/model_comparison.png")

plt.show()    
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.bar(results.keys(), results.values())

plt.ylabel("Validation Accuracy")

plt.xlabel("Machine Learning Models")

plt.title("Traditional Machine Learning Model Comparison")

for i, v in enumerate(results.values()):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center")

plt.tight_layout()

plt.savefig("screenshots/model_comparison.png")

plt.show()