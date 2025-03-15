import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the data dictionary
try:
    with open('./data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print("Error: data.pickle not found. Run create_dataset.py first.")
    exit()

# Check for consistent data length
max_length = max(len(item) for item in data_dict['data'])
padded_data = [np.pad(item, (0, max_length - len(item)), 'constant') for item in data_dict['data']]

# Convert to NumPy array
data = np.asarray(padded_data)
print(f"Final data shape: {data.shape}")

# Encode labels to numerical format
labels = np.asarray(data_dict['labels'])
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Ensure label range is correct
unique_labels = set(labels)
print("Unique labels in dataset:", unique_labels)
if max(unique_labels) >= 27:
    print("Warning: Labels should be between 0 and 26.")
    exit()

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the classifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and evaluate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f"{score * 100:.2f}% of samples were classified correctly!")

# Save the trained model and label encoder
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': label_encoder}, f)

print("Model and label encoder saved successfully!")