import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
data_path = './dataset.pickle'
data_dict = pickle.load(open(data_path, 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the Random Forest classifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
