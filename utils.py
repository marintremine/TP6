import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
#import requests
from IPython.core.display import display, HTML

def load_data(train_file = 'data/adult_train.csv', test_file = 'data/adult_test.csv'):
    train_data = pd.read_csv(train_file, sep=',', na_values='?', engine='python')
    test_data = pd.read_csv(test_file, sep=',', na_values='?', engine='python')
    return train_data, test_data

def load_data2(train_file = 'data/adult_train.csv', test_file = 'data/adult_test_no_labels.csv'):
    train_data = pd.read_csv(train_file, sep=',', na_values='?', engine='python')
    test_data = pd.read_csv(test_file, sep=',', na_values='?', engine='python')
    return train_data, test_data

def encode_attributs_category(X_train, y_train, X_test, y_test):
    # Encode categorical variables and labels
    encoder = LabelEncoder()
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = encoder.fit_transform(X_train[col])
        X_test[col] = encoder.transform(X_test[col])

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)  # 2 classes classification: '>50K' is 1, '<=50K' is 0
    y_test = label_encoder.fit_transform(y_test)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    return X_train, y_train, X_test, y_test

def convert_to_tensor(X_train, y_train, X_test, y_test):
    # # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = F.one_hot(torch.tensor(y_train, dtype=torch.long), num_classes=2).float()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = F.one_hot(torch.tensor(y_test, dtype=torch.long), num_classes=2).float()

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def pre_processing(train_data, test_data):
    # Drop rows with missing values
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)
    X_test = test_data.drop('income', axis=1)
    y_test = test_data['income']

    # Separate features (X) and target (y)
    X_train = train_data.drop('income', axis=1)
    y_train = train_data['income']
    return X_train, y_train, X_test, y_test

def get_result(name, pred_file = 'predictions.csv'):
    # Define the URL and the files you want to send
    url = "http://webvinc.iuto.ovh/evaluate"
    files = {'file': open(pred_file, 'rb')}
    data = {'name': name}
    # Make the POST request
    response = requests.post(url, files=files, data=data)
    # Print the response
    result = response.json()

    try:
        print(f"Accuracy = {result['evaluation']['accuracy']}")
        print("------------------DETAILS-------------------")
        print(f"{'Class':<6} {'F1 Score':<10} {'Precision':<10} {'Recall':<10} {'Support':<10}")
        # Extract and print each class's metrics
        for key, metrics in result['evaluation'].items():
            if key != 'accuracy':  # Exclude the accuracy key
                print(f"{key:<6} {metrics['f1-score']:<10.3f} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['support']:<10.1f}")
    except:
        print (result)
    return response

def show_ranking():
    # Step 1: Call the API
    url = "http://webvinc.iuto.ovh/ranking"  # Replace with your API endpoint
    response = requests.get(url)  # Use requests.post() if it's a POST request
    # Step 2: Check if the request was successful
    if response.status_code == 200:
        # Step 3: Display the HTML content
        display(HTML(response.text))
    else:
        print(f"Error: {response.status_code}, {response.text}")

import numpy as np
def predire (model, X_test_tensor, device):
    # Set the model to evaluation mode
    model.eval()
    y_pred_list = []
    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Move the test tensors to the specified device
        X_test_tensor = X_test_tensor.to(device)
        # Predict outputs for the entire test set
        y_test_pred = model(X_test_tensor)
        # Apply argmax to get the predicted class (returns index of the highest score)
        y_pred_tag = torch.argmax(y_test_pred, dim=1).cpu().numpy()
        # Append the predictions to the list
        y_pred_list.append(y_pred_tag)

    y_pred_list = np.concatenate(y_pred_list).flatten()