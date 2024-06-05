import torch
import torchvision
import base64
import torch.nn.functional as F
from flask import Flask, request, jsonify
from pathlib import Path
from torch import nn
from torchvision import transforms

BASE_PATH = str(Path(__file__).parent)
PATH_OF_MODEL = BASE_PATH + "/cnn_net.pt"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
app = Flask(__name__)

def load_the_model() -> nn.Module:
    """
    Load the pre-trained CNN model.
    
    Returns:
        nn.Module: The loaded CNN model.
    """
    # The architecture of our model
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 39)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    loaded_model = Net().to(device) # loads and send the model to the device
    loaded_model.load_state_dict(torch.load(PATH_OF_MODEL, map_location=device))  # loads the trained parameters
    loaded_model.eval()
    return loaded_model

def loads_data_classes() -> list:
    """
    Load the class labels for the prediction.

    Returns:
        list: A list of class labels.
    """
    class_labels = ['ا','ب','ت','ث','ج','ح','خ','د','ذ','ر','ز','س','ش','ص','ض','ط','ظ','ع','غ','ف',
                    'ق','ك','ل','لا','م','ن','ه','و','ي','٠','١','٢','٣','٤','٥','٦','٧','٨','٩']
    return class_labels

def base64_to_image(base64_string: str) -> Path:
    """
    Convert a base64 string to an image and save it locally.

    Args:
        base64_file (str): Base64 encoded string of the image.

    Returns:
        str: The path to the saved image.
    """
    # Decode the Base64 string to binary data
    image_data = base64.b64decode(base64_string)
    # Deefine the path of the image
    image_path = BASE_PATH + '/decoded.png'
    # write the image data
    with open(image_path, 'wb') as output_file:
        output_file.write(image_data)
    return image_path

def predict_on_base64(model: nn.Module, base64_string: str) -> tuple:
    """
    Predict the class label of an image given its base64 encoded string.

    Args:
        model (nn.Module): our pre-trained model.
        base64_file (str): Base64 encoded string of the image.

    Returns:
        tuple: The predicted class label and its probability.
    """
    # Getting the path of our png image to pass it to the model
    path = base64_to_image(base64_string)
    # Handling the image and convert it to float32 tensor
    custom_image = torchvision.io.read_image(str(path)).type(torch.float32)
    # Normalizing the image
    custom_image = custom_image / 255. 
    # Transforms the image to be grayscale and 32x32 (the model input)
    transform_img = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
    ])
    # Applying the transforms to the image
    custom_image_transformed = transform_img(custom_image)
    # Adding the batch size dimension to the image [1, 1, 32, 32] -> shape of the image now
    custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
    # starting the torch inference modee
    with torch.inference_mode():
        # getting the logits
        custom_image_pred = model(custom_image_transformed_with_batch_size.to(device)) 
        # converting thee logits to probability
        prob = torch.softmax(custom_image_pred, dim=1) 
        # getting the sample probability and round the number
        sample_prob = round(prob[0][prob.argmax()].item(), 3) 
        # getting the index of the label
        test_pred_labels = custom_image_pred.argmax(dim=1).item() 
    # gets the data labels
    labels = loads_data_classes() 
    # gets the label name
    test_pred_labels = labels[test_pred_labels] 
    
    return test_pred_labels, sample_prob

# Loads the model to prevent loading it every request
model = load_the_model()

@app.route('/predict', methods=['POST'])
def predict():
        ## if we will receive the data in a json format
        # Getting the request header
        #data = request.get_json()
        # Getting the json bas64 from body
        #base64_string = data.get("base64_string")

        ## if we will receive the data in a plain text format
        # Extract raw data from the body and convert it from bytes to string (HTTP request bodies are often sent as byte streams)
        base64_string = request.data.decode('utf-8') 

        # checking if there is not a base64_string in body
        if not base64_string:
            return jsonify({"error": "No base64 string provided"}), 400
        
        # Getting our prediction and probability
        prediction, probability = predict_on_base64(model=model, base64_string=base64_string)
        return jsonify({"Prediction": prediction, "Probability": probability})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
