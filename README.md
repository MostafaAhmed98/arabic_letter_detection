# Image Classification API

This API allows you to classify images based on a pre-trained CNN model. You can send images encoded as base64 strings and receive predictions along with probabilities.

## Prerequisites

Make sure you have Docker installed on your system. You can download and install Docker from [here](https://www.docker.com/get-started).

## Files

- `app.py`: The main Flask application file.
- `requirements.txt`: Lists the required Python packages.
- `cnn_net.pt`: The pre-trained model file.
- `Dockerfile`: The Docker configuration file.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-repo/image-classification-api.git
cd image-classification-api
```


### 2. Build the Docker image

```bash
docker build -t image-classification-api .
```

### 3. Run the Docker container

```bash
docker container run -d -p 5000:5000 image-classification-api
```

The API will be available at `http://127.0.0.1:5000`.

## API Usage

### Endpoint

#### `POST /predict`

This endpoint accepts a base64 encoded string of an image and returns the predicted class label along with the probability as a json file.

### Request Headers

- `Content-Type: application/json` if sending JSON
- `Content-Type: text/plain` if sending plain text

### Request Body

#### JSON Format

```json
{
  "base64_string": "<your_base64_encoded_image>"
}
```

#### Plain Text Format

```text
<your_base64_encoded_image>
```

### Example using Postman

1. Open Postman and create a new POST request.
2. Set the URL to `http://127.0.0.1:5000/predict`.
3. Set the `Content-Type` header to `text/plain` or `application/json` based on your format.
4. For plain text, paste your base64 encoded image directly in the body.
5. For JSON, use the following format in the body:
   ```json
   {
     "base64_string": "<your_base64_encoded_image>"
   }
   ```
6. Send the request.

### Example Response

```json
{
  "Prediction": "label_name",
  "Probability": 0.987
}
```
![image](https://github.com/MostafaAhmed98/arabic_letter_detection/assets/90983988/968316b5-1e07-49b1-9eb1-61dad2700c5a)



## Troubleshooting

- Make sure the Docker container is running without errors.
- Verify the base64 string is correctly formatted and represents a valid image.
- Ensure the correct headers are set based on your request format.

