# BDLPRnet

BDLPRnet is a Flask API for License Plate Recognition (LPR) that integrates with NVIDIA's TAO Toolkit for deployment. The project takes base64-encoded images as input, performs license plate detection, license plate recognition, and returns information about the detected license plates.

## Setup TAO Deploy

Before running the project, follow these setup instructions to deploy the TAO models:

1. Clone the TAO Deploy repository:

    ```bash
    git clone https://github.com/NVIDIA/tao_deploy.git
    cd tao_deploy
    ```

2. Follow the instructions in the [TAO Deploy README](https://github.com/NVIDIA/tao_deploy) for setting up the required dependencies and deploying the models.
3. Clone this repository:
    ```bash
    cd nvidia_tao_deploy
    git clone https://github.com/tanviran/BDLPRnet.git
    cd BDLPRnet/flask_api
    python3 app.py
    ```
  

## Important information 

### Input JSON Format

```json
{
    "imagedata": [
        { "car1": "<Image data in BASE64>" },
        { "car2": "<Image data in BASE64>" },
        { "car3": "<Image data in BASE64>" },
        { "car4": "<Image data in BASE64>" }
    ]
}
```
Here's the flow of this project:

![BDLPRnet ](https://github.com/tanviran/BDLPRnet/assets/97601593/c34cd5c0-eaff-4167-80f5-a4fb82736126)



### Here is an example of the expected output from the API:
```json
{
    {
        "image_name": "car1.jpg",
        "roi": ["230.733", "352.912", "323.963", "396.589"],
        "lpNum": "265996",
        "confidence": 0.999
    },
    {
        "image_name": "car2.jpg",
        "roi": ["261.487", "236.833", "368.392", "293.144"],
        "lpNum": "202864",
        "confidence": 0.998
    },
    {
        "image_name": "car3.jpg",
        "roi": ["274.906", "336.991", "379.444", "381.654"],
        "lpNum": "328301",
        "confidence": 0.998
    },
    {
        "image_name": "car4.jpg",
        "roi": ["261.189", "384.255", "370.871", "436.328"],
        "lpNum": "220080",
        "confidence": 0.998
    }
}
```
