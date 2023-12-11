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

## Usage

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
