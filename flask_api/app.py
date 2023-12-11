# app.py
import shutil
from flask import Flask, request, jsonify
from nvidia_tao_deploy.cv.BDLPRnet.flask_api.base64jpg import convert_json_to_images_parallel
from nvidia_tao_deploy.cv.BDLPRnet.scripts.LPD import main


app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'
   
@app.route('/BDLPRnet', methods=['POST'])
def BDLPRnet():
    try:
        data = request.get_json()
        output_path = convert_json_to_images_parallel(data)
        result = main(image_dir=data.get('image_dir', './images/'))
        shutil.rmtree(output_path)
        return result
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return a 500 Internal Server Error with the error message

if __name__ == "__main__":
    app.run(port=8000, host = '0.0.0.0', debug = True)

