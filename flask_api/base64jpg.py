import base64
from PIL import Image
import os
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

def convert_image(counter, image_entry, output_directory):
    for key, image_data in image_entry.items():
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data)

        # Open the image using Pillow
        image = Image.open(BytesIO(image_bytes))

        # Save the image with a sequential name
        image_path = os.path.join(output_directory, f'car{counter}.jpg')
        # print(image_path)
        image.save(image_path)

def convert_json_to_images_parallel(data, output_directory="images"):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o777)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        # Submit tasks for each image entry with a counter
        futures = [
            executor.submit(convert_image, i + 1, image_entry, output_directory)
            for i, image_entry in enumerate(data['imagedata'])
        ]

        # Wait for all tasks to complete
        for future in futures:
            future.result()

    return output_directory

