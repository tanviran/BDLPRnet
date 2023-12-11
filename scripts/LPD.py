from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pycuda.driver as cuda
import cv2
import argparse
import logging
import os
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from subprocess import run
import json 
import shutil
from nvidia_tao_deploy.cv.BDLPRnet.scripts.lpdcodes.dataloader import YOLOv3KITTILoader, aug_letterbox_resize
from nvidia_tao_deploy.cv.BDLPRnet.scripts.lpdcodes.inferencer import YOLOv3Inferencer
from nvidia_tao_deploy.cv.BDLPRnet.scripts.LPR import LPR

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)


def main(image_dir):
    cuda.init()
    cuda.Device(0).make_context()
    """YOLOv4 TRT inference."""

    model_path = '/workspace/tao-deploy/nvidia_tao_deploy/cv/BDLPRnet/model/lpd/yolov4_tiny_usa_deployable.etlt_b1_gpu0_fp32.engine'
    #model_path = './yolov4_tiny_usa_deployable.etlt_b1_gpu0_fp32.engine'
    results_dir = './results/'
    threshold = 0.3
    batch_size = 1
    roi = []
    conf = []
    lp_value = []
    image_name = []
    trt_infer = YOLOv3Inferencer(model_path, batch_size=batch_size)
    c, h, w = trt_infer._input_shape
    
    conf_thres = 0.001
    img_mean = {}

    if c == 3:
        img_mean = img_mean.get('b', 103.939), img_mean.get('g', 116.779), img_mean.get('r', 123.68)
    else:
        img_mean = [img_mean['l']] if img_mean else [117.3786]

    # Override path if provided through command line args
    image_dirs = [image_dir] 
    mapping_dict = {'LP': 'LP'}
 
    image_depth = 8 
    

    dl = YOLOv3KITTILoader(
        shape=(c, h, w),
        image_dirs=image_dirs,
        label_dirs=[None],
        mapping_dict=mapping_dict,
        exclude_difficult=True,
        batch_size=batch_size,
        is_inference=True,
        image_mean=img_mean,
        image_depth=image_depth,
        dtype=trt_infer.inputs[0].host.dtype)

    inv_classes = {v: k for k, v in dl.classes.items()}

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        os.chmod(results_dir, 0o777)

    output_annotate_root = os.path.join(results_dir, "images_annotated")
    output_label_root = os.path.join(results_dir, "labels")
    output_cropped_root = os.path.join(results_dir, "cropped_images")

    # Set permissions for the subdirectories and their contents to drwxrwxr-x (775)
    for dir_path in [output_annotate_root, output_label_root, output_cropped_root]:
        os.makedirs(dir_path, exist_ok=True)
        os.chmod(dir_path, 0o777)


    for batch_idx, (imgs, _) in tqdm(enumerate(dl), total=len(dl), desc="Producing predictions"):
        y_pred = trt_infer.infer(imgs)
        image_paths = dl.image_paths[np.arange(batch_size) + batch_size * batch_idx]

        for image_idx in range(len(y_pred)):
            y_pred_valid = y_pred[image_idx][y_pred[image_idx][:, 1] > conf_thres]

            target_size = np.array([w, h, w, h])

            # Scale back bounding box coordinates
            y_pred_valid[:, 2:6] *= target_size[None, :]

            # Load image
            img = Image.open(image_paths[image_idx])
            
            # Handle grayscale images
            img = img.convert('L') if c == 1 and image_depth == 8 else img
            img = img.convert('I') if c == 1 and image_depth == 16 else img
        
            orig_width, orig_height = img.size
            img, _, crop_coord = aug_letterbox_resize(img,
                                                      y_pred_valid[:, 2:6],
                                                      num_channels=c,
                                                      resize_shape=(trt_infer.width, trt_infer.height))
            img = Image.fromarray(img.astype('uint8'))
            
            

            # Store images
            bbox_img, label_strings = trt_infer.draw_bbox(img.copy(), y_pred_valid, inv_classes, threshold)
            bbox_img = bbox_img.crop((crop_coord[0], crop_coord[1], crop_coord[2], crop_coord[3]))
            bbox_img = bbox_img.resize((orig_width, orig_height))

            img_filename = os.path.basename(image_paths[image_idx])

            bbox_img.save(os.path.join(output_annotate_root, img_filename))

            # Store labels
            filename, _ = os.path.splitext(img_filename)
            label_file_name = os.path.join(output_label_root, filename + ".txt")

            with open(label_file_name, "w", encoding="utf-8") as f:
                for l_s in label_strings:
                    f.write(l_s)
            
            image_name.append(img_filename)
            label_string = label_strings[0]
            # Split the string into tokens
            tokens = label_string.split()
            # Extract the ROI values
            roi_values = tokens[4:8]
            confidence = float(tokens[-1])
        
            roi.append(roi_values)
            conf.append(confidence)

            # Save the cropped image
            for label_str in label_strings:
                tokens = label_str.split()
                if len(tokens) >= 6 and tokens[0] == 'lp':
                    confidence = float(tokens[-1])
                    if confidence >= threshold:
                        bbox = list(map(float, tokens[4:8]))  # Extracting relevant part of label
            
                        # Crop image using bbox values
                        left, top, right, bottom = bbox
                        cropped_image = img.crop((left, top, right, bottom))
                        
                        cropped_filename = os.path.join(output_cropped_root, f"{img_filename}")
                        cropped_image.save(cropped_filename)
                        

    logging.info("Finished inference.")
    _, lp_values = LPR(output_cropped_root)
   
    # Create a list of dictionaries for each image
    result_list = []
    for i in range(len(image_name)):
        i_name = image_name[i]
        r = roi[i]
        confidence = conf[i]
        lpNum = lp_values[i]

        # Create a dictionary for the current image
        image_dict = {
            "image_name": i_name,
            "roi": r,
            "lpNum": lpNum,
            "confidence": confidence
        }

        # Add the image dictionary to the result list
        result_list.append(image_dict)

    # Convert the result list to JSON
    json_output = json.dumps(result_list, indent=4)

    # Print the JSON output
    # print(json_output)
    
    shutil.rmtree(results_dir)
    print("\n\n\nLPR done\n\n\n")
    cuda.Context.pop()

    return json_output
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='infer', description='Inference with a YOLOv4 TRT model.')
    parser.add_argument(
        'image_dir',
        type=str,
        help='Input directory of images')
    args = parser.parse_args()

    main(args.image_dir)
