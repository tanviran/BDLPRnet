from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import logging
import os
import numpy as np

from nvidia_tao_deploy.cv.BDLPRnet.scripts.lprcodes.inferencer import LPRNetInferencer
from nvidia_tao_deploy.cv.BDLPRnet.scripts.lprcodes.dataloader import LPRNetLoader
from nvidia_tao_deploy.cv.BDLPRnet.scripts.lprcodes.utils import decode_ctc_conf


logging.getLogger('PIL').setLevel(logging.WARNING)
logging.basicConfig(format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
                    level="INFO")
logger = logging.getLogger(__name__)

def LPR(lpd_dir):
    """LPRNet TRT inference."""
    # Updated paths    
    lpr_model_path = '/workspace/tao-deploy/nvidia_tao_deploy/cv/BDLPRnet/model/lpr/Latest2022-epoch160-fp16_TRT.engine'
    characters_list_file = '/workspace/tao-deploy/nvidia_tao_deploy/cv/BDLPRnet/scripts/lprcodes/specs/dict_us.txt'
    batch_size = 1
    data_format = "channels_first"
    max_label_length = 8 
    image_dirs = [lpd_dir]
    lp_values = []
    img_names =[]
    trt_infer = LPRNetInferencer(lpr_model_path, data_format=data_format, batch_size=batch_size)

    if not os.path.exists(characters_list_file):
        raise FileNotFoundError(f"{characters_list_file} does not exist!")

    with open(characters_list_file, "r", encoding="utf-8") as f:
        temp_list = f.readlines()
    classes = [i.strip() for i in temp_list]
    blank_id = len(classes)

    dl = LPRNetLoader(
        trt_infer._input_shape,
        image_dirs,
        [None],
        classes=classes,
        is_inference=True,
        batch_size=batch_size,
        max_label_length=max_label_length,
        dtype=trt_infer.inputs[0].host.dtype)

    for i, (imgs, _) in enumerate(dl):
        y_pred = trt_infer.infer(imgs)
       
        # decode prediction
        decoded_lp, _ = decode_ctc_conf(y_pred,
                                        classes=classes,
                                        blank_id=blank_id)
        image_paths = dl.image_paths[np.arange(batch_size) + batch_size * i]
       
        for image_path, lp in zip(image_paths, decoded_lp):
            image_name = os.path.basename(image_path)
            img_names.append(image_name)
            lp_values.append(lp)

    
    logging.info("Finished LPR inference.")
    return img_names ,lp_values

