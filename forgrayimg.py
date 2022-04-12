import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import cv2
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

try:
    IMAGE_PATHS = "PATH_TO_GRAY_IMAGE"
except:
    print("Gray scale edilmiş resmin konumunu girin")

try:
    ORIGINAL_IMG_PATH = "PATH_TO_BGR_IMAGE"
except:
    print("Orijinal resminizin konumunu girin")

PATH_TO_LABELS = "./content/workspace/training_demo/annotations/label_map.pbtxt"

PATH_TO_SAVED_MODEL = "./content/workspace/training_demo/exported-models/my_model/saved_model"

print("Loading model...", end="")
start_time = time.time()
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Done! Took {elapsed_time} seconds")

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

print(f"Running inference for {IMAGE_PATHS}...", end="")

image = cv2.imread(IMAGE_PATHS)
image_expanded = np.expand_dims(image, axis=0)

input_tensor = tf.convert_to_tensor(image)
input_tensor = input_tensor[tf.newaxis, ...]
detections = detect_fn(input_tensor)

num_detections = int(detections.pop("num_detections"))
detections = {key: value[0, :num_detections ].numpy()
            for key, value in detections.items()}
detections["num_detections"] = num_detections

detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

original_img = cv2.imread(ORIGINAL_IMG_PATH)

viz_utils.visualize_boxes_and_labels_on_image_array(
    original_img,
    detections['detection_boxes'],
    detections['detection_classes'],
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=0.5,
    agnostic_mode=False
)

print("Done")
cv2.imshow("Output", original_img)
cv2.waitKey(0)