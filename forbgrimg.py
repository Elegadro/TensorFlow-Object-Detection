import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import cv2
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import warnings
warnings.filterwarnings("ignore")

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

try:
    IMAGE_PATHS = "PATH_TO_BGR_IMAGE"
except:
    print("Resminizin konumunu buraya girin")

PATH_TO_MODEL_DIR = "./content/workspace/training_demo/exported-models/my_model"

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
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

input_tensor = tf.convert_to_tensor(image)
input_tensor = input_tensor[tf.newaxis, ...]
detections = detect_fn(input_tensor)

num_detections = int(detections.pop("num_detections"))
detections = {key: value[0, :num_detections ].numpy()
            for key, value in detections.items()}
detections["num_detections"] = num_detections

detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

image_with_detections = image.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
    image_with_detections,
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
cv2.imshow("Output", image_with_detections)
cv2.waitKey(0)