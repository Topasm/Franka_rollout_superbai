import pyrealsense2 as rs
import numpy as np
import cv2
import requests
import json_numpy
json_numpy.patch()


pipeline = rs.pipeline()
config = rs.config()

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert color frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Visualize the color image (optional)
        # cv2.imshow("color frame", cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        # key = cv2.waitKey(1)
        # if key == 27:  # Press 'Esc' to exit
        #     break

        action = requests.post(
            "http://0.0.0.0:8000/act",
            json={"image": color_image, "instruction": "do something", "unnorm_key": "austin_buds_dataset_converted_externally_to_rlds"}
        ).json()

        print("Action:", action)


finally:
    pipeline.stop()
    # cv2.destroyAllWindows()