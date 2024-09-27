 
#!/usr/bin/env python
#
t1 = True
gli_image = False
import flatbuffers
import numpy as np
import random
from seerep.fb import Image
from seerep.fb import image_service_grpc_fb as imageService
from seerep.util.common import get_gRPC_channel
from seerep.util.fb_helper import (
    createQuery,
    createTimeInterval,
    createTimeStamp,
    getProject,
)
from seerep.util.visualizations import display_instances
import json
# import print_results
from typing import Set, Tuple
import cv2
import matplotlib.pyplot as plt
builder = flatbuffers.Builder(1024)
# Default server is localhost !
channel = get_gRPC_channel("agrigaia-ur.ni.dfki:9090")

# 1. Get all projects from the server
projectuuid = getProject(builder, channel, 'Test_2023_06_22_2024-08-27-10-12-19_0.bag')
projectuuid = "2f551831-c853-4541-b5c6-7a626f302f32" # 22.06.
# projectuuidString = builder.CreateString(projectuuid)
# projectUuids = [projectuuidString]
# # 2. Check if the defined project exist; if not exit
if not projectuuid:
    print("project doesn't exist!")
    exit()
# 3. Get gRPC service object
stub = imageService.ImageServiceStub(channel)
# # Create all necessary objects for the query
# l = 10
# # polygon_vertices = []
# # polygon_vertices.append(createPoint2d(builder, -1.0 * l, -1.0 * l))
# # polygon_vertices.append(createPoint2d(builder, -1.0 * l, l))
# # polygon_vertices.append(createPoint2d(builder, l, l))
# # polygon_vertices.append(createPoint2d(builder, l, -1.0 * l))
# # polygon2d = createPolygon2D(builder, 7, -1, polygon_vertices)
if t1:
    timeMin = createTimeStamp(builder, 1685610870, 0)
    timeMax = createTimeStamp(builder, 1685610880, 0)
timeInterval = createTimeInterval(builder, timeMin, timeMax)

projectUuids = [projectuuid]
# category = ["image_type"]
# category = ["image_type", "crops"]
# if gli_image:
#     imageType = [builder.CreateString("image_type_gli")]
# else:
#     imageType = [builder.CreateString("image_type_rgb")]
# crops = [builder.CreateString("white_cabbage_young")]
# # labels = [imageType]
# labels = [imageType, crops]
# labelCategory = createLabelWithCategory(builder, category, labels)
dataUuids = ["a5de2888-a3e1-4ad4-9c3d-f0841accc6b8"]
# instanceUuids = [builder.CreateString("3e12e18d-2d53-40bc-a8af-c5cca3c3b248")]
# 4. Create a query with parameters
# all parameters are optional
# with all parameters set (especially with the data and instance uuids set) the result of the query will be empty. Set the query parameters to adequate values or remove them from the query creation
time_and_detection = []
query = createQuery(
    builder,
    # boundingBox=boundingboxStamped,
    # timeInterval=timeInterval,
    # labels=labelCategory,
    # mustHaveAllLabels=True,
    projectUuids=projectUuids,
    # instanceUuids=instanceUuids,
    # dataUuids=dataUuids,
    # withoutData=False,
    # fullyEncapsulated=False,
)
builder.Finish(query)
buf = builder.Output()
# Datumaro
# basic structure of the datumaro json

# 5. Query the server for images matching the query and iterate over them
counter = 0
for responseBuf in stub.GetImage(bytes(buf)):
    datumaro_dict: dict = {
        "info": {},
        "categories": {
            "label": {
                "labels": [],
                "attributes": [],
            },
            "points": {"items": []},
        },
        "items": [],
    }
    labels: Set[Tuple[str, int]] = set()
    counter = counter + 1
    response = Image.Image.GetRootAs(responseBuf)
    print(f"uuidmsg: {response.Header().UuidMsgs().decode('utf-8')}")
    print(f"timestamp: {response.Header().Stamp().Seconds()}")
    time_and_detection.append([response.Header().Stamp().Seconds(), response.Header().Stamp().Nanos(),bool(random.getrandbits(1)),bool(random.getrandbits(1))]) #sec, nano, male_detected, child_detected
    # loop over each category
    for i in range(response.LabelsLength()):
        category_with_labels = response.Labels(i)
        item = json.loads(category_with_labels.DatumaroJson().decode())
        datumaro_dict["items"].append(item)
        for j in range(category_with_labels.LabelsLength()):
            labels.add(
                (
                    category_with_labels.Labels(j).Label().decode(),
                    category_with_labels.Labels(j).LabelIdDatumaro(),
                )
            )

    labels = sorted(labels, key=lambda x: x[1])
    # sort by the label_id to get the correct order
    datumaro_dict["categories"]["label"]["labels"] = [
        {"name": label, "parent": "", "attributes": []} for label, _ in labels
    ]
    change_pos = [(elem, idx) for idx, elem in enumerate([pair[1] for pair in labels])]
    bboxes = []
    for item in datumaro_dict["items"]:
        for annotation in item["annotations"]:
            bboxes.append(annotation["bbox"])
    # change the labels_ids to the new correct ones
    for old_pos, new_pos in change_pos:
        if old_pos != new_pos:
            for item in datumaro_dict["items"]:
                for annotation in item["annotations"]:
                    if annotation["label_id"] == old_pos:
                        annotation["label_id"] = new_pos
                    print(item["annotations"])
    # extract image
    if not response.DataIsNone():
        image = np.ascontiguousarray(response.DataAsNumpy()).astype(np.uint8)
        offset = 0  # image.min()
        scale = 1.0  # 255.0/image.max()
        if response.Encoding().decode('utf-8') == "mono8":
            image.resize(response.Height(), response.Width(), 1)
            image = (image - offset) * scale
        elif response.Encoding().decode('utf-8') == "rgb8":
            image.resize(response.Height(), response.Width(), 3)
        elif response.Encoding().decode('utf-8') == "bgr8":
            image.resize(response.Height(), response.Width(), 3)
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        for box in bboxes:
            start_point = (round(box[0]-box[2]/2),round(box[1]-box[3]/2))
            end_point = (round(box[0]+box[2]/2),round(box[1]+box[3]/2))
            cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.imshow("Image", image)
        cv2.waitKey(1)
 