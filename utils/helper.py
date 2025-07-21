import numpy as np
import cv2
import imutils

color1 = (0, 0, 255)  # red
text_color = (255, 255, 255)
color3 = (255, 0, 0)

def cxcy2xyxy(box):
    xtl = int(box[0] - (box[2]/2))
    ytl = int(box[1] - (box[3]/2))
    xbr = int(box[0] + (box[2]/2))
    ybr = int(box[1] + (box[3]/2))

    return [xtl, ytl, xbr, ybr]

def scale_boxes(box, normalized=False):
        """
        box: Bounding box generated for the image size (e.g. 512 x 512) expected by the model at triton server
        return: Scaled bounding box according to the input image from the ros topic.
        """
        if normalized:
            # TODO make it dynamic with mc.Modelshape according to CHW or HWC
            xtl, xbr = box[0] * image_dim[1], box[2] * image_dim[1]
            ytl, ybr = box[1] * image_dim[0], box[3] * image_dim[0]
        else:
            xtl, xbr = box[0] * (image_dim[1] / self.input_size[0]), box[2] * (
                image_dim[1] / self.input_size[0]
            )
            ytl, ybr = (
                box[1] * image_dim[0] / self.input_size[1],
                box[3] * image_dim[0] / self.input_size[1],
            )

        return [xtl, ytl, xbr, ybr]
    
def scale_box_array(box, 
                    model_input_dim=(512, 512), 
                    image_dim=(640, 480),
                    padded=False):
        """
        box: Bounding box generated for the image size (e.g. 512 x 512) expected by the model at triton server
        return: Scaled bounding box according to the input image from the ros topic.
        """
        # if normalized:
        #     # TODO make it dynamic with mc.Modelshape according to CHW or HWC
        #     xtl, xbr = box[0] * image_dim[1], box[2] * image_dim[1]
        #     ytl, ybr = box[1] * image_dim[0], box[3] * image_dim[0]
        if padded:
            xtl, xbr = box[:, 0] * (image_dim[1] / model_input_dim[1]), box[:, 2] * (
                image_dim[1] / model_input_dim[1]
            )
            ytl, ybr = (
                box[:, 1] * image_dim[0] / model_input_dim[0],
                box[:, 3] * image_dim[0] / model_input_dim[0],
            )
        else:
            xtl, xbr = box[:, 0] * (image_dim[1] / self.input_size[1]), box[
                :, 2
            ] * (image_dim[1] / self.input_size[1])
            ytl, ybr = (
                box[:, 1] * image_dim[0] / self.input_size[0],
                box[:, 3] * image_dim[0] / self.input_size[0],
            )
        xtl = np.reshape(xtl, (len(xtl), 1))
        xbr = np.reshape(xbr, (len(xbr), 1))

        ytl = np.reshape(ytl, (len(ytl), 1))
        ybr = np.reshape(ybr, (len(ybr), 1))
        return np.concatenate((xtl, ytl, xbr, ybr, box[:, 4:6]), axis=1)
    
def resize(image, input_metadata: list[dict]):
    """_summary_

    Args:
        image (_type_): _description_
        input_metadata (list[dict]): _description_

    Returns:
        _type_: _description_
    """
    # cv_image = cv2.resize(cv_image, (self.channel.input.shape[2], self.channel.input.shape[1]))
    tmp = image.copy()
    s_h, s_w = image.shape[0], image.shape[1]
    n_h, n_w = input_metadata[0]['shape'][1], input_metadata[0]['shape'][2]
    if s_w > s_h:
        # same aspect ratio
        padded_image = np.zeros((n_h, n_w, 3), dtype=np.uint8)
        cv_image = imutils.resize(tmp, width=n_w)
        # different aspect ratio
        if cv_image.shape[0] > n_h:
            cv_image = imutils.resize(tmp, height=n_h)
        if cv_image.shape[1] > n_w:
            cv_image = imutils.resize(tmp, width=n_w)
    else:
        padded_image = np.zeros((n_h, n_w, 3), dtype=np.uint8)
        cv_image = imutils.resize(tmp, height=n_h)
    # padded image
    padded_image[0 : cv_image.shape[0], 0 : cv_image.shape[1]] = cv_image
    # named_window = 'resized'
    # cv2.imshow(named_window, padded_image)
    # cv2.waitKey()
    # cv2.destroyWindow(named_window)
    return padded_image, cv_image.shape[0], cv_image.shape[1]
    
def visualize(sample: dict, 
            cv_window_name: str, 
            class_names: list, 
            new_model_key: False,
            model_name=None,
            ):
    """
    This function visualizes the image sample with the bounding boxes and class labels.
    The bounding boxes are fetched from the sample dictionary. The sample dictionary can have multiple annotations.
    Index 0 is the ground truth and index -1 is the predictions from the model execution. If more than two indices exist,
    they are from previous runs / models which can be fetched by model_name parameter.
    sample: dict: The sample dictionary containing the image and annotations in datumaro format.
    cv_window_name: str: The name of the window to display the image.
    class_names: list: The list of class names.
    new_model_key: bool: The flag to indicate if the model predictions are new or not compared to SEEREP version. 
    model_name: str: The name of the model to fetch the annotations from the sample dictionary based on Triton model name stored in SEEREP. 
    """
    if new_model_key:
        model_index = -1
    elif new_model_key == False and model_name != None:
        model_index = sample['annotations']['categories']['label']['labels'].index(model_name)
    else:
        model_index = 0
    for ann_idx, ann in enumerate(sample['annotations']['items'][model_index]['annotations']):
        bbox = ann['bbox']
        bbox = cxcy2xyxy(bbox)
        label = int(ann['id'])
        cv2.rectangle(sample['image'], 
                        (bbox[0], bbox[1]), 
                        (bbox[2], bbox[3]), 
                        color3, 2)
        (tw, th), _ = cv2.getTextSize(class_names[label], cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(sample['image'], 
                        (bbox[0], bbox[1] - 25), 
                        (bbox[0] + tw, bbox[1]), 
                        color3, -1)
        cv2.putText(sample['image'], 
                    class_names[label], 
                    (bbox[0], bbox[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (255,255,255), 2)
    cv2.imshow(cv_window_name, sample['image'])
    cv2.waitKey() 
    
class DatumaroAnnotation:
    """
    This class is used to create a datumaro annotation object.
    The annotation object is used to create a datumaro dataset.
    The annotation object is used to create a datumaro dataset.
    """

    def __init__(self, 
                 format: str,
                #  categories: list
                 ):
        self.format = format
        self.annotations = []
        
    def toCOCO(self, 
            sample,
            model_output, 
            sample_idx, 
            visualize=False,
            class_names=None):
        predictions = []
        tmp = {
            'bbox':[],
            'id':'1',
            'label_id':'',
            'score':1,
            }
        if len(model_output[1]) == 0:
            pass
        else:
            for obj in range(len(model_output[1])):
                start_cord, end_cord = (model_output[0][obj, 0], model_output[0][obj, 1]), \
                                    (model_output[0][obj, 2], model_output[0][obj, 3])
                x, y, w, h = (
                    np.round((start_cord[0] + end_cord[0]) / 2, 2),
                    np.round((start_cord[1] + end_cord[1]) / 2, 2),
                    np.round(end_cord[0] - start_cord[0], 2),
                    np.round(end_cord[1] - start_cord[1], 2),
                )
                assert x > 0 and y > 0 and w > 0 and h > 0
                tmp['bbox'] = [x, y, w, h]
                tmp['score'] = np.round(model_output[2][obj], 2)
                tmp['id'] = str(int(model_output[1][obj]))
                tmp['label_id'] = str(sample_idx)
                tmp['label'] = class_names[int(model_output[1][obj])]
                predictions.append(tmp)
                tmp = {
                    'bbox':[],
                    'id':'1',
                    'label_id':'',
                    'score':1,
                }
        
                # Visualize the predictions generated by triton inference
                if visualize:
                    label = class_names[int(model_output[1][obj])]
                    # Get text size
                    (tw, th), _ = cv2.getTextSize(
                        # "{} {} %".format(label, round(model_output[2][obj] * 100, 2)),
                        "{}".format(label),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        2,
                    )
                    # Plot prediction box
                    cv2.rectangle(
                        sample["image"],
                        (int(model_output[0][obj, 0]), int(model_output[0][obj, 1])),
                        (int(model_output[0][obj, 2]), int(model_output[0][obj, 3])),
                        color1,
                        2,
                    )
                    # Plot prediction label background box
                    cv2.rectangle(
                        sample["image"],
                        (int(start_cord[0]), int(start_cord[1] - 25)),
                        (int(start_cord[0] + tw), int(start_cord[1])),
                        color1,
                        -1,
                    )
                    # Put class label and confidence value
                    cv2.putText(
                        sample["image"],
                        # "{} {} %".format(label, round(model_output[2][obj], 2) * 100),
                        "{}".format(label),
                        (int(model_output[0][obj, 0]), int(model_output[0][obj, 1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        text_color,
                        2,
                    )
        return predictions
    
    def toKITTI(self, 
                sample,
                model_output, 
                sample_idx, 
                visualize=False,
                class_names=None):
        """
        This function converts the model output to KITTI format.
        """
        predictions = []
        tmp = {
                'id':'1',   # Normally this is the PC file name but here we use the sample uuid
                'type': "cuboid_3d",
                "attributes": {"occluded": False},
                "group": 0, 
                'label_id':0, # unique id for each object instance
                'score':1,  # confidence score
                "position": [], # center of the cuboid x,y,z
                "rotation": [], # rotation of the cuboid in radian w.r.t the x,y,z axis
                "scale": [], # w,h,l
                }
        
        if len(model_output[1]) == 0:
            pass
        else:
            # TODO check if these values are consistent with the DATUMARO format. It works with OpenPCDet
            for obj in range(len(model_output[1])):
                tmp['label_id'] = model_output[2][obj]
                tmp['position'] = model_output[0][obj, 0:3]
                tmp['scale'] = model_output[0][obj, 3:6]
                tmp['rotation'] = np.array([0, 0, model_output[0][obj, 6] + 1e-10])
                tmp['score'] = model_output[1][obj]
                tmp['id'] = sample['uuid']
                predictions.append(tmp)  
            return predictions
        
        
# DEPRECATED
def process_model_output(sample,
                         model_output, 
                         sample_idx, 
                         visualize=False,
                         class_names=None):
    """
    
    """
    predictions = []
    tmp = {
            'bbox':[],
            'id':'1',
            'label_id':'',
            'score':1,
            }
    if len(model_output[1]) == 0:
        pass
    else:
        for obj in range(len(model_output[1])):
            start_cord, end_cord = (model_output[0][obj, 0], model_output[0][obj, 1]), \
                                (model_output[0][obj, 2], model_output[0][obj, 3])
            x, y, w, h = (
                np.round((start_cord[0] + end_cord[0]) / 2, 2),
                np.round((start_cord[1] + end_cord[1]) / 2, 2),
                np.round(end_cord[0] - start_cord[0], 2),
                np.round(end_cord[1] - start_cord[1], 2),
            )
            assert x > 0 and y > 0 and w > 0 and h > 0
            tmp['bbox'] = [x, y, w, h]
            tmp['score'] = np.round(model_output[2][obj], 2)
            tmp['id'] = str(int(model_output[1][obj]))
            tmp['label_id'] = str(sample_idx)
            tmp['label'] = class_names[int(model_output[1][obj])]
            predictions.append(tmp)
            tmp = {
                'bbox':[],
                'id':'1',
                'label_id':'',
                'score':1,
            }
            # Visualize the predictions generated by triton inference
            if visualize:
                label = class_names[int(model_output[1][obj])]
                # Get text size
                (tw, th), _ = cv2.getTextSize(
                    # "{} {} %".format(label, round(model_output[2][obj] * 100, 2)),
                    "{}".format(label),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    2,
                )
                # Plot prediction box
                cv2.rectangle(
                    sample["image"],
                    (int(model_output[0][obj, 0]), int(model_output[0][obj, 1])),
                    (int(model_output[0][obj, 2]), int(model_output[0][obj, 3])),
                    color1,
                    2,
                )
                # Plot prediction label background box
                cv2.rectangle(
                    sample["image"],
                    (int(start_cord[0]), int(start_cord[1] - 25)),
                    (int(start_cord[0] + tw), int(start_cord[1])),
                    color1,
                    -1,
                )
                # Put class label and confidence value
                cv2.putText(
                    sample["image"],
                    # "{} {} %".format(label, round(model_output[2][obj], 2) * 100),
                    "{}".format(label),
                    (int(model_output[0][obj, 0]), int(model_output[0][obj, 1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    text_color,
                    2,
                )
    return predictions