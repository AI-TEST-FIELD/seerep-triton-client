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
    
def visualize_groundtruth(sample: dict, cv_window_name: str, class_names: list):
    # TODO add support for either ground truth or predictions
    
    for ann_idx, ann in enumerate(sample['annotations']['items'][0]['annotations']):
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
    
def process_model_output(sample,
                         model_output, 
                         sample_idx, 
                         visualize=False,
                         class_names=None):
    """
    
    """
    predictions = {
                'annotations':[],
                'dm_format_version':1,
                    }
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
            predictions['annotations'].append(tmp)
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