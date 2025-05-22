import numpy as np
import open3d
# DEPRECATED
def postprocess_model_output(sample:dict, 
                             model_output:np.array,
                             sample_idx:int,
                             visualize:bool=False,
                             class_names:list=None):
    """
    Postprocess the model output by converting it to datumaro string format to be digested by SEEREP.
    The function also visualizes the point cloud if specified.
    Args:
        sample (dict): The input sample containing the point cloud data.
        model_prediction (np.array): The model prediction to be visualized.
        sample_idx (int): The index of the sample.
        visualize (bool): Whether to visualize the point cloud or not.
        class_names (list): List of class names for visualization.
    """
    predictions = []
    tmp = {
        'id':'1',   # string class idx
        'type': "cuboid_3d",
        "attributes": {"occluded": False},
        "group": 0,
        'label_id':0, # sample ID          
        'score':1,
        "position": [], # center of the cuboid x,y,z
        "rotation": [], # rotation of the cuboid in radian w.r.t the x,y,z axis
        "scale": [], # w,h,l
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