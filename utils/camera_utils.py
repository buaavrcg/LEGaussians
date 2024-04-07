import os
import json
import torch
from PIL import Image
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, PILtoTorch_wo_res
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask, language_feature_indices=cam_info.language_feature_indices,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def clsCamera_to_JSON(id, img_name, camera : Camera):
    camera_entry = {
        'uid' : id,
        'colmap_id': camera.colmap_id,
        'img_name' : img_name,
        'R' : [x.tolist() for x in camera.R],
        'T' : camera.T.tolist(),
        'FoVx': camera.FoVx,
        'FoVy': camera.FoVy,
    }
    return camera_entry

def JSON_to_clsCamera(nvs_path, language_feature_indices_path):
    with open(os.path.join(nvs_path, "cameras.json"), 'r') as file:
        cams_json = json.load(file)
    indices = torch.load(os.path.join(nvs_path, language_feature_indices_path)).cpu().numpy()
    # To match the order of indices (due to the sort of paths list in the func get_image_paths)
    cams_json = sorted(cams_json, key=lambda x: x["img_name"])

    cams = []
    for i in range(len(cams_json)):
        cam_json = cams_json[i]
        gt_image = PILtoTorch_wo_res(Image.open(os.path.join(nvs_path, cam_json["img_name"].split("/")[-1])))
        cam = Camera(colmap_id=cam_json["colmap_id"], R=np.array(cam_json["R"]), T=np.array(cam_json["T"]), 
                FoVx=cam_json["FoVx"], FoVy=cam_json["FoVy"], 
                image=gt_image, gt_alpha_mask=None, language_feature_indices=indices[i],
                image_name=cam_json["img_name"], uid=cam_json["uid"], is_novel_view=True)
        cams.append(cam)
    
    return cams