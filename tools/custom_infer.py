import numpy as np
import copy
import json
import os
import sys
import torch
import time 

import BoundingBox, BoundingBoxArray

from pyquaternion import Quaternion
from det3d import __version__, torchie
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.core.input.voxel_generator import VoxelGenerator




def yaw2quaternion(yaw: float) -> Quaternion:
    return Quaternion(axis=[0,0,1], radians=yaw)



def get_annotations_indices(types, thresh, label_preds, scores):
    indexs = []
    annotation_indices = []
    for i in range(label_preds.shape[0]):
        if label_preds[i] == types:
            indexs.append(i)
    for index in indexs:
        if scores[index] >= thresh:
            annotation_indices.append(index)
    return annotation_indices  


def remove_low_score_nu(image_anno, thresh):
    img_filtered_annotations = {}
    label_preds_ = image_anno["label_preds"].detach().cpu().numpy()
    scores_ = image_anno["scores"].detach().cpu().numpy()
    
    car_indices =                  get_annotations_indices(0, 0.4, label_preds_, scores_)
    truck_indices =                get_annotations_indices(1, 0.4, label_preds_, scores_)
    construction_vehicle_indices = get_annotations_indices(2, 0.4, label_preds_, scores_)
    bus_indices =                  get_annotations_indices(3, 0.3, label_preds_, scores_)
    trailer_indices =              get_annotations_indices(4, 0.4, label_preds_, scores_)
    barrier_indices =              get_annotations_indices(5, 0.4, label_preds_, scores_)
    motorcycle_indices =           get_annotations_indices(6, 0.15, label_preds_, scores_)
    bicycle_indices =              get_annotations_indices(7, 0.15, label_preds_, scores_)
    pedestrain_indices =           get_annotations_indices(8, 0.1, label_preds_, scores_)
    traffic_cone_indices =         get_annotations_indices(9, 0.1, label_preds_, scores_)
    
    for key in image_anno.keys():
        if key == 'metadata':
            continue
        img_filtered_annotations[key] = (
            image_anno[key][car_indices +
                            pedestrain_indices + 
                            bicycle_indices +
                            bus_indices +
                            construction_vehicle_indices +
                            traffic_cone_indices +
                            trailer_indices +
                            barrier_indices +
                            truck_indices
                            ])

    return img_filtered_annotations


def box_to_string(name: str,
                      box: Box,
                      bbox_2d: Tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0),
                      truncation: float = -1.0,
                      occlusion: int = -1,
                      alpha: float = -10.0,
                      score=scores) -> str:
        """
        Convert box in KITTI image frame to official label string fromat.
        :param name: KITTI name of the box.
        :param box: Box class in KITTI image frame.
        :param bbox_2d: Optional, 2D bounding box obtained by projected Box into image (xmin, ymin, xmax, ymax).
            Otherwise set to KITTI default.
        :param truncation: Optional truncation, otherwise set to KITTI default.
        :param occlusion: Optional occlusion, otherwise set to KITTI default.
        :param alpha: Optional alpha, otherwise set to KITTI default.
        :return: KITTI string representation of box.
        """
        # Convert quaternion to yaw angle.
        v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
        yaw = -np.arctan2(v[2], v[0])

        # Prepare output.
        name += ' '
        trunc = '{:.2f} '.format(truncation)
        occ = '{:d} '.format(occlusion)
        a = '{:.2f} '.format(alpha)
        bb = '{:.2f} {:.2f} {:.2f} {:.2f} '.format(bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3])
        hwl = '{:.2} {:.2f} {:.2f} '.format(box.wlh[2], box.wlh[0], box.wlh[1])  # height, width, length.
        xyz = '{:.2f} {:.2f} {:.2f} '.format(box.center[0], box.center[1], box.center[2])  # x, y, z.
        y = '{:.2f}'.format(yaw)  # Yaw angle.
        s = ' {:.4f}'.format(scores)  # Classification score.

        output = name + trunc + occ + a + bb + hwl + xyz + y
        if ~np.isnan(box.score):
            output += s

        return output



class CenterPoint:
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None
        
    def initialize(self):
        self.read_config()
        
    def read_config(self):
        config_path = self.config_path
        cfg = Config.fromfile(self.config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        self.net.load_state_dict(torch.load(self.model_path)["state_dict"])
        self.net = self.net.to(self.device).eval()

        self.range = cfg.voxel_generator.range
        self.voxel_size = cfg.voxel_generator.voxel_size
        self.max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
        self.max_voxel_num = cfg.voxel_generator.max_voxel_num
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num,
        )

    def run(self, points):
        t_t = time.time()
        print(f"input points shape: {points.shape}")
        num_features = 4        
        self.points = points.reshape([-1, num_features])
        self.points[:, 3] = 0 # timestamp value 
        
        voxels, coords, num_points = self.voxel_generator.generate(self.points)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        grid_size = self.voxel_generator.grid_size
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values = 0)
        
        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=self.device)
        
        self.inputs = dict(
            voxels = voxels,
            num_points = num_points,
            num_voxels = num_voxels,
            coordinates = coords,
            shape = [grid_size]
        )
        torch.cuda.synchronize()
        t = time.time()

        with torch.no_grad():
            outputs = self.net(self.inputs, return_loss=False)[0]
    
        # print(f"output: {outputs}")
        
        torch.cuda.synchronize()
        print("  network predict time cost:", time.time() - t)

        outputs = remove_low_score_nu(outputs, 0.45)

        boxes_lidar = outputs["box3d_lidar"].detach().cpu().numpy()
        print("  predict boxes:", boxes_lidar.shape)

        scores = outputs["scores"].detach().cpu().numpy()
        types = outputs["label_preds"].detach().cpu().numpy()

        boxes_lidar[:, -1] = -boxes_lidar[:, -1] - np.pi / 2

        print(f"  total cost time: {time.time() - t_t}")

        

        return scores, boxes_lidar, types

    def bbox_conv(self,scores, det_box):
        pass

        # bbox_preds=BoundingBoxArray() #/[] 
        # if scores.size != 0:
        #     for i in range(scores.size):
        #         bbox = BoundingBox()
        #         q = yaw2quaternion(float(det_box[i][8]))
        #         bbox.pose.orientation.x = q[1]
        #         bbox.pose.orientation.y = q[2]
        #         bbox.pose.orientation.z = q[3]
        #         bbox.pose.orientation.w = q[0]           
        #         bbox.pose.position.x = float(det_box[i][0])
        #         bbox.pose.position.y = float(det_box[i][1])
        #         bbox.pose.position.z = float(det_box[i][2])
        #         bbox.dimensions.x = float(det_box[i][4])
        #         bbox.dimensions.y = float(det_box[i][3])
        #         bbox.dimensions.z = float(det_box[i][5])
        #         bbox.value = scores[i]
        #         # bbox.label = int(types[i])

        #         # # Convert quaternion to yaw angle.
        #         # v = np.dot(bbox.rotation_matrix, np.array([1, 0, 0]))
        #         # yaw = -np.arctan2(v[2], v[0])

        #         bbox_preds.boxes.append(bbox)

        # return bbox_preds


    def log_predkitti(self,scores,detbox,types):
        pass

        # # Write label file.
        # label_path = os.path.join(label_folder, sample_token + '.txt')
        # if os.path.exists(label_path):
        #     print('Skipping existing file: %s' % label_path)
        #     #continue
        # else:
        #     print('Writing file: %s' % label_path)

        # with open(label_path, "w") as label_file:
        #                     # Truncated: Set all objects to 0 which means untruncated.
        #         truncated = 0.0

        #         # Occluded: Set all objects to full visibility as this information is not available in nuScenes.
        #         occluded = 0

        #         detection_name = types

        #         if detection_name is None:
        #             detection_name = 'Unk' #continue

        #         # Convert box to output string format.
        #         output = box_to_string(name=detection_name, box=detbox,score=scores, 
        #                                         truncation=truncated, occlusion=occluded) #bbox_2d=bbox_2d,

        #                 # Write to disk.
        #         label_file.write(output + '\n')   

    def savpredpkl(self):
        pass










if __name__ == "__main__":
            ## CenterPoint
    config_path = 'configs/nusc/voxelnet/custom_cntpt_voxelnet_0075voxel_dcn_flip.py'
    model_path = 'models/last.pth'
    pc_data = np.fromfile('',dtype=np.float32)

    cntrpt = CenterPoint(config_path, model_path)
    
    cntrpt.initialize()
    scores, dt_box_lidar,types = cntrpt.run(pc_data)
    # bbox = cntrpt.bbox_conv(scores,dt_box_lidar)

    print("The scores {} and bbox {} with label {}".format(scores,dt_box_lidar,types))
    