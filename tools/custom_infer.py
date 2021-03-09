import numpy as np
import copy
import json
import os
import sys
import torch
import time 
import sys 
sys.path.append('/content/Centerpoint_PC')
import argparse


from pyquaternion import Quaternion
from det3d import torchie
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.core.input.voxel_generator import VoxelGenerator



detection_class_map = {
    5: 'barrier',
    7: 'bicycle',
    3: 'bus',
    0: 'car',
    2: 'construction_vehicle',
    6 :'motorcycle',
    8: 'pedestrian',
    9: 'traffic_cone',
    4: 'trailer',
    1: 'truck'
}

def parse_args():
    parser = argparse.ArgumentParser(description="Custome data")
    parser.add_argument("--config", help="Configy file path ")
    parser.add_argument("--save_dir", required=True, help="the dir to save logs and models")
    parser.add_argument("--checkpoint", help="the dir to checkpoint which the model read from")
    parser.add_argument("--pc_folder", help="the dir to load all the point cloud files from folder ")
    args = parser.parse_args()
    return args


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



def pred2string(scores, labels, pred_bbox, txtpath,outputfolder):
    

    # truncation =0 
    # occlusion=0 
    # alpha=-10.0
    # bbox_2d= [0,0,0,0]
    file2write = open(os.path.join(outputfolder,txtpath[:-3]+ 'txt'),'w')

    for score,label,bbox in zip(scores, labels, pred_bbox):

        # Prepare output.
        name= detection_class_map[label]
        name += ' '
        # trunc = '{:.2f} '.format(truncation)
        # occ = '{:d} '.format(occlusion)
        # a = '{:.2f} '.format(alpha)
        # bb = '{:.2f} {:.2f} {:.2f} {:.2f} '.format(bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3])
        wlh = '{:.2} {:.2f} {:.2f} '.format(bbox[3], bbox[4], bbox[5])  # height, width, length.
        xyz = '{:.2f} {:.2f} {:.2f} '.format(bbox[0], bbox[1], bbox[2])  # x, y, z.
        y = '{:.2f} '.format(bbox[8])  # Yaw angle.
        vx= '{:.2f} '.format(bbox[6])
        vy= '{:.2f} '.format(bbox[7])
        s = ' {:.4f}\n'.format(score)  # Classification score.

        output = name + xyz+wlh + y +vx+vy+s
        # output = name + trunc + occ + a + bb + hwl + xyz + y
        # if ~np.isnan(box.score):
        #     output += s
        file2write.write(output)
    file2write.close()





class CenterPoint:
    def __init__(self, config_path, model_path,savingpath):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None
        self.savepth =savingpath
        
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

    # def run(self, points):
    #     t_t = time.time()
    #     print(type(points))
    #     print(points.shape)
    #     print(f"input points shape: {points.shape}")
    #     num_features =  4       
    #     self.points = points.reshape([-1, num_features])
    #     self.points[:, 3] = 0 # timestamp value #self.points = np.hstack((aself.points, np.zeros((self.points.shape[0], 1), dtype=self.points.dtype))) 
        
    #     voxels, coords, num_points = self.voxel_generator.generate(self.points)
    #     num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
    #     grid_size = self.voxel_generator.grid_size
    #     coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values = 0)
        
    #     voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
    #     coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
    #     num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
    #     num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=self.device)
        
    #     self.inputs = dict(
    #         voxels = voxels,
    #         num_points = num_points,
    #         num_voxels = num_voxels,
    #         coordinates = coords,
    #         shape = [grid_size]
    #     )
    #     torch.cuda.synchronize()
    #     t = time.time()

    #     with torch.no_grad():
    #         outputs = self.net(self.inputs, return_loss=False)[0]
    
    #     # print(f"output: {outputs}")
        
    #     torch.cuda.synchronize()
    #     print("  network predict time cost:", time.time() - t)

    #     outputs = remove_low_score_nu(outputs, 0.45)

    #     boxes_lidar = outputs["box3d_lidar"].detach().cpu().numpy()
    #     print("  predict boxes:", boxes_lidar.shape)

    #     scores = outputs["scores"].detach().cpu().numpy()
    #     types = outputs["label_preds"].detach().cpu().numpy()

    #     boxes_lidar[:, -1] = -boxes_lidar[:, -1] - np.pi / 2

    #     print(f"  total cost time: {time.time() - t_t}")

        

    #     return scores, boxes_lidar, types




    def run(self, pc_folder):

        for ipfile in os.listdir(pc_folder):
            t_t = time.time()
            print("Processing File is {}".format(ipfile))
            self.points =  np.fromfile(os.path.join(pc_folder,ipfile),dtype=np.float32)           
            print(points.shape)
            # print(f"input points shape: {points.shape}")
            num_features =  4       
            self.points = points.reshape([-1, num_features])
            self.points[:, 3] = 0 # timestamp value #self.points = np.hstack((aself.points, np.zeros((self.points.shape[0], 1), dtype=self.points.dtype))) 
            
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

            pred2string(scores, boxes_lidar, types,ipfile,self.savepth)

            # return scores, boxes_lidar, types



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
        #                                         truncation=truncated, occlusion=occluded,bbox_2d=bbox_2d)

        #                 # Write to disk.
        #         label_file.write(output + '\n')   



def main():
    args = parse_args()  

    config_path = args.config
    model_path = args.checkpoint
    opsavepath =args.save_dir   
    pc_folder = args.pc_folder

    # config_path = '/content/Centerpoint_PC/configs/nusc/pp/cust_cntpt_pp02voxel_2fcn10sweep.py'
    # model_path = '/content/drive/MyDrive/PointCloud_model/centerpoint_nusc/latest.pth'
    # opsavepath ='/content/drive/MyDrive/PointCloud_model'   
    # pc_folder = '/content/drive/MyDrive/PointCloudData/customer'

    cntrpt = CenterPoint(config_path, model_path,opsavepath)    
    cntrpt.initialize()
    cntrpt.run(pc_folder)



if __name__ == "__main__":

    main()

    # ###cmd : custom_infer.py 
    # config_path = '/content/Centerpoint_PC/configs/nusc/pp/cust_cntpt_pp02voxel_2fcn10sweep.py'
    # model_path = '/content/drive/MyDrive/PointCloud_model/centerpoint_nusc/latest.pth'
    # opsavepath ='/content/drive/MyDrive/PointCloud_model'
   
    # pc_folder = '/content/drive/MyDrive/PointCloudData/customer'
    

    # cntrpt = CenterPoint(config_path, model_path,opsavepath)    
    # cntrpt.initialize()
    # cntrpt.run(pc_folder)

    