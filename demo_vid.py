from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import json
from typing import Dict, Optional

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO 
import re
import glob
LIGHT_PURPLE=(0.25098039,  0.274117647,  0.65882353)

def main():
    parser = argparse.ArgumentParser(description='WiLoR demo code')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*.jpeg'], help='List of file extensions to consider')

    args = parser.parse_args()

    # Download and load checkpoints
    model, model_cfg = load_wilor(checkpoint_path = './pretrained_models/wilor_final.ckpt' , cfg_path= './pretrained_models/model_config.yaml')
    detector = YOLO('./pretrained_models/detector.pt')
    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    renderer_side = Renderer(model_cfg, faces=model.mano.faces)
    
    device   = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model    = model.to(device)
    detector = detector.to(device)
    model.eval()

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    def natural_sort_key(s):
        """Natural sorting key function for proper numeric sorting"""
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

    img_paths = []
    for end in args.file_type:
        img_paths.extend(Path(args.img_folder).glob(end))
    img_paths = list(set(img_paths))
    img_paths = sorted(img_paths, key=natural_sort_key)
    imgs = []

    mano_verts = []
    mano_joints = []
    mano_joints_2d = []

    mano_global_orient = []
    mano_hand_pose = []
    mano_betas = []

    mano_masks = [] # hand masks

    # Iterate over all images in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        detections = detector(img_cv2, conf = 0.3, verbose=False)[0]
        bboxes    = []
        is_right  = []
        for det in detections: 
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())
        
        if len(bboxes) == 0:
            # 如果没有找到手部检测框，则添加一个全零的占位符
            print(f"No hands detected in image {img_path}, ADD Zero to keep the shape.")
            mano_verts.append([np.zeros((778, 3))]) #  list (778, 3)
            mano_joints.append([np.zeros((21, 3))]) # list (21, 3)
            mano_joints_2d.append([np.zeros((21, 2))]) # list (21, 2)
            mano_global_orient.append([np.zeros((1, 3, 3))]) # list (1, 3, 3)
            mano_hand_pose.append([np.zeros((15, 3, 3))]) # list (15, 3, 3)
            mano_betas.append([np.zeros((10))]) # list (10)
            mano_masks.append(np.zeros(img_cv2.shape[:2], dtype=bool))
            continue
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_joints= []
        all_kpts  = []
        all_jts_2d   = []

        all_pred_mano_global_orient = []
        all_pred_mano_hand_pose = []
        all_pred_mano_betas = []

        for batch in dataloader: 
            batch = recursive_to(batch, device)
    
            with torch.no_grad():
                out = model(batch) 
                
            multiplier    = (2*batch['right']-1)
            pred_cam      = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center    = batch["box_center"].float()
            box_size      = batch["box_size"].float()
            img_size      = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full     = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            
            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                
                verts  = out['pred_vertices'][n].detach().cpu().numpy() #(778, 3), xyz
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy() #(21, 3), xyz
                
                is_right    = batch['right'][n].cpu().numpy()
                verts[:,0]  = (2*is_right-1)*verts[:,0] # Flip the x-axis for left hand
                joints[:,0] = (2*is_right-1)*joints[:,0]
                cam_t = pred_cam_t_full[n]
                kpts_2d = project_full_img(verts, cam_t, scaled_focal_length, img_size[n])
                jts_2d = project_full_img(joints, cam_t, scaled_focal_length, img_size[n])

                pred_mano_global_orient = out['mano_global_orient'][n].detach().cpu().numpy() # (1, 3, 3), root, global
                pred_mano_hand_pose = out['mano_hand_pose'][n].detach().cpu().numpy() # (15, 3, 3), local
                pred_mano_betas = out['mano_betas'][n].detach().cpu().numpy() # (10)
                # mano_output = self.mano(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)

                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                all_joints.append(joints)
                all_kpts.append(kpts_2d)
                all_jts_2d.append(jts_2d)

                all_pred_mano_global_orient.append(pred_mano_global_orient)
                all_pred_mano_hand_pose.append(pred_mano_hand_pose)
                all_pred_mano_betas.append(pred_mano_betas)

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_PURPLE, is_right=is_right)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{n}.obj'))

        # Render front view
        if len(all_verts) > 0:
            # print(len(all_verts),len(all_joints),len(all_jts_2d),len(all_pred_mano_global_orient),len(all_pred_mano_hand_pose),len(all_pred_mano_betas))
            # print("xxxxxxxxxxxx")
            misc_args = dict(
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(0, 0, 0),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
            
            # Save overlay image
            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}.jpg'), 255*input_img_overlay[:, :, ::-1])
            imgs.append(input_img_overlay[:,:,::-1]*255)

            # Save cam_view img and hand mask
            # cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_rendered.png'), 255*cam_view[:,:,::-1])
            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_mask.png'), 255*(cam_view[:,:,3]>0).astype(np.uint8))
            
            # Save all detected hands (modified to support multi-hand filtering)
            mano_verts.append(all_verts) #  list of (778, 3)
            mano_joints.append(all_joints) # list of (21, 3)
            mano_joints_2d.append(all_jts_2d) # list of (21, 2)

            mano_global_orient.append(all_pred_mano_global_orient) # list of (1, 3, 3)
            mano_hand_pose.append(all_pred_mano_hand_pose) # list of (15, 3, 3)
            mano_betas.append(all_pred_mano_betas) # list of (10)

            mano_masks.append(cam_view[:,:,3]>0)

            
    # Save video
    if len(imgs) > 0:
        height, width = imgs[0].shape[:2]
        video_writer = cv2.VideoWriter(os.path.join(args.out_folder, 'result.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        for img in imgs:
            video_writer.write(img.astype(np.uint8))
        video_writer.release()
        print(f"Video saved to {os.path.join(args.out_folder, 'result.mp4')}")

    # Save the vertices
    if len(mano_verts) > 0:
        # Convert to numpy arrays with object dtype to handle inhomogeneous shapes
        np.savez_compressed(os.path.join(args.out_folder, 'mano_verts.npz'), mano_verts=np.array(mano_verts, dtype=object))
        np.savez_compressed(os.path.join(args.out_folder, 'mano_joints.npz'), mano_joints=np.array(mano_joints, dtype=object))
        np.savez_compressed(os.path.join(args.out_folder, 'mano_joints_2d.npz'), mano_joints_2d=np.array(mano_joints_2d, dtype=object))
        np.savez_compressed(os.path.join(args.out_folder, 'mano_global_orient.npz'), mano_global_orient=np.array(mano_global_orient, dtype=object))
        np.savez_compressed(os.path.join(args.out_folder, 'mano_hand_pose.npz'), mano_hand_pose=np.array(mano_hand_pose, dtype=object))
        np.savez_compressed(os.path.join(args.out_folder, 'mano_betas.npz'), mano_betas=np.array(mano_betas, dtype=object))
        np.savez_compressed(os.path.join(args.out_folder, 'mano_masks.npz'), mano_masks=np.array(mano_masks, dtype=object))

def project_full_img(points, cam_trans, focal_length, img_res): 
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3) 
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = camera_center[0]
    K[1,2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:] 
    
    V_2d = (K @ points.T).T 
    return V_2d[..., :-1]

if __name__ == '__main__':
    main()
