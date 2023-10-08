from __future__ import absolute_import, division, print_function

import time
import json
import datasets
import networks
import numpy as np
import torch.optim as optim
import torch.nn as nn
# generate random integer values
from random import seed
from random import randint
import copy
 
from utils import *
from layers import *
from torch.utils.data import DataLoader
import wandb

wandb.init(project="MySfMLearner", entity="respinosa")

#from tensorboardX import SummaryWriter


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}  # 字典
        self.parameters_to_train = []  # 列表

        #self.device = torch.device("cuda")
        self.device = torch.device("cuda")

        self.num_scales = len(self.opt.scales)  # 4
        self.num_input_frames = len(self.opt.frame_ids)  # 3
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames  # 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")  # 18
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.models["position_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)  # 18
        self.models["position_encoder"].to(self.device)

        self.models["position"] = networks.PositionDecoder(
            self.models["position_encoder"].num_ch_enc, self.opt.scales)

        self.models["position"].to(self.device)

        self.models["transform_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)  # 18
        self.models["transform_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["transform_encoder"].parameters())

        self.models["transform"] = networks.TransformDecoder(
            self.models["transform_encoder"].num_ch_enc, self.opt.scales)
        self.models["transform"].to(self.device)
        self.parameters_to_train += list(self.models["transform"].parameters())

        self.models["lighting"] = networks.LightingDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["lighting"].to(self.device)
        self.parameters_to_train += list(self.models["lighting"].parameters())

        self.models["motion_flow"] = networks.ResidualFLowDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["motion_flow"].to(self.device)
        self.parameters_to_train += list(self.models["motion_flow"].parameters())

        if self.use_pose_net:

            if self.opt.pose_model_type == "separate_resnet":
                print("Pose encoder ResnetEncoder")
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)
                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "cnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"endovis": datasets.SCAREDRAWDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.jpg'  

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=1, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        #self.writers = {}
        #for mode in ["train", "val"]:
        #    self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)


        self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width))
        self.spatial_transform.to(self.device)

        self.get_occu_mask_backward = get_occu_mask_backward((self.opt.height, self.opt.width))
        self.get_occu_mask_backward.to(self.device)

        self.get_occu_mask_bidirection = get_occu_mask_bidirection((self.opt.height, self.opt.width))
        self.get_occu_mask_bidirection.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        self.position_depth = {}
        
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

            self.position_depth[scale] = optical_flow((h, w), self.opt.batch_size, h, w)
            self.position_depth[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        self.models["encoder"].train()
        self.models["depth"].train()
        self.models["transform_encoder"].train()
        self.models["transform"].train()
        self.models["pose_encoder"].train()
        self.models["pose"].train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        self.models["encoder"].eval()
        self.models["depth"].eval()
        self.models["transform_encoder"].eval()
        self.models["transform"].eval()
        self.models["pose_encoder"].eval()
        self.models["pose"].eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # self.model_lr_scheduler.step()
        r = randint(0, 63)
        print("Training - " + str(r))
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs,r)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            phase = batch_idx % self.opt.log_frequency == 0

            if phase:

                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val(r)

            self.step += 1
            
        self.model_lr_scheduler.step()


    def process_batch(self, inputs,r):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        
        #DepthNet Prediction
        features = self.models["encoder"](inputs["color_aug", 0, 0])
        #self.models["acual_encoder_depth"]=copy.deepcopy(self.models["encoder"])
        #weights_path = 'depth_weights_temp.pth'
        #torch.save(self.models["encoder"].state_dict(), weights_path)
        outputs = self.models["depth"](features)
        outputs["f1"] = features[0][:,r,:, :].detach()
        #print("Shape of feaures depth encoder")
        #print(features[1].shape)
    
        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features, outputs))
            #outputs.update(self.predict_lighting(inputs, features, outputs))
        self.generate_images_pred(inputs, outputs,r)

        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features, disps):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                #pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            
            
            for f_i in self.opt.frame_ids[1:]:
                #print("Entro"+str(f_i))
                if f_i != "s":

                    inputs_all = [pose_feats[f_i], pose_feats[0]]
                    inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

                    # OF Prediction normal and reversed
                    position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
                    position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                    outputs_0 = self.models["position"](position_inputs)
                    outputs_1 = self.models["position"](position_inputs_reverse)

                    for scale in self.opt.scales:
                        outputs["p_"+str(scale)+"_"+str(f_i)] = outputs_0["position_"+str(scale)]
                        outputs["ph_"+str(scale)+"_"+str(f_i)] = F.interpolate(
                            outputs["p_"+str(scale)+"_"+str(f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                        #outputs["r_"+str(scale)+"_"+str(f_i)] = self.spatial_transform(inputs[("color", f_i, 0)], outputs["ph_"+str(scale)+"_"+str(f_i)])
                        outputs["pr_"+str(scale)+"_"+str(f_i)] = outputs_1["position_"+str(scale)]
                        outputs["prh_"+str(scale)+"_"+str(f_i)] = F.interpolate(
                            outputs["pr_"+str(scale)+"_"+str(f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                        
                        outputs["omaskb_"+str(scale)+"_"+str(f_i)],  outputs["omapb_"+str(scale)+"_"+str(f_i)]= self.get_occu_mask_backward(outputs["prh_"+str(scale)+"_"+str(f_i)])
                        outputs["omapbi_"+str(scale)+"_"+str(f_i)] = self.get_occu_mask_bidirection(outputs["ph_"+str(scale)+"_"+str(f_i)],
                                                                                                          outputs["prh_"+str(scale)+"_"+str(f_i)])

                    # Input for the AFNet
                    #transform_input = [outputs["r_0"+"_"+str(f_i)], inputs[("color", 0, 0)]]
                    # Output from AFNet
                    #transform_inputs = self.models["transform_encoder"](torch.cat(transform_input, 1))
                    #outputs_2 = self.models["transform"](transform_inputs)

                    # Input for PoseNet
                    pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all, 1))]
                    #print(len(pose_inputs))
                    #input_lighting = pose_inputs[0][2]
                    #input_lighting = self.models["pose_encoder"](torch.cat(inputs_all, 1)).lastlayer
                    #wandb.log({"inputs_pose_{}_{}".format(f_i, scale): wandb.Image(inputs_all[f_i])},step=self.step)
                    #wandb.log({"inputs_pose_{}_{}".format(f_i, scale): wandb.Image(inputs_all[0])},step=self.step)
                    axisangle, translation = self.models["pose"](pose_inputs)

                    # Input for Lighting
                    outputs_lighting = self.models["lighting"](pose_inputs[0])

                    # Input motion flow
                    outputs_mf = self.models["motion_flow"](pose_inputs[0])
                    outputs["axisangle_0_"+str(f_i)] = axisangle
                    outputs["translation_0_"+str(f_i)] = translation
                    outputs["cam_T_cam_0_"+str(f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0])
                    #outputs["constrast_0_"+str(f_1)] = contrast
                    #outputs["constrast_0_"+str(f_1)] = brightness
                    
                    #if f_i < 0:
                    
                    for scale in self.opt.scales:
                        outputs["b_"+str(scale)+"_"+str(f_i)] = outputs_lighting[("lighting", scale)][:,0,None,:, :]
                        outputs["c_"+str(scale)+"_"+str(f_i)] = outputs_lighting[("lighting", scale)][:,1,None,:, :]
                        outputs["mf_"+str(scale)] = outputs_mf[("flow", scale)]
                        
                        #print(outputs["mf_"+str(scale)].shape)

                    
                   
                    
        return outputs

    def generate_images_pred(self, inputs, outputs,r):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            
            disp = outputs["disp_"+str(scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs["depth_"+str(scale)] = depth

            source_scale = 0
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs["cam_T_cam_0_"+str(frame_id)]
                #print("generate_images_pred"+str(frame_id))
                #   from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs["axisangle_0_"+str(f_i)]
                    translation = outputs["translation_0_"+str(f_i)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs["sample_"+str(frame_id)+"_"+str(scale)] = pix_coords

                #outputs["mf_"+str(scale)] = outputs["mf_"+str(scale)].reshape(12,256,128,2)
                flow = F.interpolate(
                    outputs["mf_"+str(scale)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
                #outputs["sample_"+str(frame_id)+"_"+str(scale)] = outputs["sample_"+str(frame_id)+"_"+str(scale)] + flow
                #print(outputs["sample_"+str(frame_id)+"_"+str(scale)].shape)
                #print(flow.shape)
                outputs["color_"+str(frame_id)+"_"+str(scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs["sample_"+str(frame_id)+"_"+str(scale)],
                    padding_mode="border",align_corners=True)
                #Flow
                outputs["color_"+str(frame_id)+"_"+str(scale)] = (flow * outputs[("occu_mask_backward", 0, f_i)].detach() + outputs["color_"+str(frame_id)+"_"+str(scale)])
                outputs[("color_", scale, frame_id)] = torch.clamp(outputs[("color_", scale, f_i)], min=0.0, max=1.0)
                #Lighting compensation - Funciona
                #if frame_id < 0:
                
                outputs["ch_"+str(scale)+"_"+str(frame_id)] = F.interpolate(
                            outputs["c_"+str(scale)+"_"+str(frame_id)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                outputs["bh_"+str(scale)+"_"+str(frame_id)] = F.interpolate(
                            outputs["b_"+str(scale)+"_"+str(frame_id)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)                            

                outputs["refinedCB_"+str(frame_id)+"_"+str(scale)] = outputs["ch_"+str(scale)+"_"+str(frame_id)] * outputs["color_"+str(frame_id)+"_"+str(scale)]  + outputs["bh_"+str(scale)+"_"+str(frame_id)]
                
                    #wandb.log({"CH_{}_{}".format(frame_id, scale): wandb.Image(outputs["ch_"+str(scale)+"_"+str(frame_id)].data)},step=self.step)
                    #wandb.log({"BH_{}_{}".format(frame_id, scale): wandb.Image(outputs["bh_"+str(scale)+"_"+str(frame_id)].data)},step=self.step)
                    #wandb.log({"refinedCB_{}_{}".format(frame_id, scale): wandb.Image(outputs["refinedCB_"+str(frame_id)+"_"+str(scale)].data)},step=self.step)
                #outputs["refined_"+str(frame_id)+"_"+str(scale)] = brightnes_equator(outputs["color_"+str(frame_id)+"_"+str(scale)],inputs[("color", frame_id, source_scale)])  
        
        #Feature similairty and depth consistency loss
        """
        self.models["encoder"].eval()
        self.models["depth"].eval()
        features = self.models["encoder"](outputs["color_"+str(-1)+"_"+str(0)].detach())
        #outputs["f2"] = features[0][:,r,:, :].detach()
        predicted_disp = self.models["depth"](features)
        _, predicted_depth = disp_to_depth(predicted_disp["disp_"+str(0)].detach(), self.opt.min_depth, self.opt.max_depth)
        outputs["pdepth_"+str(0)] = predicted_depth
        self.models["encoder"].train()
        self.models["depth"].train()"""
        
                
    def compute_reprojection_loss(self, pred, target):

        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_feature_similarity_loss(self, pred, target):
        fs_loss = self.ssim(pred, target).mean(1, True)
        return fs_loss

    def get_ilumination_invariant_loss(self, pred, target):
        features_p = get_ilumination_invariant_features(pred)
        features_t = get_ilumination_invariant_features(target)
        #abs_diff = torch.abs(features_t - features_p)
        #l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(features_p, features_t).mean(1, True)
        #ii_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        #print(ii_loss.shape)
        return ssim_loss
    
    #def get_motion_flow_loss(self, flow):
    def get_motion_flow_loss(self,motion_map):
        """A regularizer that encourages sparsity.
        This regularizer penalizes nonzero values. Close to zero it behaves like an L1
        regularizer, and far away from zero its strength decreases. The scale that
        distinguishes "close" from "far" is the mean value of the absolute of
        `motion_map`.
        Args:
            motion_map: A torch.Tensor of shape [B, C, H, W]
        Returns:
            A scalar torch.Tensor, the regularizer to be added to the training loss.
        """
        tensor_abs = torch.abs(motion_map)
        mean = torch.mean(tensor_abs, dim=(2, 3), keepdim=True).detach()
        # We used L0.5 norm here because it's more sparsity encouraging than L1.
        # The coefficients are designed in a way that the norm asymptotes to L1 in
        # the small value limit.
        return torch.mean(2 * mean * torch.sqrt(tensor_abs / (mean + 1e-24) + 1))

    

    def compute_losses(self, inputs, outputs):

        losses = {}
        total_loss = 0
        
        #outputs = outputs.reverse()
        for scale in self.opt.scales:
            
            loss = 0
            loss_reprojection = 0
            loss_ilumination_invariant = 0
            loss_transform = 0
            loss_cvt = 0
            feature_similarity_loss = 0
            depth_similarity_loss = 0
            loss_motion_flow = 0
            
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs["disp_"+str(scale)]
            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                
                occu_mask_backward = outputs["omaskb_"+str(0)+"_"+str(frame_id)].detach()
                occu_mask_backward_ = get_feature_oclution_mask(occu_mask_backward)
                #occu_mask_backward_ = get_ilumination_invariant_features(occu_mask_backward)
                #il = get_ilumination_invariant_features(outputs["color_"+str(frame_id)+"_"+str(scale)])
                #print(il.shape)
                #Original
                #loss_reprojection += (
                #    self.compute_reprojection_loss(outputs["refinedCB_"+str(frame_id)+"_"+str(scale)], inputs[("color",0,0)]) * occu_mask_backward).sum() / occu_mask_backward.sum()
                #Cambios   
                            
                loss_reprojection += (
                    self.compute_reprojection_loss(outputs["color_"+str(frame_id)+"_"+str(scale)], inputs[("color",0,0)]) * occu_mask_backward).sum() / occu_mask_backward.sum()
                loss_ilumination_invariant += (
                    self.get_ilumination_invariant_loss(outputs["color_"+str(frame_id)+"_"+str(scale)], inputs[("color",0,0)]) * occu_mask_backward_).sum() / occu_mask_backward_.sum()
            loss_motion_flow += (
                self.get_motion_flow_loss(outputs["mf_"+str(scale)])
            )
            

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += loss_reprojection / 2.0
            loss += 0.20 * loss_ilumination_invariant / 2.0

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            loss += 1e-4 * loss_motion_flow / (2 ** scale)
            #a = outputs["f1"].detach()
            #b = outputs["f2"].detach()
            #feature_similarity_loss += (self.compute_feature_similarity_loss(a,b)).sum() 

            #depth_similarity_loss += get_depth_loss(outputs["depth_"+str(0)].detach(),outputs["pdepth_"+str(0)].detach())

            #loss += 0.1 * feature_similarity_loss 
            #loss += 0.1 * depth_similarity_loss 
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses
    
    def val(self,r):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            #inputs = self.val_iter.next()
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)
            #inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch_val(inputs,r)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def process_batch_val(self, inputs,r):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            #features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features, outputs))
            #outputs.update(self.predict_lighting(inputs, features, outputs))

        self.generate_images_pred(inputs, outputs,r)
        losses = self.compute_losses_val(inputs, outputs)

        return outputs, losses


    def compute_losses_val(self, inputs, outputs):
        """Compute the reprojection, perception_loss and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:

            loss = 0
            registration_losses = []

            target = inputs[("color", 0, 0)]

            for frame_id in self.opt.frame_ids[1:]:
                registration_losses.append(
                    ncc_loss(outputs["color_"+str(frame_id)+"_"+str(scale)].mean(1, True), target.mean(1, True)))

            registration_losses = torch.cat(registration_losses, 1)
            registration_losses, idxs_registration = torch.min(registration_losses, dim=1)

            loss += registration_losses.mean()
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = -1 * total_loss

        return losses
    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        #writer = self.writers[mode]
        for l, v in losses.items():
            wandb.log({mode+"{}".format(l):v},step =self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids[1:]:
                    #if frame_id < 0:
                    wandb.log({mode+"_Output_{}_{}_{}".format(frame_id, s, j): wandb.Image(outputs["color_"+str(frame_id)+"_"+str(s)][j].data)},step=self.step)
                    wandb.log({mode+"_Refined_{}_{}_{}".format(frame_id, s, j): wandb.Image(outputs["refinedCB_"+str(frame_id)+"_"+str(s)][j].data)},step=self.step)
                    
                    wandb.log({mode+"_Brightness_{}_{}_{}".format(frame_id, s, j): wandb.Image(outputs["bh_"+str(s)+"_"+str(frame_id)][j].data)},step=self.step)

                    wandb.log({mode+"_Contrast_{}_{}_{}".format(frame_id, s, j): wandb.Image(outputs["ch_"+str(s)+"_"+str(frame_id)][j].data)},step=self.step)
                wandb.log({mode+"_Motion_Flow_{}_{}".format(s, j): wandb.Image(normalize_image(outputs["mf_"+str(s)][j]))},step=self.step)
                    
 
                wandb.log({mode+"_disp_{}_{}".format(s, j): wandb.Image(normalize_image(outputs["disp_"+str(s)][j]))},step=self.step)
                                
                    

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)


        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)


    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
            self.models[n].eval()
            for param in self.models[n].parameters():
                param.requires_grad = False

        # loading adam state
        # optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        # if os.path.isfile(optimizer_load_path):
            # print("Loading Adam weights")
            # optimizer_dict = torch.load(optimizer_load_path)
            # self.model_optimizer.load_state_dict(optimizer_dict)
        # else:
        print("Adam is randomly initialized")
    
    def flow2rgb_raw(flow_map, max_value):
        flow_map_np = flow_map.detach().cpu().numpy()
        _, h, w = flow_map_np.shape
        flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
        rgb_map = np.ones((3,h,w)).astype(np.float32)
        if max_value is not None:
            normalized_flow_map = flow_map_np / max_value
        else:
            normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
        rgb_map[0] += normalized_flow_map[0]
        rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
        rgb_map[2] += normalized_flow_map[1]
        return rgb_map.clip(0,1)