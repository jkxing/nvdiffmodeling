
import torch
import cv2
import numpy as np
from torch.autograd import Function
from torch.nn import Module
from util import transform_pos
import spot
import nvdiffrast.torch as dr

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
class GatherPoints(Function):
    @staticmethod
    def forward(ctx, points, uvw, tri_id, num_per_view, vertex_num):
        ctx.save_for_backward(tri_id, num_per_view, uvw, torch.tensor(vertex_num,device = tri_id.device))
        return points

    @staticmethod
    def backward(ctx, grad_output):
        tri_id, num_per_view, uvw, vertex_num= ctx.saved_tensors
        vertex_num = vertex_num.item()
        num_per_view = tuple(num_per_view.tolist())
        tri_id_view = torch.split(tri_id, num_per_view)
        grad_output_view = torch.split(grad_output, num_per_view)
        uvw_view = torch.split(uvw, num_per_view)
        num = 0
        for i in range(len(tri_id_view)):
            if num_per_view[i]==0:
                continue
            uvw_view_i = uvw_view[i].reshape(-1)
            index_i = tri_id_view[i].reshape(-1).to(torch.int64)
            x = torch.zeros((vertex_num),device = grad_output.device)
            x.scatter_(src=uvw_view_i, dim=0, index=index_i, reduce='add')
            weight = x[index_i].reshape(num_per_view[i],-1)+1e-6
            grad_output[num:num+num_per_view[i],...]/=weight.detach()[...,None]
            num+=num_per_view[i]
        return grad_output, None, None, None, None


class PointRenderer(Module):
    def __init__(self):
        super().__init__()
        self.gp = GatherPoints()
   
    def forward(self, points, uvw, tri_id, num_per_view, vertex_num):
        points = self.gp.apply(points, uvw, tri_id, num_per_view, vertex_num)
        pos_3d = points.permute(2,0,1)*uvw
        pos_3d = pos_3d.permute(1,2,0)
        pos_2d = torch.sum(pos_3d[...,:2],dim = 1)
        return pos_2d

class OldPointRenderer(Function):  
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def render_point_pytorch(self, image_fragments, mesh):
        pix_to_face = image_fragments.pix_to_face.detach()[:,:,:,0:1]
        bary_coords = image_fragments.bary_coords[:,:,:,0:1,:]
        faces_list = mesh.faces_list()[0]
        verts_list = image_fragments.proj_mesh.verts_list()[0][None,...]
        h,w = pix_to_face.shape[1:3]
        msk = (pix_to_face[...,0]!=-1).detach()
        info = pix_to_face[msk].detach()
        tri_id = faces_list[info[...,0].long()].detach()
        points = verts_list[:,tri_id.long(),:]
        uvw = bary_coords[msk].detach().permute(1,0,2).squeeze(0)
        has_pos = torch.zeros((*pix_to_face.shape[:-1]),device = pix_to_face.device)
        has_pos[msk] = 1
        pos_3d = points.permute(0,3,1,2)*uvw
        pos_3d = pos_3d.permute(0,2,3,1)
        pos_2d = -torch.sum(pos_3d[...,:2],dim = 2)
        if self.debug:
            pos_2d_img = torch.zeros((*pix_to_face.shape[:-1],2),device = pix_to_face.device)
            pos_2d_img[msk] = pos_2d
            debug_img = torch.zeros((*pix_to_face.shape[:-1],3),device = pix_to_face.device)
            debug_img[...,:2] = (pos_2d_img+1)/2
            cv2.imshow("debug_img",debug_img[0].detach().cpu().numpy())
        return has_pos, uvw, tri_id, pos_2d

    def point_renderer_pytorch(self, pointlist, uvw, image_fragments):
        verts_list = image_fragments.proj_mesh.verts_list()[0][None,...]
        points = verts_list[:,pointlist.long(),:]
        pos_3d = points.permute(0,3,1,2)*uvw
        pos_3d = pos_3d.permute(0,2,3,1)
        pos_2d = -torch.sum(pos_3d[...,:2],dim = 2)
        return pos_2d
    
    def match2(self, target, source):
        source = source.detach().cpu().numpy()[0]
        target = target.detach().cpu().numpy()[0]
        siz = min(1000,min(source.shape[0],target.shape[0]))
        src_idx = np.random.choice(np.arange(source.shape[0]), size=siz, replace=False, p=None)
        tar_idx = np.random.choice(np.arange(target.shape[0]), size=siz, replace=False, p=None)
        src = source[src_idx]
        tar = target[tar_idx]
        cost_matrix = cdist(src,  tar, 'sqeuclidean')
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        source[src_idx] = tar[col_idx]
        return torch.from_numpy(source)[None,...],siz

    def match1(self, gt, pre):
        pre = pre.detach().cpu().numpy()[0]
        gt = gt.detach().cpu().numpy()[0]
        #pre[...,:3]=0
        #gt[...,:3]=0
        res = spot.spot(pre, gt)
        #res[...,:3]=pre[...,:3]
        return torch.from_numpy(res)[None,...]

    def match(self, gt, pre, reso):
        h,w = gt.shape[:2]
        source = np.zeros((h,w,5),dtype = np.float32)
        target = np.zeros((h,w,5),dtype = np.float32)
        source[...,:3]=pre.detach().cpu().numpy()
        target[...,:3]=gt.detach().cpu().numpy()
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        pos = np.array(np.meshgrid(x, y))
        pos = np.transpose(pos,(1,2,0))
        source[...,3:] = pos
        target[...,3:] = pos
        source = source.reshape(-1,5)
        target = target.reshape(-1,5)
        res = spot.spot(source, target)
        res = res.reshape(h,w,5)
        #if True:
        #    self.visualize_match(res, pre, gt, reso)
        return torch.from_numpy(res)[None,...]
    #def match(self, color,color_opt,resolution):
    #    img_gt = torch.clone(color).reshape((resolution,resolution, 3))
    #    img_pre = torch.clone(color_opt).reshape((resolution,resolution, 3))
    #    with torch.no_grad():
    #        #matching_res = matching.estimate_flow(img_pre.permute(2,0,1).unsqueeze(0)*255,img_gt.permute(2,0,1).unsqueeze(0)*255, mode='channel_first')
    #        matching_res = self.network.estimate_flow(img_gt.permute(2,0,1).unsqueeze(0)*255,img_pre.permute(2,0,1).unsqueeze(0)*255, mode='channel_first')
    #    matching_res = matching_res.squeeze().permute(1,2,0) 
    #    if True:
    #        self.visualize_match(matching_res, img_gt.detach().cpu().numpy(), img_pre.detach().cpu().numpy(), resolution)
    #    return matching_res[None,...] / resolution * 2
    
    def render_point(self, glctx, mtx, pos, pos_idx, resolution: int):
        pos_clip    = transform_pos(mtx, pos)
        rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
        msk = (rast_out[...,3]!=0).detach()
        info = rast_out[rast_out[...,3]!=0].detach()
        uvw = torch.cat((info[...,0:1],info[...,1:2],1-info[...,0:1]-info[...,1:2]),axis = 1).detach()
        tri_id = pos_idx[info[...,3].long()-1].detach()
        has_pos = torch.zeros((*rast_out.shape[:-1]),device = rast_out.device)
        has_pos[msk] = 1
        return has_pos, uvw, tri_id

    def point_renderer_nvdiffrast(self, glctx, mtx, pointlist, uvw, pos):
        pos_clip = transform_pos(mtx, pos)
        points = pos_clip[:,pointlist.long(),:]
        points = points[...,:3]/points[...,3:4].detach()
        pos_3d = points.permute(0,3,1,2)*uvw
        pos_3d = pos_3d.permute(0,2,3,1)
        pos_2d = torch.sum(pos_3d[...,:2],dim = 2)
        return pos_2d
        
    def visualize_match(self, matching_res,img_pre_cpu, img_gt_cpu, resolution):
        disp_x = matching_res[...,0]
        disp_y = matching_res[...,1]
        print(disp_y)
        grad = -disp_x*(disp_x<0)
        cv2.imshow("grad", grad)
        cv2.waitKey(0)
        return
        h_scale, w_scale=disp_x.shape[:2]
        X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                        np.linspace(0, h_scale - 1, h_scale))
        map_x = (X+disp_x).astype(np.float32)
        map_y = (Y+disp_y).astype(np.float32)
        remapped_image = cv2.remap(img_pre_cpu, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[0.5,0.5,0.5])
        bord = np.ones((resolution,10,3))
        img = np.hstack([img_gt_cpu,bord,remapped_image,bord,img_pre_cpu])
        cv2.imshow("remap",img)

    