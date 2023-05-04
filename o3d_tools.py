from distutils.command.sdist import sdist
import numpy as np
import open3d as o3d
from pathlib import Path
import json

# from https://github.com/facebookresearch/pifuhd 
def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr

def read_obj_to_o3d(obj_path):
    ''' read obj file and return open3d mesh object '''
    with open(obj_path, 'r') as f:
        lines = f.readlines()
    vertices = []
    faces = []
    normals = []
    for line in lines:
        if line.startswith('v '):
            vertex_str = line.strip().split()
            vertex = [float(vertex_str[1]), float(vertex_str[2]), float(vertex_str[3])]
            vertices.append(vertex)
        if line.startswith('vn'):
            normal_str = line.strip().split()
            normal = [float(normal_str[1]), float(normal_str[2]), float(normal_str[3])]
            normals.append(normal)
        if line.startswith('f '):
            face_str = line.strip().split()
            face = [int(face_str[1].split('/')[0]) - 1,
                    int(face_str[2].split('/')[0]) - 1,
                    int(face_str[3].split('/')[0]) - 1]
            faces.append(face)
    vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
    faces = o3d.utility.Vector3iVector(np.asarray(faces))
    o3d_mesh = o3d.geometry.TriangleMesh()

    o3d_mesh.vertices = vertices
    o3d_mesh.triangles = faces
    if len(normals) != 0:
        o3d_mesh.compute_vertex_normals()
        o3d_mesh.vertex_normals.clear()
        for normal in normals:
            o3d_mesh.vertex_normals.append(np.asarray(normal))
        # o3d_mesh.normalize_normals()
    
    return o3d_mesh

class Visualizer():
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
    
    def run(self):
        self.vis.run()

    def create_window(self):
        self.vis.create_window()
        self.vis.get_render_option().load_from_json("./fit/tools/renderoption.json")
        # self.view_from_json('./fit/tools/view.json')
        # self.vis.get_render_option().point_size = 10
        # self.vis.get_render_option().line_width = 10  # not working

    def view_from_json(self, json_file):
        ctr = self.vis.get_view_control()
        with open(json_file) as f:
            view = json.load(f)['trajectory'][0]
        ctr.change_field_of_view(view['field_of_view'])
        ctr.set_front(view['front'])
        ctr.set_lookat(view['lookat'])
        ctr.set_up(view['up'])
        ctr.set_zoom(view['zoom'])    

    def destroy_window(self):
        self.vis.destroy_window()
    
    def rotate_view(self, step=10.0):
        ctr = self.vis.get_view_control()
        ctr.rotate(step, 0.0)

    def add_mesh(self, mesh: o3d.geometry.TriangleMesh, transparent=False):
        if transparent:
            mesh = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            mesh.colors = o3d.utility.Vector3dVector(np.repeat(np.array([[0.5, 0.5, 0.5]]), len(mesh.lines), axis=0))

        self.vis.add_geometry(mesh)
    
    def add_mesh_with_verts_and_faces(self,
                                      verts: np.array,   # [N_v, 3] array
                                      faces: np.array,  # [N_f, 3] array
                                      transparent=False):
        vertices = o3d.utility.Vector3dVector(verts)
        triangles = o3d.utility.Vector3iVector(faces)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = vertices
        mesh.triangles = triangles
        mesh.compute_vertex_normals()

        if transparent:
            mesh = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)

        self.vis.add_geometry(mesh)

    def add_3d_skeleton(self, joints, joint_type='openpose', color='r'):
        '''
            joints: N x 3 numpy array
        '''
        color_dict = {'r': np.array([1.0, 0, 0]), 'g': np.array([0, 1.0, 0]), 'b': np.array([0, 0, 1.0])}
        if joint_type=='openpose':
            # joint pairs for drawing skeleton line
            joint_pairs = np.array([[15, 17], [15, 0], [0, 16], [16, 18], [0, 1], [1, 2], [2, 3], [3, 4],
                [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [10, 11], [11, 24], [11, 22], [22, 23],
                [8, 12], [12, 13], [13, 14], [14, 21], [14, 19], [19, 20]])

        elif joint_type=='smpl':
            joint_pairs = np.array([[15, 12], [12, 9], [9, 14], [14, 17], [17, 19], [19, 21], [21, 23],
                [9, 13], [13, 16], [16, 18], [18, 20], [20, 22], [9, 6], [6, 3], [3, 0], [0, 2], [2, 5], [5, 8],
                [8, 11], [0, 1], [1, 4], [4, 7], [7, 10], [15, 24], [24, 25], [24, 26], [25, 27], [26, 28]])
        else:
            raise NotImplementedError

        skeleton = o3d.geometry.LineSet()
        skeleton.points = o3d.utility.Vector3dVector(joints)
        skeleton.lines = o3d.utility.Vector2iVector(joint_pairs)

        joints_pcd = o3d.geometry.PointCloud()
        joints_pcd.points = o3d.utility.Vector3dVector(joints)
        
        
        if joint_type == 'openpose':
            joint_colors_25 =[[255, 0, 85], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], 
                    [170, 255, 0], [85, 255, 0], [0, 255, 0], [255, 0, 0], [0, 255, 85], 
                    [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], 
                    [255, 0, 170], [170, 0, 255], [255, 0, 255], [85, 0, 255], [0, 0, 255], 
                    [0, 0, 255], [0, 0, 255], [0, 255, 255], [0, 255, 255], [0, 255, 255]]
            skeleton.colors = o3d.utility.Vector3dVector(np.array(joint_colors_25) / 255.0)
            joints_pcd.colors = o3d.utility.Vector3dVector(np.array(joint_colors_25) / 255.0)
        else:
            skeleton.colors = o3d.utility.Vector3dVector(np.repeat(color_dict[color][None], len(joint_pairs), axis=0))
            joints_pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.array(color_dict[color][None]), len(joints), axis=0))

        self.vis.add_geometry(skeleton)
        self.vis.add_geometry(joints_pcd)

        return skeleton, joints_pcd
        
    def update_3d_skeleton(self, skeleton, joints_pcd, new_joints, joint_type='openpose'):
        '''
            skeleton: o3d.geometry.LineSet
            joints_pcd: o3d.geometry.pcd
            new_joints: N x 3 numpy array
        '''

        if joint_type=='openpose':
            # joint pairs for drawing skeleton line
            joint_pairs = np.array([[15, 17], [15, 0], [0, 16], [16, 18], [0, 1], [1, 2], [2, 3], [3, 4],
                [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [10, 11], [11, 24], [11, 22], [22, 23],
                [8, 12], [12, 13], [13, 14], [14, 21], [14, 19], [19, 20]])

        elif joint_type=='smpl':
            joint_pairs = np.array([[15, 12], [12, 9], [9, 14], [14, 17], [17, 19], [19, 21], [21, 23],
                [9, 13], [13, 16], [16, 18], [18, 20], [20, 22], [9, 6], [6, 3], [3, 0], [0, 2], [2, 5], [5, 8],
                [8, 11], [0, 1], [1, 4], [4, 7], [7, 10], [15, 24], [24, 25], [24, 26], [25, 27], [26, 28]])
        else:
            raise NotImplementedError

        skeleton.points = o3d.utility.Vector3dVector(new_joints)
        skeleton.lines = o3d.utility.Vector2iVector(joint_pairs)

        joints_pcd.points = o3d.utility.Vector3dVector(new_joints)

        self.vis.update_geometry(skeleton)
        self.vis.update_geometry(joints_pcd)