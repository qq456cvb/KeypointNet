import open3d as o3d
import json
import numpy as np
import seaborn as sns


def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float)
    colors = np.array(data[:, -1], dtype=np.int)
    colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
    return pc, colors
    
    
if __name__ == "__main__":
    class_id = '03001627'
    model_id = '88382b877be91b2a572f8e1c1caad99e'
    
    labels = json.load(open('annotations/chair.json'))
    label = [label for label in labels if label['model_id'] == model_id][0]
    
    pc, colors = naive_read_pcd('pcds/{}/{}.pcd'.format(class_id, model_id))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.)
    
    palette = sns.color_palette("bright", 20)  # create color palette
    
    # draw pcd and keypoints
    mesh_spheres = []
    for kp in label['keypoints']:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        mesh_sphere.translate(pc[kp['pcd_info']['point_index']])
        mesh_sphere.paint_uniform_color(palette[kp['semantic_id']])
        mesh_spheres.append(mesh_sphere)
    
    print('visualizing point cloud with keypoints highlighted')
    o3d.visualization.draw_geometries([pcd, *mesh_spheres])
    
    # draw mesh and keypoints
    textured_mesh = o3d.io.read_triangle_mesh('models/{}/{}/models/model_normalized.obj'.format(class_id, model_id))
    textured_mesh.compute_vertex_normals()
    vertices = np.array(textured_mesh.vertices)
    faces = np.array(textured_mesh.triangles)
    
    mesh_spheres = []
    for kp in label['keypoints']:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        face_coords = vertices[faces[kp['mesh_info']['face_index']]]
        mesh_sphere.translate(face_coords.T @ kp['mesh_info']['face_uv'])
        mesh_sphere.paint_uniform_color(palette[kp['semantic_id']])
        mesh_spheres.append(mesh_sphere)
        
    print('visualizing mesh with keypoints highlighted')
    o3d.visualization.draw_geometries([textured_mesh, *mesh_spheres])
    
    # draw ply and keypoints
    textured_mesh = o3d.io.read_triangle_mesh('models/{}/{}/models/model_normalized.ply'.format(class_id, model_id))
    textured_mesh.compute_vertex_normals()
    vertices = np.array(textured_mesh.vertices)
    faces = np.array(textured_mesh.triangles)
    
    mesh_spheres = []
    for kp in label['keypoints']:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        face_coords = vertices[faces[kp['mesh_info']['face_index']]]
        mesh_sphere.translate(face_coords.T @ kp['mesh_info']['face_uv'])
        mesh_sphere.paint_uniform_color(palette[kp['semantic_id']])
        mesh_spheres.append(mesh_sphere)
        
    print('visualizing ply mesh with keypoints highlighted')
    o3d.visualization.draw_geometries([textured_mesh, *mesh_spheres])