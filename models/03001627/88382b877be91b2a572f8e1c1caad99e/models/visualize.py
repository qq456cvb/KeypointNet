import open3d as o3d
import numpy as np

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("model_normalized.ply")
    mesh = o3d.io.read_triangle_mesh("model_normalized.obj")
    
    print(np.array(pcd.points).shape, np.array(mesh.vertices).shape)
    print(np.allclose(np.array(pcd.points), np.array(mesh.vertices)))
    o3d.visualization.draw_geometries([pcd, mesh])