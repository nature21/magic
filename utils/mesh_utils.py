from typing import List

import numpy as np
import open3d as o3d
import sapien.core as sapien
import trimesh
from open3d import geometry
from trimesh import Trimesh


def get_actor_meshes(actor: sapien.Entity):
    """Get actor (collision) meshes in the actor frame."""
    meshes = []
    for col_shape in actor.get_collision_shapes():
        geom = col_shape.geometry
        if isinstance(geom, sapien.BoxGeometry):
            mesh = trimesh.creation.box(extents=2 * geom.half_lengths)
        elif isinstance(geom, sapien.CapsuleGeometry):
            mesh = trimesh.creation.capsule(
                height=2 * geom.half_length, radius=geom.radius
            )
        elif isinstance(geom, sapien.SphereGeometry):
            mesh = trimesh.creation.icosphere(radius=geom.radius)
        elif isinstance(geom, sapien.PlaneGeometry):
            continue
        elif isinstance(
            geom, (sapien.ConvexMeshGeometry, sapien.NonconvexMeshGeometry)
        ):
            vertices = geom.vertices  # [n, 3]
            faces = geom.indices.reshape(-1, 3)  # [m * 3]
            vertices = vertices * geom.scale
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        else:
            raise TypeError(type(geom))
        mesh.apply_transform(col_shape.get_local_pose().to_transformation_matrix())
        meshes.append(mesh)
    return meshes


def get_visual_body_meshes(visual_body: sapien.RenderBody):
    meshes = []
    for render_shape in visual_body.get_render_shapes():
        vertices = render_shape.mesh.vertices * visual_body.scale  # [n, 3]
        faces = render_shape.mesh.indices.reshape(-1, 3)  # [m * 3]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.apply_transform(visual_body.local_pose.to_transformation_matrix())
        meshes.append(mesh)
    return meshes


def get_actor_visual_meshes(actor: sapien.ActorBase):
    """Get actor (visual) meshes in the actor frame."""
    meshes = []
    for vb in actor.get_visual_bodies():
        meshes.extend(get_visual_body_meshes(vb))
    return meshes


def merge_meshes(meshes: List[trimesh.Trimesh]):
    n, vs, fs = 0, [], []
    for mesh in meshes:
        v, f = mesh.vertices, mesh.faces
        vs.append(v)
        fs.append(f + n)
        n = n + v.shape[0]
    if n:
        return trimesh.Trimesh(np.vstack(vs), np.vstack(fs))
    else:
        return None


def get_actor_mesh(actor: sapien.ActorBase, to_world_frame=True, visual=False):
    if visual:
        mesh = merge_meshes(get_actor_visual_meshes(actor))
    else:
        mesh = merge_meshes(get_actor_meshes(actor))
    if mesh is None:
        return None
    if to_world_frame:
        T = actor.pose.to_transformation_matrix()
        mesh.apply_transform(T)
    return mesh


def get_actor_visual_mesh(actor: sapien.ActorBase):
    mesh = merge_meshes(get_actor_visual_meshes(actor))
    if mesh is None:
        return None
    return mesh


def get_articulation_meshes(
    articulation: sapien.ArticulationBase, exclude_link_names=(), visual=False
):
    """Get link meshes in the world frame."""
    meshes = []
    for link in articulation.get_links():
        if link.name in exclude_link_names:
            continue
        mesh = get_actor_mesh(link, True, visual=visual)
        if mesh is None:
            continue
        meshes.append(mesh)
    return meshes


def trimesh_to_open3d_mesh(trimesh_mesh: Trimesh) -> geometry.TriangleMesh:
    open3d_mesh = geometry.TriangleMesh()
    open3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    open3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)

    # Check and transfer vertex normals, if they exist
    if trimesh_mesh.vertex_normals.size > 0:
        open3d_mesh.vertex_normals = o3d.utility.Vector3dVector(trimesh_mesh.vertex_normals)

    # Check and transfer vertex colors, if they exist
    # Trimesh stores colors in the 'visual' attribute
    if hasattr(trimesh_mesh.visual, 'vertex_colors') and trimesh_mesh.visual.vertex_colors.size > 0:
        # Open3D expects colors in the range [0, 1], but Trimesh uses [0, 255]
        vertex_colors_normalized = trimesh_mesh.visual.vertex_colors[:, :3] / 255.0
        open3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_normalized)

    return open3d_mesh


def open3d_mesh_to_trimesh(open3d_mesh: geometry.TriangleMesh) -> Trimesh:
    # Ensure the input is an Open3D mesh
    if not isinstance(open3d_mesh, geometry.TriangleMesh):
        raise TypeError("Input must be an Open3D TriangleMesh")

    # Convert Open3D mesh to Trimesh by extracting vertices and faces
    vertices = np.asarray(open3d_mesh.vertices)
    faces = np.asarray(open3d_mesh.triangles)

    # Create a Trimesh object using the extracted vertices and faces
    trimesh_mesh = Trimesh(vertices=vertices, faces=faces)

    return trimesh_mesh


def get_normal_of_nearest_point(pcd, query_point, return_point=False):
    if not pcd.has_normals():
        raise ValueError("The point cloud must have normals.")

    # Convert query point to a numpy array
    query_point = np.array(query_point)

    # Use KDTree to find the nearest neighbor
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    [_, idx, _] = kdtree.search_knn_vector_3d(query_point, 1)  # Search for the 1 nearest neighbor

    nearest_normal = np.asarray(pcd.normals)[idx[0]]
    if return_point:
        nearest_point = np.asarray(pcd.points)[idx[0]]
        return nearest_normal, nearest_point
    return nearest_normal
