import random
import numpy as np
import math
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import shutil
import imageio.v2 as imageio
import os
import argparse
from tqdm import tqdm

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def construct_mesh(fn):
    """
    Constructs the mesh so it can be added into the Open3D scene.

    Args:
        fn (str): Filename of the mesh.

    Returns:
        open3d.geometry.TriangleMesh: Open3D mesh constructed from the source file.
    """
    
    mesh = o3d.io.read_triangle_mesh(fn)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    R = mesh.get_rotation_matrix_from_xyz((-(np.pi / 2), 0, 0))
    mesh.rotate(R, center=(0, 0, 0))
    
    return mesh
    

def create_snapshots(fns, count=12):
    """
    Creates the snapshots for all meshes in "fns" with use of Open3D
    library.

    Args:
        fns (list): Filenames of the meshes.
        count (int, optional): Number of snapshots to be taken. Defaults to 12.
    """
    
    # Create the basic material
    material = rendering.MaterialRecord()
    material.base_color = [.5, .5, .5, 1]
    material.shader = "defaultLit"

    # Setup scene app and window
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window(f"window", 500, 500)
    window.show(False)
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene_widget)
    bbox = o3d.geometry.AxisAlignedBoundingBox([-2000, -2000, -2000], [2000, 2000, 2000])

    # Setup camera
    fov = 60
    fov_rad = fov * (np.pi / 180)
    scene_widget.setup_camera(fov, bbox, [0, 0, 0])
    
    # Generate "count" snapshots for all of the files
    for i in tqdm(range(len(fns))):
        fn = fns[i]
        
        # print(f'Generating snapshots for: {fn}:')
        
        # Folder path for snapshots of each mesh
        folder_path = f'{fn[:-4]}_snapshots'
        
        # Add mesh to the scene
        mesh = construct_mesh(fn)
        mesh_max = np.abs(np.asarray(mesh.vertices)).max()
        # print(mesh_max)
        
        # Move camera so the object is visible
        # eye_x = (mesh_max / 46) * 100
        eye_x = abs((mesh_max * 2) / math.sin(fov_rad / 2))
        center = [0, 0, 0]
        eye = [eye_x, 0, 0]
        up = [0, 1, 0]
        scene_widget.scene.camera.look_at(center, eye, up)
        
        # Setup light so it shines from same point as camera does (simulate MeshLab behaviour - TODO check with others)
        scene_widget.scene.scene.enable_sun_light(False)
        scene_widget.scene.scene.remove_light('light')
        scene_widget.scene.scene.add_directional_light('light', [1,1,1], [-eye[0], -eye[1], -eye[2]], (1e6 / 5), False)
        
        # Generate each snapshot
        for i in range(12):
            snapshot_path = f'{folder_path}/{str(i)}.png'
            
            # Rotate the mesh - do it in two steps so it can be easily undone
            R = mesh.get_rotation_matrix_from_xyz((0, (np.pi / 6) * i, 0))
            mesh.rotate(R, center=(0,0,0))
            
            R = mesh.get_rotation_matrix_from_xyz((0, 0, -(np.pi / 6)))
            mesh.rotate(R, center=(0,0,0))

            # Update the geometry - yes this is the only way
            scene_widget.scene.clear_geometry()
            scene_widget.scene.add_geometry('mesh', mesh, material)
            
            # Save the snapshot
            snapshot = gui.Application.instance.render_to_image(scene_widget.scene, 256, 256)
            o3d.io.write_image(snapshot_path, snapshot)
            
            # Undo the rotations - for easier rotation
            R = mesh.get_rotation_matrix_from_xyz((0, -(np.pi / 6) * i, (np.pi / 6)))
            mesh.rotate(R, center=(0,0,0))
            
            # gui.Application.instance.run()
            # break
            
        
        # Remove the mesh, so we don't have to recreate the scene
        scene_widget.scene.remove_geometry('mesh')
        
        # Break the loop for only one mesh
        # break
    
    # gui.Application.instance.quit()


def create_snapshot_folders(fns):
    """
    Creates the folders for snapshots.

    Args:
        fns (list): Filenames of meshes.

    Returns:
        list: Foldernames of the created folders.
    """
    
    folders = []
    
    for fn in fns:
        folder_path = f'{fn[:-4]}_snapshots'
        folders.append(folder_path)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    return folders

def remove_snapshot_folders(dataset_path="../dataset/ModelNet10"):
    """
    Removes all of the snapshot folders.

    Args:
        dataset_path (str, optional): Path to dataset. Defaults to "../dataset/ModelNet10".
    """
    
    for root, dirs, files in os.walk(dataset_path):
        for dir in dirs:
            if 'snapshots' in dir:
                shutil.rmtree(os.path.join(root, dir))


def get_obj_names(dataset_path="../dataset/ModelNet10", category="all"):
    """
    Gets all the mesh filenames.

    Args:
        dataset_path (str, optional): Path to dataset. Defaults to "../dataset/ModelNet10".
        category (str, optional): Specifies which category to process. Defaults to "all".

    Returns:
        _type_: _description_
    """
    
    fns = []
    
    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            if category == "all" or category in filename:
                if '.off' in filename:
                    fns.append(os.path.join(root, filename))
    
    fns.sort()
    return fns

def randomize_downsample(fns, count):
    """
    Randomly chooses meshes for each category.

    Args:
        fns (list): Filenames from which the random meshes are chosen.
        count (int): Number of meshes chosen per category.

    Returns:
        list: Filenames for processing.
    """
    
    randomized_fns = []
    categories = [fn.split('/')[-3] + '/' + fn.split('/')[-2] for fn in fns]
    categories = sorted(list(set(categories)))
    
    for cat in categories:
        cat_fns = [fn for fn in fns if cat in fn]
        random_fns = random.sample(cat_fns, count)
        randomized_fns += random_fns
    
    random.shuffle(randomized_fns)
    return randomized_fns

def make_gifs(dataset_path="../dataset/ModelNet10", output_folder='../dataset/ModelNet10/a_gifs'):
    """
    Makes gifs out of the generated snapshots.

    Args:
        dataset_path (str, optional): Path to the dataset folder. Defaults to "../dataset/ModelNet10".
        output_folder (str, optional): Output folder where the gifs will be saved. Defaults to '../dataset/ModelNet10/a_gifs'.
    """
    
    folders = []
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for root, dirs, files in os.walk(dataset_path):
        for dir in dirs:
            if 'snapshots' in dir:
                folders.append(os.path.join(root, dir))
                                     
    
    for folder in folders:
        print(f'Generating gif for {folder}...')
        imgs = []
        gif_name = folder.split('/')[-1]
        for root, dirs, files in os.walk(folder):
            files_ints = [int(file[:-4]) for file in files]
            files_ints.sort()
            for file_int in files_ints:
                filename = f'{str(file_int)}.png'
                if ".png" in filename:
                    imgs.append(imageio.imread(os.path.join(root, filename)))
            if len(imgs) > 0:
                imageio.mimsave(os.path.join(output_folder, f'{gif_name}.gif'), imgs, format='GIF', fps=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the dataset meshes into 12 snapshots')
    parser.add_argument('-g', action='store_true', default=False, help='Create gifs from the snapshots.')
    parser.add_argument('-r', action='store_true', default=False, help='Remove snapshot folders.')
    parser.add_argument('-ds_path', action='store', type=str, default='../dataset/ModelNet10', help='Dataset path.')
    parser.add_argument('-d', action='store', type=int, default=-1, help='Downsample the dataset.')
    
    args = parser.parse_args()
    fns = get_obj_names(args.ds_path, category="all")
    
    # Using only subset of the dataset
    if args.d > 0:
        fns = randomize_downsample(fns, args.d)
    
    # Remove all the folders with snapshots
    if args.r:
        remove_snapshot_folders()

    create_snapshot_folders(fns)
    create_snapshots(fns)
    
    # Make gifs out of all the snapshotted model
    if args.g:
        make_gifs(dataset_path=args.ds_path)
    