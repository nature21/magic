import os

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from magic.utils_2d import calculate_transformation, apply_transformation


def visualize_local(local_edge_points: np.ndarray, radius_of_region: float, radius_of_curvature: float,
                    save_path: str = None, human_view: bool = False):
    fig, ax = plt.subplots()
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.scatter(local_edge_points[:, 0], local_edge_points[:, 1], color='red', s=1)  # the edge points
    x_model = np.linspace(min(local_edge_points[:, 0]), max(local_edge_points[:, 0]), 300)
    a = 1 / (2 * radius_of_curvature)
    y_model = a * (x_model ** 2)
    plt.plot(x_model, y_model, color='gray', label=f'Model: y = {a:.2f}x²')  # the parabola
    circle = Circle((0, radius_of_curvature), radius_of_curvature, edgecolor='blue', facecolor='none', linewidth=2)
    ax.add_patch(circle)  # the circle
    plt.title(
        f'local visualization, radius of region = {radius_of_region}, radius of curvature = {int(radius_of_curvature)}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path)
    else:
        if human_view:
            plt.show()

    plt.close()


def visualize_global(
        edge_points: np.ndarray,
        region_points: np.ndarray,
        center_of_contact: np.ndarray,
        center_of_curvature: np.ndarray,
        radius_of_curvature: float,
        view_point: np.ndarray = None,
        save_path: str = None,
        human_view: bool = False
):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.scatter(edge_points[:, 0], edge_points[:, 1], color='red', s=1)
    plt.scatter(region_points[:, 0], region_points[:, 1], color='orange', s=2)
    plt.scatter(*center_of_contact, color='blue', s=10)
    plt.plot([center_of_contact[0], center_of_curvature[0]], [center_of_contact[1], center_of_curvature[1]],
             color='blue')
    circle = Circle(
        (center_of_curvature[0], center_of_curvature[1]),
        radius_of_curvature, edgecolor='blue', facecolor='none', linewidth=2)
    ax.add_patch(circle)

    if view_point is not None:
        plt.scatter(*view_point, color='black', s=10)

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path)
    else:
        if human_view:
            plt.show()

    plt.close()


def visualize_alignment(
        vec1: np.ndarray,
        vec2: np.ndarray,
        edge_points_1: np.ndarray,
        edge_points_2: np.ndarray,
        region_points_1: np.ndarray,
        region_points_2: np.ndarray,
        no_scaling: bool = False,
        force_alignment: bool = False,
        point_1: np.ndarray = None,
        point_2: np.ndarray = None,
        save_path: str = None,
        human_view: bool = False
):
    scale, rotation_matrix = calculate_transformation(vec2, vec1)
    if no_scaling:
        scale = 1
    edge_points_2_transformed = apply_transformation(scale, rotation_matrix,
                                                     edge_points_2 - vec2[0]) + vec1[0]
    region_points_2_transformed = apply_transformation(scale, rotation_matrix,
                                                       region_points_2 - vec2[0]) + vec1[0]

    if force_alignment:
        point_2_transformed = apply_transformation(scale, rotation_matrix, point_2.reshape(1, 2) - vec2[0])[0] + vec1[0]
        offset = point_1 - point_2_transformed
        edge_points_2_transformed += offset
        region_points_2_transformed += offset

    fig, ax = plt.subplots()
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.scatter(edge_points_2_transformed[:, 0], edge_points_2_transformed[:, 1], color='red', s=1)
    plt.scatter(region_points_2_transformed[:, 0], region_points_2_transformed[:, 1], color='orange', s=2)

    plt.scatter(edge_points_1[:, 0], edge_points_1[:, 1], color='blue', s=1)
    plt.scatter(region_points_1[:, 0], region_points_1[:, 1], color='green', s=2)

    plt.scatter(vec1[0][0], vec1[0][1], color='red', marker='*', s=20)

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path)
    else:
        if human_view:
            plt.show()

    plt.close()


def create_column(images, name):
    max_height = max(img.height for img in images)
    # Create a new image with the width of the widest image and the height of the tallest image
    column_image = Image.new('RGB', (max(img.width for img in images), max_height * len(images) + 20), color='white')
    draw = ImageDraw.Draw(column_image)
    font = ImageFont.truetype("arial.ttf", 70)
    # Paste each image into the column
    y_offset = 20
    # Get maximum height among all images
    # Paste each image into the column
    for img in images:
        column_image.paste(img, (0, y_offset))
        y_offset += img.height
    draw.text((275, 5), name, fill='red', font=font, align='center')
    return column_image


def merge_images_to_column(outdir, save_path='results/temp', num_rows=1, start_index=0, end_index=10):
    # Get all directories within the output directory
    def custom_sort(x):
        try:
            # Try converting to integer
            return int(x)
        except ValueError:
            # If conversion fails, return the original string
            return x

    directories = [str(dir_) for dir_ in sorted(os.listdir(outdir), key=custom_sort)][start_index:end_index]
    directories = directories[:5] + directories[6:]
    layer1_directories = [os.path.join(outdir, d) for d in directories]

    # Iterate through each directory
    column_images = []
    for i in range(len(layer1_directories)):
        # Get all image files within the directory
        image_files = []
        for j in range(3):
            image_files.append(os.path.join(layer1_directories[i], f'{j}', 'alignment.png'))
        images = []
        for image_file in image_files:
            # Open each image file
            img = Image.open(image_file)
            images.append(img)
        # Create a column for the images in this directory
        column_image = create_column(images, layer1_directories[i].split('/')[-1])
        column_images.append(column_image)

    # Concatenate all column images horizontally
    final_image = Image.new(
        'RGB',
        (sum(img.width for img in column_images) // num_rows, max(img.height for img in column_images) * num_rows)
    )
    x_offset = 0
    for index, column_image in enumerate(column_images):
        y_offset = (index % num_rows) * final_image.height // num_rows
        final_image.paste(column_image, (x_offset, y_offset))
        if index % num_rows == num_rows - 1:
            x_offset += column_image.width

    # Save the final image
    final_image.save(os.path.join(save_path, "merged_image.png"))


def present_result(result_path, save_path='results/temp', num_rows=1, start_index=0, end_index=10):
    merge_images_to_column(result_path, save_path, num_rows, start_index, end_index)
