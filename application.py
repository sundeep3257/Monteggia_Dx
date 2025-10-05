##tempcomments
import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import cv2
import csv
import math
import segmentation_models_pytorch as smp
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from scipy.ndimage import center_of_mass
from scipy.spatial import distance
import tempfile
import shutil
import logging
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, session
from werkzeug.utils import secure_filename
from PIL import Image
import matplotlib.pyplot as plt
import timm
import torch.nn as nn

# --- 1. FLASK APP CONFIGURATION ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULTS_FOLDER = os.path.join('static', 'results')
ALLOWED_EXTENSIONS = {'nii', 'png'}

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
application.config['RESULTS_FOLDER'] = RESULTS_FOLDER
application.config['SECRET_KEY'] = 'supersecretkey' # Needed for flashing messages
application.logger.setLevel(logging.INFO)

# --- 2. MODEL AND CROP PARAMETERS ---
MODEL_INPUT_SIZE = (224, 224)
CROP_SIZE = 244

# --- 3. VERIFY MODEL WEIGHTS PATHS ---
# Assumes a 'models' folder is in the same directory as application.py
MODELS_DIR = 'models'
MODEL_WEIGHTS = {
    "lat_small_cap": os.path.join(MODELS_DIR, "Lat_Small_Cap_Model_EFB1.pth"),
    "lat_big_cap": os.path.join(MODELS_DIR, "Lat_Big_Cap_Model_EFB1.pth"),
    "lat_rad_head": os.path.join(MODELS_DIR, "Lat_Small_RadHead_Model_EFB1.pth"),
    "lat_big_rad": os.path.join(MODELS_DIR, "Lat_Big_RadHead_Model_EFB1.pth"),
    "ap_small_cap": os.path.join(MODELS_DIR, "AP_Small_Cap_Model_EFB1.pth"),
    "ap_big_cap": os.path.join(MODELS_DIR, "AP_Big_Cap_Model_EFB1.pth"),
    "ap_rad_head": os.path.join(MODELS_DIR, "AP_Small_RadHead_Model_EFB1.pth"),
    "ap_big_rad": os.path.join(MODELS_DIR, "AP_Big_RadHead_Model_EFB1.pth"),
    "ulnar_classification": os.path.join(MODELS_DIR, "Detect_Ulnar_Fx.pth"),
    "ulnar_segmentation": os.path.join(MODELS_DIR, "Segment_Ulnar_Fx.pth")
}

# --- Check if model files exist ---
for name, path in MODEL_WEIGHTS.items():
    if not os.path.exists(path):
        print(f"FATAL ERROR: Model file not found at {path}. Please ensure the 'models' directory is set up correctly.")
        exit()

###################################################################################################
#                                      HELPER FUNCTIONS (Your Original Code)                      #
###################################################################################################

def hex_to_bgr(hex_color):
    """Converts a hexadecimal color string (e.g., '#RRGGBBFF') to an OpenCV BGR tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))

def detect_line_circle_intersection(radial_quants_csv_path, cap_quants_csv_path):
    """
    Determines if the radiocapitellar line intersects with the capitellum circle.
    Returns True if they intersect, False otherwise.
    """
    try:
        df_rad = pd.read_csv(radial_quants_csv_path)
        df_cap = pd.read_csv(cap_quants_csv_path)
        
        # Get the radiocapitellar line (from midpoints)
        midpoints_df = df_rad[df_rad['Point_Type'].str.contains('Midpoint')]
        if len(midpoints_df) < 2:
            application.logger.warning("Not enough midpoints found for line detection")
            return False
        
        # Line points
        p1_x, p1_y = midpoints_df['Global_X_row'].iloc[0], midpoints_df['Global_Y_col'].iloc[0]
        p2_x, p2_y = midpoints_df['Global_X_row'].iloc[1], midpoints_df['Global_Y_col'].iloc[1]
        
        # Circle center and radius
        center_x = df_cap['Centroid_X_row'].iloc[0]
        center_y = df_cap['Centroid_Y_col'].iloc[0]
        diameter = df_cap['Diameter_Length'].iloc[0]
        
        if np.isnan(diameter) or diameter <= 0:
            application.logger.warning("Invalid capitellum diameter")
            return False
        
        radius = diameter / 2
        
        # Calculate distance from line to circle center
        # Using the formula for distance from point to line
        A = p2_y - p1_y
        B = p1_x - p2_x
        C = p2_x * p1_y - p1_x * p2_y
        
        distance_to_center = abs(A * center_x + B * center_y + C) / np.sqrt(A**2 + B**2)
        
        # Check if line intersects circle
        intersects = distance_to_center <= radius
        
        application.logger.info(f"Line-circle intersection detection: distance={distance_to_center:.2f}, radius={radius:.2f}, intersects={intersects}")
        return intersects
        
    except Exception as e:
        application.logger.error(f"Error in line-circle intersection detection: {e}")
        return False

def create_annotated_image(original_image_path, radial_quants_csv_path, cap_quants_csv_path, output_annotated_path):
    """Loads the original image and plots coordinates, a bisecting line, and a diameter circle."""
    nii_original = nib.load(original_image_path)
    img_data = nii_original.get_fdata().squeeze().T
    img_normalized = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    annotated_image = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)

    COLOR_RAD_VERTEX = hex_to_bgr("#d55e00ff")
    COLOR_RAD_MIDPOINT_AND_LINE = hex_to_bgr("#25524aff")
    COLOR_CAP_CENTROID = hex_to_bgr("#56b4e9ff")

    df_rad = pd.read_csv(radial_quants_csv_path)
    midpoints_df = df_rad[df_rad['Point_Type'].str.contains('Midpoint')]

    if len(midpoints_df) == 2:
        p1_col, p1_row = midpoints_df['Global_X_row'].iloc[0], midpoints_df['Global_Y_col'].iloc[0]
        p2_col, p2_row = midpoints_df['Global_X_row'].iloc[1], midpoints_df['Global_Y_col'].iloc[1]
        dx, dy = p2_col - p1_col, p2_row - p1_row
        start_point = (int(p1_col - dx * 1000), int(p1_row - dy * 1000))
        end_point = (int(p1_col + dx * 1000), int(p1_row + dy * 1000))
        cv2.line(annotated_image, start_point, end_point, color=COLOR_RAD_MIDPOINT_AND_LINE, thickness=2)

    radial_point_radius = 6
    for _, row in df_rad.iterrows():
        center_point = (int(row['Global_X_row']), int(row['Global_Y_col']))
        point_color = COLOR_RAD_MIDPOINT_AND_LINE if 'Midpoint' in row['Point_Type'] else COLOR_RAD_VERTEX
        cv2.circle(annotated_image, center_point, radius=radial_point_radius, color=point_color, thickness=-1)

    df_cap = pd.read_csv(cap_quants_csv_path)
    centroid_point = (int(df_cap['Centroid_X_row'].iloc[0]), int(df_cap['Centroid_Y_col'].iloc[0]))
    cv2.circle(annotated_image, centroid_point, radius=8, color=COLOR_CAP_CENTROID, thickness=-1)

    diameter_length = df_cap['Diameter_Length'].iloc[0]
    if not np.isnan(diameter_length) and diameter_length > 0:
        circle_radius = int(round(diameter_length / 2))
        cv2.circle(annotated_image, centroid_point, radius=circle_radius, color=COLOR_RAD_MIDPOINT_AND_LINE, thickness=2)

    cv2.imwrite(output_annotated_path, annotated_image)
    application.logger.info(f"Successfully created annotated image: {output_annotated_path}")


def predict_mask(model, mri_path, device, original_shape_hw):
    transform_to_model_size = T.Resize(MODEL_INPUT_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True)
    nifti_img = nib.load(mri_path)
    img_data = nifti_img.get_fdata().astype(np.float32).squeeze()
    min_val, max_val = np.min(img_data), np.max(img_data)
    img_normalized = (img_data - min_val) / (max_val - min_val) if (max_val - min_val) > 1e-8 else np.zeros_like(img_data)
    img_tensor = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0)
    img_transformed = transform_to_model_size(img_tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_transformed)
        probs = torch.sigmoid(logits)
        mask_model_size = (probs > 0.5).squeeze().cpu()
    resize_back = T.Resize(original_shape_hw, interpolation=InterpolationMode.NEAREST)
    mask_original_size = resize_back(mask_model_size.unsqueeze(0).float()).squeeze().numpy().astype(np.uint8)
    return mask_original_size, nifti_img.affine, nifti_img.header


def get_center_of_mass(mask_data, output_csv_path):
    if np.sum(mask_data) == 0:
        raise ValueError("The segmentation model produced an empty mask. Cannot calculate center of mass.")
    center_y_row, center_x_col = center_of_mass(mask_data)
    center_x_col, center_y_row = int(round(center_x_col)), int(round(center_y_row))
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Center_X_col', 'Center_Y_row'])
        writer.writerow([center_x_col, center_y_row])
    return center_x_col, center_y_row


def calculate_crop_offsets(img_shape_hw, center_x_col, center_y_row):
    img_height, img_width = img_shape_hw
    half_crop = CROP_SIZE // 2
    y_start_row, x_start_col = max(center_y_row - half_crop, 0), max(center_x_col - half_crop, 0)
    y_end_row, x_end_col = min(y_start_row + CROP_SIZE, img_height), min(x_start_col + CROP_SIZE, img_width)
    y_start_row, x_start_col = y_end_row - CROP_SIZE, x_end_col - CROP_SIZE
    return int(y_start_row), int(x_start_col)


def crop_image_around_center(nii_path, center_x_col, center_y_row, output_path):
    nii_img = nib.load(nii_path)
    img_data = nii_img.get_fdata().squeeze()
    y_start, x_start = calculate_crop_offsets(img_data.shape, center_x_col, center_y_row)
    cropped_img = img_data[y_start: y_start + CROP_SIZE, x_start: x_start + CROP_SIZE]
    cropped_nii = nib.Nifti1Image(np.expand_dims(cropped_img, axis=-1), affine=nii_img.affine)
    nib.save(cropped_nii, output_path)

def quantify_capitellum(mask_path, center_csv_path, original_shape_hw, output_csv_path):
    nifti_img, center_df = nib.load(mask_path), pd.read_csv(center_csv_path)
    image_data = nifti_img.get_fdata().squeeze()
    crop_center_x_col, crop_center_y_row = center_df['Center_X_col'].iloc[0], center_df['Center_Y_row'].iloc[0]
    y_offset, x_offset = calculate_crop_offsets(original_shape_hw, crop_center_x_col, crop_center_y_row)
    binary_mask = (image_data > 0).astype(np.uint8)
    if np.sum(binary_mask) == 0: raise ValueError("Capitellum mask is empty after cropping.")
    cy_local_row, cx_local_col = center_of_mass(binary_mask)
    contours, _ = cv2.findContours(binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError("No contours found in the capitellum mask.")
    main_contour = max(contours, key=cv2.contourArea)
    contour_points = main_contour.reshape(-1, 2)
    max_distance, diameter_pt1_local, diameter_pt2_local = 0, None, None
    if len(contour_points) >= 2:
        for i in range(len(contour_points)):
            for j in range(i + 1, len(contour_points)):
                dist = distance.euclidean(contour_points[i], contour_points[j])
                if dist > max_distance:
                    max_distance, diameter_pt1_local, diameter_pt2_local = dist, contour_points[i], contour_points[j]
    output_data = {
        'Centroid_Y_col': [cx_local_col + x_offset], 'Centroid_X_row': [cy_local_row + y_offset],
        'Diameter_Start_Y_col': [diameter_pt1_local[0] + x_offset if diameter_pt1_local is not None else np.nan],
        'Diameter_Start_X_row': [diameter_pt1_local[1] + y_offset if diameter_pt1_local is not None else np.nan],
        'Diameter_End_Y_col': [diameter_pt2_local[0] + x_offset if diameter_pt2_local is not None else np.nan],
        'Diameter_End_X_row': [diameter_pt2_local[1] + y_offset if diameter_pt2_local is not None else np.nan],
        'Diameter_Length': [max_distance]
    }
    pd.DataFrame(output_data).to_csv(output_csv_path, index=False)


def project_point_onto_segment(point, seg_start, seg_end):
    p, a, b = np.array(point, dtype=float), np.array(seg_start, dtype=float), np.array(seg_end, dtype=float)
    ap, ab = p - a, b - a
    ab_squared_len = np.dot(ab, ab)
    if ab_squared_len == 0.0: return tuple(a.astype(int))
    t = max(0, min(1, np.dot(ap, ab) / ab_squared_len))
    return tuple((a + t * ab).astype(int))


def quantify_radius(mask_path, center_csv_path, original_shape_hw, output_csv_path):
    nii_image = nib.load(mask_path)
    img_data = np.squeeze(nii_image.get_fdata())
    if np.sum(img_data) == 0: raise ValueError("Radial head mask is empty.")
    img_normalized = (255 * (img_data > 0)).astype(np.uint8)
    contours, _ = cv2.findContours(img_normalized.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError("No contours found in the radial head mask.")
    main_contour = max(contours, key=cv2.contourArea)
    if len(main_contour) < 5: raise ValueError("Not enough contour points to analyze radial head shape.")
    hull_indices = cv2.convexHull(main_contour, returnPoints=False)
    defects = cv2.convexityDefects(main_contour, hull_indices)
    if defects is None: raise ValueError("Could not find convexity defects.")
    max_dist, p1 = 0, None
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        if d / 256.0 > max_dist: max_dist, p1 = d / 256.0, tuple(main_contour[f][0])
    if p1 is None: raise ValueError("Could not identify the primary point (p1) in radial head analysis.")
    rect = cv2.minAreaRect(main_contour)
    box = np.intp(cv2.boxPoints(rect))
    sides = [(box[i], box[(i + 1) % 4]) for i in range(4)]
    distances_to_sides = [np.linalg.norm(np.array(p1) - np.array(project_point_onto_segment(p1, s, e))) for s, e in sides]
    opposite_side = sides[(np.argmin(distances_to_sides) + 2) % 4]
    target_point = project_point_onto_segment(p1, opposite_side[0], opposite_side[1])
    min_dist_to_target, p_opposite = float('inf'), None
    for point in main_contour:
        pt = tuple(point[0])
        dist = math.dist(pt, target_point)
        if dist < min_dist_to_target: min_dist_to_target, p_opposite = dist, pt
    if p_opposite is None: raise ValueError("Could not identify opposite point in radial head analysis.")
    p1_idx, p_opposite_idx = np.argmin(np.sum(np.abs(main_contour - p1), axis=2)), np.argmin(np.sum(np.abs(main_contour - p_opposite), axis=2))
    i1, i2 = sorted([p1_idx, p_opposite_idx])
    arc1, arc2 = main_contour[i1:i2 + 1], np.concatenate((main_contour[i2:], main_contour[:i1 + 1]))
    image_center = (img_normalized.shape[1] / 2, img_normalized.shape[0] / 2)
    dist1_to_center = math.dist(np.mean(arc1.reshape(-1, 2), axis=0), image_center)
    dist2_to_center = math.dist(np.mean(arc2.reshape(-1, 2), axis=0), image_center)
    tip_contour = arc1 if dist1_to_center < dist2_to_center else arc2
    tip_polygon_vertices = None
    for p in np.linspace(0.001, 0.2, 1000):
        approx = cv2.approxPolyDP(tip_contour, p * cv2.arcLength(tip_contour, closed=False), closed=False)
        if len(approx) == 5: tip_polygon_vertices = approx; break
    if tip_polygon_vertices is None: raise ValueError("Could not approximate a 5-sided polygon for the radial head.")
    v = [tuple(p[0]) for p in tip_polygon_vertices]
    start_idx, end_idx = v.index(min(v, key=lambda p_find: math.dist(p_find, p1))), v.index(min(v, key=lambda p_find: math.dist(p_find, p_opposite)))
    ordered_v, v_circ, i = [], v + v, start_idx
    while True:
        ordered_v.append(v_circ[i])
        if v_circ[i] == v[end_idx]: break
        i += 1
    if len(ordered_v) != 5:
        ordered_v, i = [], start_idx
        while True:
            ordered_v.append(v_circ[i])
            if v_circ[i] == v[end_idx]: break
            i = (i - 1 + len(v)) % len(v)
    final_vertex_list = ordered_v if len(ordered_v) == 5 else v
    v1, v2, v3, v4, v5 = final_vertex_list
    midpoint_1, midpoint_2 = ((v1[0] + v5[0]) / 2, (v1[1] + v5[1]) / 2), ((v2[0] + v4[0]) / 2, (v2[1] + v4[1]) / 2)
    all_coords_local_xy = np.array([v1, v2, v3, v4, v5, midpoint_1, midpoint_2], dtype=float)
    center_df = pd.read_csv(center_csv_path)
    center_x_col, center_y_row = center_df['Center_X_col'].iloc[0], center_df['Center_Y_row'].iloc[0]
    y_offset, x_offset = calculate_crop_offsets(original_shape_hw, center_x_col, center_y_row)
    transformed_coords_xy = all_coords_local_xy.copy()
    transformed_coords_xy[:, 0], transformed_coords_xy[:, 1] = all_coords_local_xy[:, 0] + x_offset, all_coords_local_xy[:, 1] + y_offset
    point_labels = ['Vertex_1', 'Vertex_2', 'Vertex_3', 'Vertex_4', 'Vertex_5', 'Midpoint_V1_V5', 'Midpoint_V2_V4']
    pd.DataFrame({'Point_Type': point_labels, 'Global_Y_col': transformed_coords_xy[:, 0],
                  'Global_X_row': transformed_coords_xy[:, 1]}).to_csv(output_csv_path, index=False)


###################################################################################################
#                                    MAIN PROCESSING FUNCTION                                     #
###################################################################################################

def process_image(input_image_path, annotated_output_path, image_type="lateral"):
    """
    Main function to process a single image and produce the annotated output file.
    This function encapsulates the entire pipeline from the original script.
    image_type: "lateral" or "ap" to determine which models to use
    """
    temp_dir = tempfile.mkdtemp()
    application.logger.info(f"Created temporary directory: {temp_dir}")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        application.logger.info(f"Using device: {device}")
        
        application.logger.info(f"Loading segmentation models for {image_type} images...")
        models = {}
        model_prefix = "lat" if image_type == "lateral" else "ap"
        
        # Load only the models needed for this image type
        model_keys = [f"{model_prefix}_small_cap", f"{model_prefix}_big_cap", 
                     f"{model_prefix}_rad_head", f"{model_prefix}_big_rad"]
        
        for name in model_keys:
            if name in MODEL_WEIGHTS:
                model = smp.UnetPlusPlus("efficientnet-b1", encoder_weights=None, in_channels=1, classes=1).to(device)
                model.load_state_dict(torch.load(MODEL_WEIGHTS[name], map_location=device, weights_only=False))
                model.eval()
                models[name] = model
        application.logger.info(f"All {image_type} models loaded successfully.")
        
        base_name = os.path.basename(input_image_path).split('.nii')[0]
        
        # Define paths for all temporary and final output files
        paths = {
            "input_image": input_image_path,
            "rad_quants": os.path.join(temp_dir, f"{base_name}_rad_quants.csv"),
            "cap_quants": os.path.join(temp_dir, f"{base_name}_cap_quants.csv"),
            "small_cap_mask": os.path.join(temp_dir, f"{base_name}_small_cap_mask.nii.gz"),
            "small_cap_center": os.path.join(temp_dir, f"{base_name}_small_cap_center.csv"),
            "cropped_cap_input": os.path.join(temp_dir, f"{base_name}_cropped_cap.nii.gz"),
            "big_cap_mask": os.path.join(temp_dir, f"{base_name}_big_cap_mask.nii.gz"),
            "rad_head_mask": os.path.join(temp_dir, f"{base_name}_rad_head_mask.nii.gz"),
            "rad_head_center": os.path.join(temp_dir, f"{base_name}_rad_head_center.csv"),
            "cropped_rad_input": os.path.join(temp_dir, f"{base_name}_cropped_rad.nii.gz"),
            "big_rad_mask": os.path.join(temp_dir, f"{base_name}_big_rad_mask.nii.gz"),
        }
        
        original_nii = nib.load(paths["input_image"])
        original_shape_hw = original_nii.shape[:2]

        # --- Capitellum Pipeline ---
        application.logger.info("Starting capitellum segmentation...")
        small_cap_key = f"{model_prefix}_small_cap"
        big_cap_key = f"{model_prefix}_big_cap"
        small_cap_mask_data, affine, header = predict_mask(models[small_cap_key], paths["input_image"], device, original_shape_hw)
        nib.save(nib.Nifti1Image(np.expand_dims(small_cap_mask_data, axis=-1), affine, header), paths["small_cap_mask"])
        cap_center_x, cap_center_y = get_center_of_mass(small_cap_mask_data, paths["small_cap_center"])
        crop_image_around_center(paths["input_image"], cap_center_x, cap_center_y, paths["cropped_cap_input"])
        big_cap_mask_data, _, _ = predict_mask(models[big_cap_key], paths["cropped_cap_input"], device, (CROP_SIZE, CROP_SIZE))
        nib.save(nib.Nifti1Image(np.expand_dims(big_cap_mask_data, axis=-1), affine, header), paths["big_cap_mask"])
        quantify_capitellum(paths["big_cap_mask"], paths["small_cap_center"], original_shape_hw, paths["cap_quants"])
        application.logger.info("Capitellum quantification complete.")

        # --- Radial Head Pipeline ---
        application.logger.info("Starting radial head segmentation...")
        rad_head_key = f"{model_prefix}_rad_head"
        big_rad_key = f"{model_prefix}_big_rad"
        rad_head_mask_data, _, _ = predict_mask(models[rad_head_key], paths["input_image"], device, original_shape_hw)
        nib.save(nib.Nifti1Image(np.expand_dims(rad_head_mask_data, axis=-1), affine, header), paths["rad_head_mask"])
        rad_center_x, rad_center_y = get_center_of_mass(rad_head_mask_data, paths["rad_head_center"])
        crop_image_around_center(paths["input_image"], rad_center_x, rad_center_y, paths["cropped_rad_input"])
        big_rad_mask_data, _, _ = predict_mask(models[big_rad_key], paths["cropped_rad_input"], device, (CROP_SIZE, CROP_SIZE))
        nib.save(nib.Nifti1Image(np.expand_dims(big_rad_mask_data, axis=-1), affine, header), paths["big_rad_mask"])
        quantify_radius(paths["big_rad_mask"], paths["rad_head_center"], original_shape_hw, paths["rad_quants"])
        application.logger.info("Radial head quantification complete.")

        # --- Final Annotation Step ---
        application.logger.info("Creating final annotated image...")
        create_annotated_image(
            original_image_path=paths["input_image"],
            radial_quants_csv_path=paths["rad_quants"],
            cap_quants_csv_path=paths["cap_quants"],
            output_annotated_path=annotated_output_path
        )
        
        # --- Intersection Detection Step ---
        application.logger.info("Detecting line-circle intersection...")
        intersects = detect_line_circle_intersection(paths["rad_quants"], paths["cap_quants"])
        
        application.logger.info("✅ Pipeline finished successfully!")
        return True, bool(intersects) # Success with intersection result (convert to Python bool)

    except Exception as e:
        application.logger.error(f"An error occurred during processing: {e}", exc_info=True)
        return False, str(e) # Failure
    
    finally:
        shutil.rmtree(temp_dir)
        application.logger.info(f"Cleaned up temporary directory: {temp_dir}")


###################################################################################################
#                                        FLASK ROUTES                                             #
###################################################################################################

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_png_to_nifti(png_path, nii_path):
    """
    Converts a PNG image to NIfTI format.
    """
    try:
        # Load PNG image and convert to grayscale
        img = Image.open(png_path).convert("L")  # convert to grayscale (1 channel)
        img_array = np.array(img, dtype=np.float32)
        
        # Transpose the array
        img_array = np.transpose(img_array)
        
        # Create NIfTI image with identity affine matrix
        nii_img = nib.Nifti1Image(img_array, affine=np.eye(4))
        
        # Save as NIfTI
        nib.save(nii_img, nii_path)
        
        application.logger.info(f"Successfully converted PNG to NIfTI: {nii_path}")
        return True, None
        
    except Exception as e:
        application.logger.error(f"Error converting PNG to NIfTI: {e}")
        return False, str(e)

def create_2slice_nifti(lateral_path, ap_path, output_path):
    """
    Creates a 2-slice NIfTI from separate lateral and AP images.
    First slice is lateral, second slice is AP.
    If images have different dimensions, they will be resized to match.
    """
    try:
        # Load lateral image
        if lateral_path.lower().endswith('.png'):
            lat_img = Image.open(lateral_path).convert("L")
            lat_array = np.array(lat_img, dtype=np.float32)
            lat_array = np.transpose(lat_array)
        else:
            lat_nii = nib.load(lateral_path)
            lat_array = lat_nii.get_fdata().squeeze().T
        
        # Load AP image
        if ap_path.lower().endswith('.png'):
            ap_img = Image.open(ap_path).convert("L")
            ap_array = np.array(ap_img, dtype=np.float32)
            ap_array = np.transpose(ap_array)
        else:
            ap_nii = nib.load(ap_path)
            ap_array = ap_nii.get_fdata().squeeze().T
        
        # Check if images have different dimensions
        if lat_array.shape != ap_array.shape:
            application.logger.info(f"Images have different dimensions: lateral {lat_array.shape}, AP {ap_array.shape}")
            application.logger.info("Resizing images to match dimensions...")
            
            # Determine target size (use the larger dimensions)
            target_height = max(lat_array.shape[0], ap_array.shape[0])
            target_width = max(lat_array.shape[1], ap_array.shape[1])
            
            # Resize lateral image
            if lat_array.shape != (target_height, target_width):
                lat_resized = cv2.resize(lat_array, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                application.logger.info(f"Resized lateral image from {lat_array.shape} to {lat_resized.shape}")
            else:
                lat_resized = lat_array
            
            # Resize AP image
            if ap_array.shape != (target_height, target_width):
                ap_resized = cv2.resize(ap_array, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                application.logger.info(f"Resized AP image from {ap_array.shape} to {ap_resized.shape}")
            else:
                ap_resized = ap_array
            
            # Use resized arrays
            lat_array = lat_resized
            ap_array = ap_resized
        
        # Stack the images along the third dimension (slices)
        combined_array = np.stack([lat_array, ap_array], axis=2)
        
        # Create NIfTI image
        nii_img = nib.Nifti1Image(combined_array, affine=np.eye(4))
        nib.save(nii_img, output_path)
        
        application.logger.info(f"Successfully created 2-slice NIfTI: {output_path}")
        return True, None
        
    except Exception as e:
        application.logger.error(f"Error creating 2-slice NIfTI: {e}")
        return False, str(e)

# --- Ulnar Fracture Detection Models ---
class EfficientNetB1_2Channel(nn.Module):
    """ Model for classifying an image as 'Healthy' or 'Fracture'. """
    def __init__(self, num_classes=2):
        super(EfficientNetB1_2Channel, self).__init__()
        self.model = timm.create_model("efficientnet_b1", pretrained=False, in_chans=2, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

class EffUNet(nn.Module):
    """ Model for generating a pixel-wise segmentation mask of a fracture. """
    def __init__(self):
        super(EffUNet, self).__init__()
        self.backbone = timm.create_model('efficientnet_b1', pretrained=True, in_chans=2, num_classes=0, global_pool='')
        num_features = self.backbone.num_features  # 1280 for efficientnet_b1

        self.decoder = nn.Sequential(
            nn.Conv2d(num_features, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 2, kernel_size=1)  # Output 2 channels for AP/Lat masks
        )

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        seg_logits = self.decoder(feats)
        # Resize output to match input image size before the model
        seg_logits = T.functional.resize(seg_logits, size=x.shape[-2:], antialias=True)
        return seg_logits

def run_ulnar_fracture_analysis(image_path, output_path, device):
    """
    Runs the combined ulnar fracture analysis pipeline.
    """
    try:
        # Load models
        cls_model = EfficientNetB1_2Channel().to(device)
        cls_model.load_state_dict(torch.load(MODEL_WEIGHTS["ulnar_classification"], map_location=device, weights_only=False))
        cls_model.eval()
        
        seg_model = EffUNet().to(device)
        seg_model.load_state_dict(torch.load(MODEL_WEIGHTS["ulnar_segmentation"], map_location=device, weights_only=False))
        seg_model.eval()
        
        application.logger.info("Ulnar fracture models loaded successfully.")
        
        # Load and preprocess image
        nii_image = nib.load(image_path)
        image_data = nii_image.get_fdata().astype(np.float32)
        
        # Store original data for high-resolution visualization
        original_ap = image_data[:, :, 0]
        original_lat = image_data[:, :, 1]
        
        # Normalize image data to [0, 1]
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data) + 1e-5)
        
        # Stack views and convert to tensor
        image_tensor = torch.tensor(np.stack([image_data[:, :, 0], image_data[:, :, 1]], axis=0))
        
        # Define resize transform and prepare tensor for models
        transform = T.Resize(size=(240, 240), antialias=True)
        input_tensor = transform(image_tensor).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            # Classification
            cls_outputs = cls_model(input_tensor)
            cls_probs = torch.softmax(cls_outputs, dim=1).cpu().numpy()[0]
            pred_class_idx = np.argmax(cls_probs)
            
            # Only run segmentation if fracture is detected
            if pred_class_idx == 1:  # Fracture detected
                application.logger.info("Fracture detected, running segmentation...")
                seg_logits = seg_model(input_tensor)
                seg_probs = torch.sigmoid(seg_logits).squeeze(0).cpu()
                
                # Generate visualization with segmentation
                resized_mask_ap = T.functional.resize(seg_probs[0:1], original_ap.shape, antialias=True).squeeze().numpy()
                resized_mask_lat = T.functional.resize(seg_probs[1:2], original_lat.shape, antialias=True).squeeze().numpy()
                
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                fig.suptitle("Ulnar Fracture Segmentation Heatmap", fontsize=16)
                
                # AP View
                axs[0].imshow(original_ap.T, cmap="gray", origin="lower")
                axs[0].imshow(resized_mask_ap.T, cmap="hot", alpha=0.3, origin="lower")
                axs[0].set_title("Lateral View with Predicted Fracture Mask")
                axs[0].axis('off')
                
                # Lateral View
                axs[1].imshow(original_lat.T, cmap="gray", origin="lower")
                axs[1].imshow(resized_mask_lat.T, cmap="hot", alpha=0.3, origin="lower")
                axs[1].set_title("AP View with Predicted Fracture Mask")
                axs[1].axis('off')
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
                plt.close(fig)
            else:
                application.logger.info("No fracture detected, skipping segmentation...")
                # Generate visualization without segmentation
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                fig.suptitle("Ulnar Fracture Analysis - No Fracture Detected", fontsize=16)
                
                # AP View
                axs[0].imshow(original_ap.T, cmap="gray", origin="lower")
                axs[0].set_title("Lateral View - No Fracture Detected")
                axs[0].axis('off')
                
                # Lateral View
                axs[1].imshow(original_lat.T, cmap="gray", origin="lower")
                axs[1].set_title("AP View - No Fracture Detected")
                axs[1].axis('off')
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
                plt.close(fig)
        
        # Return results
        labels = {0: "Healthy", 1: "Fracture"}
        prediction = labels[pred_class_idx]
        healthy_prob = float(cls_probs[0])
        fracture_prob = float(cls_probs[1])
        
        application.logger.info(f"Ulnar fracture analysis complete: {prediction}")
        return True, {
            'prediction': prediction,
            'healthy_prob': healthy_prob,
            'fracture_prob': fracture_prob
        }
        
    except Exception as e:
        application.logger.error(f"Error in ulnar fracture analysis: {e}")
        return False, str(e)

@application.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Create directories if they don't exist
        os.makedirs(application.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(application.config['RESULTS_FOLDER'], exist_ok=True)
        
        results = {}
        errors = []
        
        # Check for lateral image upload
        lateral_file = request.files.get('lateral_file')
        if lateral_file and lateral_file.filename and allowed_file(lateral_file.filename):
            filename = secure_filename(lateral_file.filename)
            input_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
            lateral_file.save(input_path)
            
            # Check if it's a PNG file and convert to NIfTI if needed
            if filename.lower().endswith('.png'):
                nii_filename = filename.replace('.png', '.nii')
                nii_path = os.path.join(application.config['UPLOAD_FOLDER'], nii_filename)
                convert_success, convert_error = convert_png_to_nifti(input_path, nii_path)
                
                if not convert_success:
                    errors.append(f"Lateral PNG conversion failed: {convert_error}")
                    if os.path.exists(input_path):
                        os.remove(input_path)
                else:
                    # Use the converted NIfTI file for processing
                    processing_input_path = nii_path
                    # Clean up the original PNG
                    if os.path.exists(input_path):
                        os.remove(input_path)
                    
                    # Process the converted NIfTI file
                    output_filename = f"lateral_{filename.replace('.png', '')}.png"
                    output_path = os.path.join(application.config['RESULTS_FOLDER'], output_filename)
                    
                success, result = process_image(processing_input_path, output_path, "lateral")
                
                # Clean up the processing input file
                if os.path.exists(processing_input_path):
                    os.remove(processing_input_path)
                
                if success:
                    results['lateral'] = {
                        'filename': output_filename,
                        'intersects': bool(result)
                    }
                else:
                    errors.append(f"Lateral image processing failed: {result}")
            else:
                # Process NIfTI file directly
                output_filename = f"lateral_{filename.replace('.nii', '')}.png"
                output_path = os.path.join(application.config['RESULTS_FOLDER'], output_filename)
                
                success, result = process_image(input_path, output_path, "lateral")
                
                # Clean up the processing input file
                if os.path.exists(input_path):
                    os.remove(input_path)
                
                if success:
                    results['lateral'] = {
                        'filename': output_filename,
                        'intersects': bool(result)
                    }
                else:
                    errors.append(f"Lateral image processing failed: {result}")
        
        # Check for AP image upload
        ap_file = request.files.get('ap_file')
        if ap_file and ap_file.filename and allowed_file(ap_file.filename):
            filename = secure_filename(ap_file.filename)
            input_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
            ap_file.save(input_path)
            
            # Check if it's a PNG file and convert to NIfTI if needed
            if filename.lower().endswith('.png'):
                nii_filename = filename.replace('.png', '.nii')
                nii_path = os.path.join(application.config['UPLOAD_FOLDER'], nii_filename)
                convert_success, convert_error = convert_png_to_nifti(input_path, nii_path)
                
                if not convert_success:
                    errors.append(f"AP PNG conversion failed: {convert_error}")
                    if os.path.exists(input_path):
                        os.remove(input_path)
                else:
                    # Use the converted NIfTI file for processing
                    processing_input_path = nii_path
                    # Clean up the original PNG
                    if os.path.exists(input_path):
                        os.remove(input_path)
                    
                    # Process the converted NIfTI file
                    output_filename = f"ap_{filename.replace('.png', '')}.png"
                    output_path = os.path.join(application.config['RESULTS_FOLDER'], output_filename)
                    
                    success, result = process_image(processing_input_path, output_path, "ap")
                    
                    # Clean up the processing input file
                    if os.path.exists(processing_input_path):
                        os.remove(processing_input_path)
                    
                    if success:
                        results['ap'] = {
                            'filename': output_filename,
                            'intersects': bool(result)
                        }
                    else:
                        errors.append(f"AP image processing failed: {result}")
            else:
                # Process NIfTI file directly
                output_filename = f"ap_{filename.replace('.nii', '')}.png"
                output_path = os.path.join(application.config['RESULTS_FOLDER'], output_filename)
                
                success, result = process_image(input_path, output_path, "ap")
                
                # Clean up the processing input file
                if os.path.exists(input_path):
                    os.remove(input_path)
                
                if success:
                    results['ap'] = {
                        'filename': output_filename,
                        'intersects': bool(result)
                    }
                else:
                    errors.append(f"AP image processing failed: {result}")
        
        # Check if we have any successful results
        if results:
            if errors:
                for error in errors:
                    flash(error)
            # Store results in session
            session['results'] = results
            return redirect(url_for('show_results'))
        else:
            if not errors:
                flash('Please upload at least one .nii or .png file to analyze.')
            else:
                for error in errors:
                    flash(error)
            return redirect(request.url)

    return render_template('index.html')

@application.route('/ulnar', methods=['GET', 'POST'])
def ulnar_upload():
    if request.method == 'POST':
        # Create directories if they don't exist
        os.makedirs(application.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(application.config['RESULTS_FOLDER'], exist_ok=True)
        
        # Check for both lateral and AP files
        lateral_file = request.files.get('lateral_file')
        ap_file = request.files.get('ap_file')
        
        if not lateral_file or not lateral_file.filename:
            flash('Lateral forearm X-ray is required.')
            return redirect(request.url)
        
        if not ap_file or not ap_file.filename:
            flash('AP forearm X-ray is required.')
            return redirect(request.url)
        
        if not (allowed_file(lateral_file.filename) and allowed_file(ap_file.filename)):
            flash('Invalid file types. Please upload .nii or .png files.')
            return redirect(request.url)
        
        try:
            # Save uploaded files
            lateral_filename = secure_filename(lateral_file.filename)
            ap_filename = secure_filename(ap_file.filename)
            
            lateral_path = os.path.join(application.config['UPLOAD_FOLDER'], lateral_filename)
            ap_path = os.path.join(application.config['UPLOAD_FOLDER'], ap_filename)
            
            lateral_file.save(lateral_path)
            ap_file.save(ap_path)
            
            # Create 2-slice NIfTI
            combined_filename = f"ulnar_combined_{lateral_filename.split('.')[0]}_{ap_filename.split('.')[0]}.nii"
            combined_path = os.path.join(application.config['UPLOAD_FOLDER'], combined_filename)
            
            success, error = create_2slice_nifti(lateral_path, ap_path, combined_path)
            if not success:
                flash(f"Error creating combined image: {error}")
                return redirect(request.url)
            
            # Run ulnar fracture analysis
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            output_filename = f"ulnar_analysis_{lateral_filename.split('.')[0]}_{ap_filename.split('.')[0]}.png"
            output_path = os.path.join(application.config['RESULTS_FOLDER'], output_filename)
            
            success, result = run_ulnar_fracture_analysis(combined_path, output_path, device)
            
            # Clean up uploaded files
            for path in [lateral_path, ap_path, combined_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            if success:
                # Store results in session
                results = {
                    'filename': output_filename,
                    'prediction': result['prediction'],
                    'healthy_prob': result['healthy_prob'],
                    'fracture_prob': result['fracture_prob']
                }
                session['ulnar_results'] = results
                return redirect(url_for('ulnar_results'))
            else:
                flash(f"Analysis failed: {result}")
                return redirect(request.url)
                
        except Exception as e:
            application.logger.error(f"Error in ulnar upload processing: {e}")
            flash(f"An error occurred: {str(e)}")
            return redirect(request.url)
    
    return render_template('ulnar_index.html')

@application.route('/ulnar/results')
def ulnar_results():
    """Displays the ulnar fracture analysis results."""
    results = session.get('ulnar_results', {})
    return render_template('ulnar_results.html', results=results)

@application.route('/download_test_images')
def download_test_images():
    """Downloads the test images as a zip file."""
    import zipfile
    import io
    
    try:
        # Create a zip file in memory
        memory_file = io.BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Get the path to the sample files folder
            sample_files_path = os.path.join('static', 'Web_Sample_Files')
            
            if os.path.exists(sample_files_path):
                # Walk through the directory and add all files to the zip
                for root, dirs, files in os.walk(sample_files_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Add file to zip with relative path
                        arcname = os.path.relpath(file_path, sample_files_path)
                        zip_file.write(file_path, arcname)
                
                application.logger.info(f"Created zip file with {len(zip_file.namelist())} files")
            else:
                application.logger.error("Sample files directory not found")
                flash("Test images not available.")
                return redirect(url_for('upload_file'))
        
        # Prepare the response
        memory_file.seek(0)
        
        from flask import Response
        return Response(
            memory_file.getvalue(),
            mimetype='application/zip',
            headers={
                'Content-Disposition': 'attachment; filename=Monteggia_Test_Images.zip'
            }
        )
        
    except Exception as e:
        application.logger.error(f"Error creating test images zip: {e}")
        flash("Error preparing test images.")
        return redirect(url_for('upload_file'))

@application.route('/results')
def show_results():
    """Displays the final annotated images to the user."""
    results = session.get('results', {})
    return render_template('results.html', results=results)

# This allows serving the result images directly
@application.route('/static/results/<filename>')
def send_result_file(filename):
    return send_from_directory(application.config['RESULTS_FOLDER'], filename)


if __name__ == '__main__':

    application.run(debug=True)





