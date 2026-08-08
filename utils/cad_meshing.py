import os
import cv2
import time
import torch
import trimesh
import mcubes
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from shapely.geometry import GeometryCollection, Polygon, MultiPolygon
from shapely.validation import make_valid
from pyquaternion import Quaternion
from utils import utils
from .utils import add_latent
import OCC.Core.gp as gp
import OCC.Core.BRepPrimAPI as BRepPrimAPI
import OCC.Core.BRepBuilderAPI as BRepBuilderAPI
import OCC.Core.TopoDS as TopoDS
import OCC.Core.BRep as BRep
import OCC.Core.BRepAlgoAPI as BRepAlgoAPI
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.gp import gp_Pnt, gp_Pnt2d, gp_Vec
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.BRepTools import breptools

def create_mesh_mc(
    generator, shape_3d, shape_code, filename, N=256, max_batch=32**3, threshold=0
):
    """
    Create a mesh using the marching cubes algorithm.

    Args:
        generator: The generator of network.
        shape_3d: 3D shape parameters.
        shape_code: Shape code.
        N: Resolution parameter.
        threshold: Marching cubes threshold value.
    """
    start = time.time()
    mesh_filename = filename

    voxel_origin = [0, 0, 0]
    voxel_size = 1

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4) # x y z sdf cell_num

    # Transform the first 3 columns to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # Scale the samples to the voxel size and shift by the voxel origin
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    samples[:, :3] = (samples[:, :3]+0.5)/N-0.5

    num_samples = N**3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        sdf, _, _ = generator(sample_subset.unsqueeze(0), shape_3d, shape_code)
        samples[head : min(head + max_batch, num_samples), 3] = (
            sdf.reshape(-1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print(f"Sampling took: {end - start:.3f} seconds")

    numpy_3d_sdf_tensor = sdf_values.numpy()

    verts, faces = mcubes.marching_cubes(numpy_3d_sdf_tensor, threshold)

    faces = faces[:, [0, 2, 1]]

    mesh_points = verts
    mesh_points = (mesh_points + 0.5) / N - 0.5

    if not os.path.exists(os.path.dirname(mesh_filename)):
        os.makedirs(os.path.dirname(mesh_filename))

    utils.save_obj_data(f"{mesh_filename}.obj", mesh_points, faces)

def get_sketch_list(generator, shape_code, wh):
    """
    Sampling the shape of a 2D sketch from an implicit network.
    """
    B, N = 1, wh * wh
    x1, x2 = np.mgrid[-0.5:0.5:complex(0, wh), -0.5:0.5:complex(0, wh)]
    x1 = torch.from_numpy(x1).float()
    x2 = torch.from_numpy(x2).float()
    sample_points = torch.dstack((x1, x2)).view(-1, 2).unsqueeze(0).cuda()

    shape_code_cuda = shape_code.cuda()
 
    sdfs_2d_list = []
    for i in range(4):
        head = getattr(generator, f'sketch_head_{i}')
        
        # Expand features to match sample points
        feat = shape_code_cuda.unsqueeze(1).repeat(1, N, 1)  # [B, N, 256]
        
        # Use add_latent with the combined features
        latent = add_latent(sample_points, feat).float()
        
        sdfs_2d = head(latent).reshape(B, N, -1).float().squeeze().detach().cpu().unsqueeze(-1).numpy()
        sdfs_2d_list.append(sdfs_2d)

    sample_points = sample_points.detach().cpu().numpy()[0][:, :2] / 1 + 0.5
 
    fill_sk_list = []
    for dis in sdfs_2d_list:
        a = np.hstack((sample_points, dis))
        canvas = np.zeros((wh + 80, wh + 80))
        for i in a:
            canvas[int((i[1]) * wh)][int((i[0]) * wh)] = i[2]
        sk = canvas
        result = sk[:wh, :wh]
        bin_img = (result < -0.01).astype('uint8') * 255
        
        ret, thresh = cv2.threshold(bin_img, 254, 255, 0)
        contours, b = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        my_contour_list = []  
        fill_polygon_list = []
        if len(contours) == 0:
            my_contour_list.append(None)
            fill_sk_list.append(None)
            continue
        
        hir = b[0][..., 3]
        
        for c in contours:
            my_contour_list.append(c[:, 0, :])
            
        polygon_list = trimesh.path.polygons.paths_to_polygons(my_contour_list)
        for polygon in polygon_list:
            if polygon is None:
                continue

            path_2d = trimesh.path.exchange.misc.polygon_to_path(polygon)
            path = trimesh.path.Path2D(path_2d['entities'], path_2d['vertices'])

            max_values = np.max(path_2d['vertices'], axis=0)
            min_values = np.min(path_2d['vertices'], axis=0)
            size = np.linalg.norm(max_values - min_values)
            smooth_value = size
            sm_path = trimesh.path.simplify.simplify_spline(path, smooth=smooth_value, verbose=True)
   
            a, _ = sm_path.triangulate(engine='earcut')
            polygon = trimesh.path.polygons.paths_to_polygons([a])
            if polygon[0] is None:
                continue
            Matrix = np.eye(3)
            Matrix[0, 2] = -wh / 2
            Matrix[1, 2] = -wh / 2
            polygon = trimesh.path.polygons.transform_polygon(polygon[0], Matrix)
            fill_polygon_list.append(polygon)
        if len(fill_polygon_list) == 0:
            fill_sk_list.append(None)
            continue
        fill_sk = fill_polygon_list[0]
        for i in range(1, len(fill_polygon_list)):
            if hir[i] % 2 == 1:
                fill_sk = fill_sk | fill_polygon_list[i]
            else:
                fill_sk = fill_sk - fill_polygon_list[i]
            # Ensure the result is valid
            fill_sk = make_valid(fill_sk)
            # Filter to keep only Polygon or MultiPolygon
            if isinstance(fill_sk, GeometryCollection):
                valid_polys = [geom for geom in fill_sk.geoms if isinstance(geom, (Polygon, MultiPolygon))]
                if valid_polys:
                    fill_sk = valid_polys[0] if len(valid_polys) == 1 else MultiPolygon(valid_polys)
                else:
                    fill_sk = None
        fill_sk_list.append(fill_sk)
    return fill_sk_list

def simplify_points(points, min_distance=1e-6):
    """Remove points that are too close to each other."""
    if len(points) < 3:
        return None  # Not enough points for a valid polygon
    simplified = [points[0]]
    for pt in points[1:]:
        last_pt = simplified[-1]
        dist = np.linalg.norm(np.array([pt[0] - last_pt[0], pt[1] - last_pt[1]]))
        if dist > min_distance:
            simplified.append(pt)
    # Ensure the polygon is closed
    if len(simplified) >= 3 and np.linalg.norm(np.array([simplified[-1][0] - simplified[0][0], simplified[-1][1] - simplified[0][1]])) > min_distance:
        simplified.append(simplified[0])
    return simplified if len(simplified) >= 3 else None

def create_cylinder(polygon, builder, compound, height, wh):
    """
    Extruding 3D cylinders from 2D contours.
    If it's an internal contour, subtract it from the whole;
    If it's an external contour, add it to the whole.
    """
    for ring in [polygon.exterior, *polygon.interiors]:
        coords = np.array(ring.coords[:-1])  # Exclude the last point (repeated for closure)
        simplified_coords = simplify_points(coords, min_distance=1e-4)  # Adjust min_distance as needed
        if simplified_coords is None:
            continue  # Skip invalid rings

        points = [gp_Pnt(float(pt[0]), float(pt[1]), 0) for pt in simplified_coords]
        if len(points) < 3:
            continue  # Skip if not enough points for a valid curve

        points_array = TColgp_Array1OfPnt(1, len(points))
        for i_, pt in enumerate(points):
            points_array.SetValue(i_ + 1, pt)

        try:
            bspline_curve = GeomAPI_PointsToBSpline(points_array, 3, 8, 2, 1e-6).Curve()  # Added parameters for robustness
            edge = BRepBuilderAPI_MakeEdge(bspline_curve).Edge()
            wire_builder = BRepBuilderAPI.BRepBuilderAPI_MakeWire()
            wire_builder.Add(edge)
            exterior_wire = wire_builder

            exterior_face = BRepBuilderAPI.BRepBuilderAPI_MakeFace(exterior_wire.Wire())
            shape = BRepPrimAPI.BRepPrimAPI_MakePrism(exterior_face.Face(), gp.gp_Vec(0, 0, abs(height / 1) * wh * 2 + np.finfo(float).eps)).Shape()

            if ring == polygon.exterior:
                builder.Add(compound, shape)
            else:
                compound = BRepAlgoAPI.BRepAlgoAPI_Cut(compound, shape).Shape()
        except Exception as e:
            print(f"Warning: Failed to process ring in create_cylinder: {e}")
            continue

    return compound

def create_CAD_mesh(generator, shape_code, shape_3d, CAD_mesh_filepath):
	"""
	Reconstruct shapes with sketch-extrude operations.
	"""
	wh = 500
	fill_sk_list = get_sketch_list(generator, shape_code, wh)
	ext_3d_list=[]
	for i in range(len(fill_sk_list)):
		if fill_sk_list[i]==None:
			continue
		
		rotation_qua = shape_3d[0,:4,i].detach().cpu().numpy()
		translation = shape_3d[0,4:7,i].detach().cpu().numpy()
		height = shape_3d[0,7,i].detach().cpu().numpy()
		print(f"height: {height}")
		quaternion = Quaternion(rotation_qua)  #[w,x,y,z]
		
		inverse = quaternion.inverse
		quaternion = np.asarray([inverse[3], inverse[0], inverse[1],inverse[2]]) # [x,y,z,w]
		if abs(height)*wh*2+np.finfo(float).eps<1:
			continue
		
		compound = TopoDS.TopoDS_Compound()
		builder = BRep.BRep_Builder()
		builder.MakeCompound(compound)
  
		# Handle different geometry types
		if isinstance(fill_sk_list[i], MultiPolygon):
			for polygon in fill_sk_list[i].geoms:
				if isinstance(polygon, Polygon):  # Ensure it's a Polygon
					compound = create_cylinder(polygon, builder, compound, height, wh)
		elif isinstance(fill_sk_list[i], GeometryCollection):
			for geom in fill_sk_list[i].geoms:
				if isinstance(geom, Polygon):  # Process only Polygon objects
					compound = create_cylinder(geom, builder, compound, height, wh)
				else:
					print(f"Warning: Skipping non-Polygon geometry {type(geom)} in GeometryCollection")
		elif isinstance(fill_sk_list[i], Polygon):
			compound = create_cylinder(fill_sk_list[i], builder, compound, height, wh)
		else:
			print(f"Warning: Skipping unsupported geometry type {type(fill_sk_list[i])}")
			continue

		# Apply translation and rotation
		transformation = gp.gp_Trsf()
		transformation.SetTranslationPart(gp.gp_Vec(0, 0, -abs(height) * wh))
		compound = BRepBuilderAPI.BRepBuilderAPI_Transform(compound, transformation).Shape()

		# Scale all points back
		transformation = gp.gp_Trsf()
		transformation.SetScaleFactor(1/wh)
		compound = BRepBuilderAPI.BRepBuilderAPI_Transform(compound, transformation).Shape()

		# Apply quaternion rotation
		transformation = gp.gp_Trsf()
		quaternion =  np.asarray(nn.functional.normalize(torch.from_numpy(quaternion), dim=-1))
		
		transformation.SetRotation(gp.gp_Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3]))
		compound = BRepBuilderAPI.BRepBuilderAPI_Transform(compound, transformation).Shape()

		# Apply translation along X, Y, and Z axes
		transformation = gp.gp_Trsf()
		transformation.SetTranslationPart(gp.gp_Vec(translation[0] * 1, translation[1] * 1, translation[2] * 1))
		compound = BRepBuilderAPI.BRepBuilderAPI_Transform(compound, transformation).Shape()

		ext_3d_list.append(compound)
  
	# Create a compound to hold all the cylinders
	compound = TopoDS.TopoDS_Compound()
	builder = BRep.BRep_Builder()
	builder.MakeCompound(compound)

	for shape in ext_3d_list:
		builder.Add(compound, shape)
  
	# Export the shapes to stl format (meshed)
	mesh = BRepMesh_IncrementalMesh(compound, 0.05)
	from OCC.Core.StlAPI import StlAPI_Writer
	writer = StlAPI_Writer()
	writer.Write(mesh.Shape(), CAD_mesh_filepath + '_CAD.stl')
 
	status = breptools.Write(compound, CAD_mesh_filepath + '_CAD.brep')

	if status:
		print('CAD saving ', CAD_mesh_filepath + '_CAD.brep')
	else:
		print('Failed to save the CAD file.')

def draw_2d_im_sketch(shape_code, generator, sk_filepath):
    """
    Draw 2D sketch images from the shape code and generator.

    Args:
        shape_code: Shape code tensor.
        generator: The generator network.
        sk_filepath: File path to save the sketch images
    """
    wh = 500  # Resolution of the sketch image, consistent with get_sketch_list
    B, N = 1, wh * wh

    # Generate 2D sample points
    x1, x2 = np.mgrid[-0.5:0.5:complex(0, wh), -0.5:0.5:complex(0, wh)]
    x1 = torch.from_numpy(x1).float()
    x2 = torch.from_numpy(x2).float()
    sample_points = torch.dstack((x1, x2)).view(-1, 2).unsqueeze(0).cuda()

    # Generate SDFs for each primitive
    shape_code_cuda = shape_code.cuda()
    
    sdfs_2d_list = []
    for i in range(4):
        head = getattr(generator, f'sketch_head_{i}')
        
        # Expand global features to match sample points
        feat = shape_code_cuda.unsqueeze(1).repeat(1, N, 1)  # [B, N, 256]
        
        # Use add_latent with the combined features
        latent = add_latent(sample_points, feat).float()
        
        sdfs_2d = head(latent).reshape(B, N, -1).float().squeeze().detach().cpu().numpy()
        sdfs_2d_list.append(sdfs_2d)

    # Convert sample points to image coordinates
    sample_points = sample_points.detach().cpu().numpy()[0][:, :2] / 1 + 0.5

    # Create and save a sketch image for each primitive
    for i, sdfs_2d in enumerate(sdfs_2d_list):
        # Create a canvas and fill with SDF values
        canvas = np.zeros((wh, wh))
        for pt, sdf in zip(sample_points, sdfs_2d):
            x, y = int(pt[0] * wh), int(pt[1] * wh)
            if 0 <= x < wh and 0 <= y < wh:
                canvas[y, x] = sdf  # Note: y, x order due to image coordinates

        # Binarize the image (threshold at -0.01, consistent with get_sketch_list)
        bin_img = (canvas < -0.01).astype(np.uint8) * 255

        # Save the image
        output_path = f"{sk_filepath}_sketch_{i}.png"
        cv2.imwrite(output_path, bin_img)
        print(f"Saved 2D sketch image: {output_path}")