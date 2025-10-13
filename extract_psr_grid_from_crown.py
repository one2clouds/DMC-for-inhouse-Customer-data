import copy
import open3d as o3d
import numpy as np
from SAP.src.dpsr import DPSR
import torch
import glob 
import os 
import warnings
warnings.filterwarnings("ignore")

def normalize_for_dpsr(points, method='grid_space'):
    """
    Normalize points specifically for DPSR grid indexing
    
    DPSR expects coordinates in [0, res-1] range for grid indexing.
    The point_rasterize function maps coordinates to grid indices.
    
    Args:
        points: [B, N, 3] or [N, 3] point coordinates
        method: normalization method
    
    Returns:
        normalized_points: points in correct range for DPSR
        norm_params: parameters for denormalization
    """
    original_shape = points.shape
    if points.dim() == 2:
        points = points.unsqueeze(0)
    
    batch_size = points.shape[0]
    
    # Calculate bounding box
    min_coords = torch.min(points, dim=1, keepdim=True)[0]  # [B, 1, 3]
    max_coords = torch.max(points, dim=1, keepdim=True)[0]  # [B, 1, 3]
    center = (min_coords + max_coords) / 2.0
    
    if method == 'grid_space':
        # Method 1: Normalize to [0.1, 0.9] of grid space
        # This ensures points are well within grid bounds
        range_coords = max_coords - min_coords
        max_range = torch.max(range_coords, dim=2, keepdim=True)[0]  # [B, 1, 1]
        
        # Center and scale to fit in [0.1, 0.9]
        centered = points - center
        scale_factor = 0.8 / (max_range + 1e-8)  # Scale to fit in 0.8 range
        normalized_points = centered * scale_factor + 0.5  # Center at 0.5
        
        # Ensure bounds [0.1, 0.9]
        normalized_points = torch.clamp(normalized_points, 0.1, 0.9)
        
    elif method == 'symmetric_unit':
        # Method 2: Normalize to [-0.4, 0.4] then shift to [0.1, 0.9]
        centered = points - center
        max_extent = torch.max(torch.abs(centered))
        scale_factor = 0.4 / (max_extent + 1e-8)
        
        normalized_centered = centered * scale_factor  # [-0.4, 0.4]
        normalized_points = normalized_centered + 0.5  # [0.1, 0.9]
        normalized_points = torch.clamp(normalized_points, 0.1, 0.9)
        
    elif method == 'unit_cube_safe':
        # Method 3: Safe unit cube mapping
        range_coords = max_coords - min_coords
        
        # Map [min, max] -> [0.15, 0.85]
        normalized_points = (points - min_coords) / (range_coords + 1e-8)
        normalized_points = 0.15 + 0.7 * normalized_points  # [0.15, 0.85]
        normalized_points = torch.clamp(normalized_points, 0.1, 0.9)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Store normalization parameters
    norm_params = {
        'method': method,
        'min_coords': min_coords,
        'max_coords': max_coords, 
        'center': center,
        'original_shape': original_shape
    }
    
    if method == 'grid_space':
        norm_params.update({
            'range_coords': range_coords,
            'max_range': max_range,
            'scale_factor': scale_factor
        })
    elif method == 'symmetric_unit':
        norm_params.update({
            'max_extent': max_extent,
            'scale_factor': scale_factor
        })
    elif method == 'unit_cube_safe':
        norm_params.update({
            'range_coords': range_coords
        })
    
    # Restore original shape if needed
    if len(original_shape) == 2:
        normalized_points = normalized_points.squeeze(0)
    
    return normalized_points, norm_params

def pcl2psr_corrected(inputs, resolution=(128, 128, 128), sigma=2):
    """
    Convert point cloud to PSR with correct normalization for DPSR
    """
    original_shape = inputs.shape
    if inputs.dim() == 2:
        inputs = inputs.unsqueeze(0)
    
    points = inputs[..., :3]
    normals = inputs[..., 3:]
    
    # print(f"Original points range: [{points.min():.6f}, {points.max():.6f}]")
    
    # Try different normalization methods
    methods = ['grid_space', 'unit_cube_safe', 'symmetric_unit']
    
    for method in methods:
        try:
            # print(f"\n--- Trying normalization method: {method} ---")
            
            # Normalize points for DPSR grid indexing
            norm_points, norm_params = normalize_for_dpsr(points, method=method)
            
            # Clean and normalize normal vectors
            clean_normals = torch.nn.functional.normalize(normals, p=2, dim=-1)
            
            # Replace any NaN normals with default
            nan_mask = torch.isnan(clean_normals)
            if nan_mask.any():
                clean_normals = torch.where(nan_mask, 
                                          torch.tensor([0., 0., 1.], device=clean_normals.device), 
                                          clean_normals)
            
            # print(f"Normalized points range: [{norm_points.min():.6f}, {norm_points.max():.6f}]")
            # print(f"Normals range: [{clean_normals.min():.6f}, {clean_normals.max():.6f}]")
            
            # Ensure correct dtypes and device
            norm_points = norm_points.float()
            clean_normals = clean_normals.float()
            
            # Move to same device
            if torch.cuda.is_available():
                norm_points = norm_points.cuda()
                clean_normals = clean_normals.cuda()
            
            # Validate inputs
            assert not torch.isnan(norm_points).any(), "Points contain NaN"
            assert not torch.isinf(norm_points).any(), "Points contain Inf"
            assert not torch.isnan(clean_normals).any(), "Normals contain NaN"
            assert not torch.isinf(clean_normals).any(), "Normals contain Inf"
            
            # Check coordinate bounds
            # print(f"Point coordinate check:")
            # print(f"  Min per dim: {norm_points.min(dim=1)[0]}")
            # print(f"  Max per dim: {norm_points.max(dim=1)[0]}")
            
            # Create DPSR with specified parameters
            dpsr = DPSR(res=resolution, sig=sigma)
            if torch.cuda.is_available():
                dpsr = dpsr.cuda()
            
            # print(f"Running DPSR with resolution {resolution}, sigma {sigma}")
            
            # Run DPSR
            psr_grid = dpsr(norm_points, clean_normals)
            
            # Add channel dimension if needed
            if psr_grid.dim() == 4:
                psr_grid = psr_grid.unsqueeze(1)
            
            # print(f"SUCCESS! PSR grid shape: {psr_grid.shape}")
            # print(f"PSR grid range: [{psr_grid.min():.6f}, {psr_grid.max():.6f}]")
            
            # Restore original dimensions if needed
            if len(original_shape) == 2:
                norm_points = norm_points.squeeze(0)
                clean_normals = clean_normals.squeeze(0)
            
            return psr_grid, norm_points, clean_normals, norm_params
            
        except Exception as e:
            print(f"Method {method} failed: {str(e)}")
            if method == methods[-1]:
                print("\nDetailed error information:")
                print(f"Points shape: {norm_points.shape}")
                print(f"Points dtype: {norm_points.dtype}")
                print(f"Points device: {norm_points.device}")
                print(f"Normals shape: {clean_normals.shape}")
                print(f"Grid resolution: {resolution}")
                
                # Additional debugging
                print(f"\nCoordinate analysis:")
                for dim in range(3):
                    dim_coords = norm_points[..., dim]
                    print(f"  Dim {dim}: [{dim_coords.min():.6f}, {dim_coords.max():.6f}]")
                    print(f"    Mean: {dim_coords.mean():.6f}, Std: {dim_coords.std():.6f}")
                
                raise
            continue
    
    raise RuntimeError("All normalization methods failed")

def extract_pointcloud_from_mesh(mesh_path, num_points=8192):
    """
    Extract point cloud with robust mesh processing
    """
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0:
        raise ValueError(f"Could not load mesh from {mesh_path}")
    # print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    # Clean mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    # print(f"Cleaned mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    if len(mesh.triangles) == 0:
        raise ValueError("No triangles remain after cleaning")
    
    # Sample points uniformly
    pointcloud = mesh.sample_points_uniformly(number_of_points=num_points)
    # Compute normals
    pointcloud.estimate_normals()
    pointcloud.normalize_normals()
    # Try to orient normals consistently
    try:
        pointcloud.orient_normals_consistent_tangent_plane(k=min(30, num_points // 100))
    except:
        print("Warning: Could not orient normals consistently")
    # Convert to numpy
    points = np.asarray(pointcloud.points, dtype=np.float32)
    normals = np.asarray(pointcloud.normals, dtype=np.float32)
    # print(f"Sampled {len(points)} points")
    # print(f"Point bounds:")
    # print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    # print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]") 
    # print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    # Convert to tensors
    points_tensor = torch.from_numpy(points).float()
    normals_tensor = torch.from_numpy(normals).float()
    # Combine [N, 6]
    oriented_pointcloud = torch.cat([points_tensor, normals_tensor], dim=1)
    return oriented_pointcloud



if __name__=="__main__":
    count = 0
    meshes_path_ITERO = glob.glob("../CROWN_GEN_DATASET/Crown-for-abutment-ITERO-Only/*/*")
    meshes_path_PARTIAL = glob.glob("../CROWN_GEN_DATASET/Crown-for-abutment-PARTIAL-Only/*/*")
    combined_mesh_paths = sorted(meshes_path_ITERO + meshes_path_PARTIAL)
    
    output_path_ITERO = glob.glob("./outputs-CROWN-ITERO-ANNOTATED/*")
    output_path_PARTIAL = glob.glob("./outputs-CROWN-PARTIAL-ANNOTATED/*")
    combined_output_paths = sorted(output_path_ITERO + output_path_PARTIAL)
    
    # print(len(output_path_ITERO))
    # [print(os.path.basename(i)) for i in output_path_ITERO]
    # [print(os.path.basename(os.path.dirname(i))) for i in combined_mesh_paths]
    
    
    # print(meshes_path)
    for mesh_path in combined_mesh_paths:
        
        oriented_pointcloud = extract_pointcloud_from_mesh(mesh_path, num_points=8192)
        # Convert to PSR  
        psr_grid, norm_points, norm_normals, norm_params = pcl2psr_corrected(
        oriented_pointcloud, 
        resolution=(128,128,128), 
        sigma=2)

        psr_numpy = psr_grid.detach().cpu().numpy()

        if psr_numpy.ndim == 5:  # [B, C, H, W, D]
            psr_numpy = psr_numpy[0, 0]  # Take first batch, remove channel dim -> [H, W, D]
        
        for output_path in combined_output_paths:
            if os.path.basename(os.path.dirname(mesh_path)) in os.path.basename(output_path):
                print(output_path)
                print(os.path.basename(output_path))
                print(os.path.basename(os.path.dirname(mesh_path)))
            
                filename = f"{os.path.basename(output_path)}_psr.npz"
                print(filename)
                count += 1
                break

        full_path = os.path.join(output_path, filename)
        print(full_path)
        
        print("="*50)
        np.savez(full_path, psr=psr_numpy)
    print(count)





# [print(os.path.basename(i)) for i in glob.glob("/home/ank-server4/shirshak/DMC-for-inhouse-Customer-data/data/data_npy/*")]
