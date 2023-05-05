def read_cameras_txt(file_path, camera_id):
    """Reads camera parameters from a text file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('#'):  # Skip comment lines
            continue
        camera_info = line.strip().split()
        if int(camera_info[0]) == camera_id:
            return [float(val) for val in camera_info[4:]]

    raise ValueError(f"Camera with id {camera_id} not found in file.")

def read_images_txt(file_path):
    """Reads image data from a text file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    image_data = []
    for i in range(4, len(lines), 2):
        image_info = lines[i].strip().split()
        img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name = image_info
        image_data.append({
            'id': int(img_id),
            'quaternion': [float(qw), float(qx), float(qy), float(qz)],
            'translation': [float(tx), float(ty), float(tz)],
            'camera_id': int(cam_id),
            'name': name
        })
    return image_data

def read_points3D_txt(file_path):
    """Reads 3D points data from a text file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    points3D = []
    for line in lines[3:]:
        data = line.strip().split()
        point_id, x, y, z = map(float, data[:4])
        points3D.append([point_id, x, y, z])
    return points3D