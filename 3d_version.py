import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation


def create_cylinder(radius, height, resolution=20):
    """Create a vertical cylinder mesh"""
    theta = np.linspace(0, 2*np.pi, resolution)
    z = np.linspace(0, height, 2)
    theta, z = np.meshgrid(theta, z)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y, z


def create_sphere(radius, resolution=20):
    """Create a sphere mesh"""
    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def create_box(width, depth, height):
    """Create a box mesh"""
    vertices = np.array([
        [0, 0, 0], [width, 0, 0], [width, depth, 0], [0, depth, 0],
        [0, 0, height], [width, 0, height], [width, depth, height], [0, depth, height]
    ])
    
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
        [vertices[3], vertices[0], vertices[4], vertices[7]]   # left
    ]
    
    return faces


def rotate_point_3d(point, axis, angle):
    """Rotate a point around an axis by an angle"""
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Rotation matrix (Rodrigues' rotation formula)
    rot_matrix = np.array([
        [cos_a + axis[0]**2 * (1 - cos_a),
         axis[0] * axis[1] * (1 - cos_a) - axis[2] * sin_a,
         axis[0] * axis[2] * (1 - cos_a) + axis[1] * sin_a],
        [axis[1] * axis[0] * (1 - cos_a) + axis[2] * sin_a,
         cos_a + axis[1]**2 * (1 - cos_a),
         axis[1] * axis[2] * (1 - cos_a) - axis[0] * sin_a],
        [axis[2] * axis[0] * (1 - cos_a) - axis[1] * sin_a,
         axis[2] * axis[1] * (1 - cos_a) + axis[0] * sin_a,
         cos_a + axis[2]**2 * (1 - cos_a)]
    ])
    
    return rot_matrix @ point


def forward_kinematics_articulated(lengths, angles):
    """
    Forward kinematics for articulated robot arm (typical industrial robot).
    angles: array of shape (n_joints, 3) where each row is [base_rot, joint_angle, twist]
           or simpler: [base_rot, shoulder, elbow, wrist]
    """
    positions = []
    orientations = []
    
    # Starting position (base)
    pos = np.array([0.0, 0.0, 0.0])
    # Base orientation (pointing up)
    current_dir = np.array([0.0, 0.0, 1.0])
    current_right = np.array([1.0, 0.0, 0.0])
    current_up = np.array([0.0, 1.0, 0.0])
    
    positions.append(pos.copy())
    
    # Base rotation (rotate around Z axis)
    base_rot = angles[0, 0] if len(angles) > 0 else 0
    current_dir = rotate_point_3d(current_dir, [0, 0, 1], base_rot)
    current_right = rotate_point_3d(current_right, [0, 0, 1], base_rot)
    current_up = rotate_point_3d(current_up, [0, 0, 1], base_rot)
    
    # Shoulder joint (rotate around Y axis in local frame)
    if len(angles) > 1:
        shoulder_angle = angles[1, 0]
        current_dir = rotate_point_3d(current_dir, current_right, shoulder_angle)
        current_up = rotate_point_3d(current_up, current_right, shoulder_angle)
    
    # Move along first segment
    if len(lengths) > 0:
        pos += lengths[0] * current_dir
        positions.append(pos.copy())
    
    # Elbow joint (rotate around Y axis)
    if len(angles) > 2:
        elbow_angle = angles[2, 0]
        current_dir = rotate_point_3d(current_dir, current_right, elbow_angle)
        current_up = rotate_point_3d(current_up, current_right, elbow_angle)
    
    # Move along second segment
    if len(lengths) > 1:
        pos += lengths[1] * current_dir
        positions.append(pos.copy())
    
    # Wrist joint (rotate around Y axis)
    if len(angles) > 3:
        wrist_angle = angles[3, 0]
        current_dir = rotate_point_3d(current_dir, current_right, wrist_angle)
        current_up = rotate_point_3d(current_up, current_right, wrist_angle)
    
    # Move along third segment
    if len(lengths) > 2:
        pos += lengths[2] * current_dir
        positions.append(pos.copy())
    
    # Final segment (forearm)
    if len(lengths) > 3:
        pos += lengths[3] * current_dir
        positions.append(pos.copy())
    
    return np.array(positions)


def forward_kinematics_3d(lengths, angles):
    """
    Simplified forward kinematics using yaw and pitch per joint.
    angles: array of shape (n_joints, 2) where each row is [yaw, pitch]
    """
    x, y, z = 0, 0, 0
    positions = [(x, y, z)]
    
    for i, length in enumerate(lengths):
        yaw, pitch = angles[i, 0], angles[i, 1]
        # Spherical to Cartesian conversion
        dx = length * np.sin(pitch) * np.cos(yaw)
        dy = length * np.sin(pitch) * np.sin(yaw)
        dz = length * np.cos(pitch)
        
        x += dx
        y += dy
        z += dz
        positions.append((x, y, z))
    
    return np.array(positions)


def fitness(lengths, xg, yg, zg, angles):
    """Calculate fitness as distance squared from target in 3D"""
    positions = forward_kinematics_3d(lengths, angles.reshape(-1, 2))
    end_pos = positions[-1]
    res = (end_pos[0] - xg)**2 + (end_pos[1] - yg)**2 + (end_pos[2] - zg)**2
    return res


def pso(lengths, xg, yg, zg, num_particles=100, num_iterations=50, w=0.7, c1=1.5, c2=1.5):
    """Particle Swarm Optimization for 3D inverse kinematics"""
    num_joints = len(lengths)
    num_dimensions = num_joints * 2  # 2 angles per joint (yaw, pitch)
    
    # Bounds: yaw [0, 2π], pitch [0, π]
    bounds = np.array([[0, 2*np.pi], [0, np.pi]] * num_joints)
    positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles, num_dimensions))
    velocities = np.zeros((num_particles, num_dimensions))
    personal_best_positions = positions.copy()
    personal_best_scores = np.array([fitness(lengths, xg, yg, zg, pos) for pos in positions])
    global_best_idx = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_idx].copy()

    for _ in range(num_iterations):
        r1 = np.random.rand(num_particles, num_dimensions)
        r2 = np.random.rand(num_particles, num_dimensions)
        velocities = (
            w * velocities
            + c1 * r1 * (personal_best_positions - positions)
            + c2 * r2 * (global_best_position - positions)
        )
        positions += velocities
        positions = np.clip(positions, bounds[:, 0], bounds[:, 1])
        
        scores = np.array([fitness(lengths, xg, yg, zg, pos) for pos in positions])
        mask = scores < personal_best_scores
        personal_best_scores[mask] = scores[mask]
        personal_best_positions[mask] = positions[mask]
        best_idx = np.argmin(personal_best_scores)
        if personal_best_scores[best_idx] < fitness(lengths, xg, yg, zg, global_best_position.reshape(-1, 2)):
            global_best_position = personal_best_positions[best_idx].copy()
        
    return global_best_position.reshape(-1, 2)


def draw_robot_arm(ax, positions, lengths, base_size=3, joint_radius=0.8, segment_radius=0.6):
    """Draw a 3D robot arm with solid geometric shapes"""
    # Clear previous drawings (we'll redraw each frame)
    
    # Draw base
    base_height = base_size * 0.3
    base_width = base_size
    base_faces = create_box(base_width, base_width, base_height)
    base_collection = Poly3DCollection(base_faces, facecolors='#888888', edgecolors='#555555', alpha=1.0, linewidths=1)
    ax.add_collection3d(base_collection)
    
    # Base cylinder (waist)
    base_cyl_height = base_size * 0.4
    base_x, base_y, base_z = create_cylinder(segment_radius * 1.2, base_cyl_height, resolution=20)
    base_z = base_z + base_height
    ax.plot_surface(base_x, base_y, base_z, color='#888888', alpha=1.0, shade=True)
    
    if len(positions) < 2:
        return
    
    # Draw each segment
    for i in range(len(positions) - 1):
        start = positions[i]
        end = positions[i + 1]
        length = lengths[i] if i < len(lengths) else np.linalg.norm(end - start)
        
        # Direction vector
        direction = end - start
        if np.linalg.norm(direction) < 1e-6:
            continue
        
        direction = direction / np.linalg.norm(direction)
        
        # Calculate rotation to align cylinder with direction
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction)
        if np.linalg.norm(rotation_axis) < 1e-6:
            rotation_axis = np.array([1, 0, 0])
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
        
        # Create cylinder aligned with direction
        x, y, z = create_cylinder(segment_radius, length, resolution=16)
        
        # Rotate and translate cylinder
        for j in range(len(x)):
            for k in range(len(x[0])):
                point = np.array([x[j, k], y[j, k], z[j, k]])
                if rotation_angle > 1e-6:
                    point = rotate_point_3d(point, rotation_axis, rotation_angle)
                point += start
                x[j, k] = point[0]
                y[j, k] = point[1]
                z[j, k] = point[2]
        
        ax.plot_surface(x, y, z, color='#AAAAAA', alpha=1.0, shade=True)
        
        # Draw joint (sphere) at the end of segment
        if i < len(positions) - 1:
            joint_x, joint_y, joint_z = create_sphere(joint_radius, resolution=16)
            joint_x += end[0]
            joint_y += end[1]
            joint_z += end[2]
            ax.plot_surface(joint_x, joint_y, joint_z, color='#666666', alpha=1.0, shade=True)
    
    # Draw gripper at end effector
    if len(positions) > 1:
        gripper_pos = positions[-1]
        gripper_dir = (positions[-1] - positions[-2]) if len(positions) > 1 else np.array([0, 0, 1])
        gripper_dir = gripper_dir / (np.linalg.norm(gripper_dir) + 1e-6)
        
        # Simple gripper: two fingers
        finger_length = 1.5
        finger_width = 0.2
        finger_offset = 0.3
        
        # Calculate perpendicular directions for gripper fingers
        if abs(gripper_dir[2]) < 0.9:
            perp1 = np.cross(gripper_dir, [0, 0, 1])
        else:
            perp1 = np.cross(gripper_dir, [1, 0, 0])
        perp1 = perp1 / (np.linalg.norm(perp1) + 1e-6)
        perp2 = np.cross(gripper_dir, perp1)
        perp2 = perp2 / (np.linalg.norm(perp2) + 1e-6)
        
        # Draw two gripper fingers
        for offset_dir in [perp1, -perp1]:
            finger_start = gripper_pos + offset_dir * finger_offset
            finger_end = finger_start + gripper_dir * finger_length
            
            # Create small cylinder for finger
            x, y, z = create_cylinder(finger_width * 0.5, finger_length, resolution=12)
            
            # Rotate and translate
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, gripper_dir)
            if np.linalg.norm(rotation_axis) > 1e-6:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                rotation_angle = np.arccos(np.clip(np.dot(z_axis, gripper_dir), -1, 1))
                
                for j in range(len(x)):
                    for k in range(len(x[0])):
                        point = np.array([x[j, k], y[j, k], z[j, k]])
                        if rotation_angle > 1e-6:
                            point = rotate_point_3d(point, rotation_axis, rotation_angle)
                        point += finger_start
                        x[j, k] = point[0]
                        y[j, k] = point[1]
                        z[j, k] = point[2]
                
                ax.plot_surface(x, y, z, color='#333333', alpha=1.0, shade=True)


def update(frame, ax, ball_pos, lengths, theta, max_reach, base_height, base_cyl_height, a=5):
    """Update animation frame"""
    global perc
    global temp_theta
    
    perc += (1 - perc)/a
    
    # Interpolate between previous and target angles
    delta = theta - prev_theta
    # Handle angle wraparound for yaw
    delta[:, 0] = ((delta[:, 0] + np.pi) % (2*np.pi)) - np.pi
    temp_theta[:, :] = prev_theta + perc * delta
    temp_theta[:, 0] = temp_theta[:, 0] % (2*np.pi)
    temp_theta[:, 1] = np.clip(temp_theta[:, 1], 0, np.pi)
    
    # Calculate new positions
    positions = forward_kinematics_3d(lengths, temp_theta)
    # Adjust for base height
    positions[:, 2] += base_height + base_cyl_height
    
    # Clear and redraw robot arm
    ax.clear()
    setup_axes(ax, max_reach)
    draw_robot_arm(ax, positions, lengths)
    
    # Update ball position
    ax.scatter([ball_pos[0]], [ball_pos[1]], [ball_pos[2] + base_height + base_cyl_height], 
               s=300, c="#55cd97", marker='o', alpha=0.9)
    
    return []


def setup_axes(ax, max_reach):
    """Setup axes appearance"""
    limit = max_reach * 1.1
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([0, limit * 1.2])
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.tick_params(colors='white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#444444')
    ax.yaxis.pane.set_edgecolor('#444444')
    ax.zaxis.pane.set_edgecolor('#444444')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    ax.grid(True, alpha=0.2, color='white')
    ax.set_facecolor('#1a1a1a')
    
    # Draw ground plane
    ground_size = max_reach * 0.6
    xx = np.linspace(-ground_size, ground_size, 10)
    yy = np.linspace(-ground_size, ground_size, 10)
    X, Y = np.meshgrid(xx, yy)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.15, color='gray')


def on_click(event, ax, ball_pos, theta, lengths, num_particles, num_iterations, w, c1, c2, base_height, base_cyl_height):
    """Handle mouse click to set new target position"""
    if event.inaxes is None or event.inaxes != ax:
        return
    if event.button != 1:  # Only left click
        return
    
    global perc
    global prev_theta
    
    prev_theta[:, :] = theta.copy()
    perc = 0
    
    # Get the clicked coordinates
    x_click = event.xdata
    y_click = event.ydata
    
    if x_click is None or y_click is None:
        return
    
    # Get current view limits for z estimation
    z_center = np.mean(ax.get_zlim())
    z_target = event.zdata if hasattr(event, 'zdata') and event.zdata is not None else z_center
    
    ball_pos[0] = x_click
    ball_pos[1] = y_click
    ball_pos[2] = z_target - base_height - base_cyl_height
    
    theta[:, :] = pso(lengths, ball_pos[0], ball_pos[1], ball_pos[2], num_particles, num_iterations, w, c1, c2)


if __name__ == '__main__':
    lengths = np.array([10, 8, 5, 2])
    max_reach = sum(lengths)
    base_size = 3
    base_height = base_size * 0.3
    base_cyl_height = base_size * 0.4
    
    ball_pos = np.array([max_reach*0.5, 0, max_reach*0.3])

    a = 5
    num_particles = 200
    num_iterations = 100
    w = 0.7
    c1 = 1.5
    c2 = 1.5

    # Initialize angles (yaw, pitch for each joint)
    theta = pso(lengths, ball_pos[0], ball_pos[1], ball_pos[2], num_particles, num_iterations, w, c1, c2)
    prev_theta = np.zeros((len(lengths), 2))
    temp_theta = np.zeros((len(lengths), 2))
    perc = 0
    
    # Create 3D figure
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up dark theme
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_title('3D Robot Arm - Click to change target position', color='white', pad=20, fontsize=14)
    
    setup_axes(ax, max_reach)
    
    # Initial robot arm position
    positions = forward_kinematics_3d(lengths, theta)
    positions[:, 2] += base_height + base_cyl_height
    
    # Draw initial robot arm
    draw_robot_arm(ax, positions, lengths)
    
    # Draw target ball
    ball = ax.scatter([ball_pos[0]], [ball_pos[1]], [ball_pos[2] + base_height + base_cyl_height], 
                      s=300, c="#55cd97", marker='o', alpha=0.9)

    # Animation
    ani = FuncAnimation(fig, lambda f: update(f, ax, ball_pos, lengths, theta, max_reach, base_height, base_cyl_height, a), 
                       interval=1000/60, blit=False)
    
    # Mouse click handler
    def handle_click(event):
        on_click(event, ax, ball_pos, theta, lengths, num_particles, num_iterations, w, c1, c2, base_height, base_cyl_height)
    
    fig.canvas.mpl_connect('button_press_event', handle_click)
    
    plt.show()
