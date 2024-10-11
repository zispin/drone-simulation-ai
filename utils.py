import airsim
import numpy as np
import math

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
client.takeoffAsync().join()

# Set up object detection
camera_name = "0"
image_type = airsim.ImageType.Scene
client.simSetDetectionFilterRadius(camera_name, image_type, 1000 * 100)  # 1000 meters in cm
client.simAddDetectionFilterMeshName(camera_name, image_type, "Sphere*")  # Detect objects with names starting with "Sphere"

def get_drone_state():
    kinematics = client.getMultirotorState().kinematics_estimated
    position = kinematics.position
    velocity = kinematics.linear_velocity
    orientation = kinematics.orientation

    state = np.array([position.x_val, position.y_val, position.z_val,
                      velocity.x_val, velocity.y_val, velocity.z_val,
                      orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val])
    return state

def perform_action(action):
    vx, vy, vz, yaw_rate = action
    duration = 1  # Duration of the action in seconds

    # Convert velocities from m/s to AirSim's scale
    vx, vy, vz = vx * 5, vy * 5, vz * 5  # Assuming max speed of 5 m/s in each direction
    
    # Convert yaw rate from rad/s to degrees/s
    yaw_rate = np.degrees(yaw_rate)

    client.moveByVelocityAsync(vx, vy, vz, duration).join()
    client.rotateByYawRateAsync(yaw_rate, duration).join()

def get_ball_state():
    detections = client.simGetDetections(camera_name, image_type)
    
    if not detections:
        print("Ball not detected")
        return np.array([0, 0, 0])  # Return default state when ball is not detected
    
    # Assume the first detection is our ball
    ball = detections[0]
    
    # Get the relative position of the ball
    relative_position = ball.relative_pose.position

    # Get the current drone position
    drone_position = get_drone_state()[:3]  # Get only x, y, z

    # Estimate the global position of the sphere
    sphere_x = drone_position[0] + relative_position.x_val
    sphere_y = drone_position[1] + relative_position.y_val
    sphere_z = drone_position[2] + relative_position.z_val  # Adjust if necessary

    # Calculate the size of the ball (you might want to adjust this based on your needs)
    size = (ball.box2D.max.x_val - ball.box2D.min.x_val) * (ball.box2D.max.y_val - ball.box2D.min.y_val)
    size = size / (100 * 100)  # Normalize size
    
    return np.array([sphere_x, sphere_y, sphere_z, size])


def compute_reward(ball_state, drone_state, action):
    ball_x, ball_y, ball_size = ball_state
    drone_x, drone_y, drone_z = drone_state[:3]
    
    if ball_size == 0:  # Ball not detected
        return -10  # Penalty for not seeing the ball
    
    # Reward for keeping the ball in view
    view_reward = 10 * ball_size
    
    # Reward for maintaining desired altitude
    altitude_reward = -abs(drone_z + 5)  # Assuming desired altitude is -5
    
    # Reward for circular movement
    circular_reward = 5 * (action[0]**2 + action[1]**2)**0.5  # Reward horizontal movement
    
    # Penalty for being too close or too far from the ball
    distance = (ball_x**2 + ball_y**2)**0.5
    distance_penalty = -abs(distance - 0.5) * 10  # Assuming desired distance is 0.5 (in normalized coordinates)
    
    total_reward = view_reward + altitude_reward + circular_reward + distance_penalty
    
    return total_reward

def get_state():
    drone_state = get_drone_state()
    ball_state = get_ball_state()
    
    # Ensure that the state vector always has the correct length
    if len(ball_state) < 3:
        ball_state = np.zeros(3)  # Fill with zeros if ball is not detected
    
    return np.concatenate([drone_state, ball_state])

# Function to calculate the direction to the target
def get_direction_to_target(ball_state):
    ball_x, ball_y, _ = ball_state
    return np.array([ball_x, ball_y])

# Function to select action based on the direction vector
def select_action(ball_state):
    direction_vector = get_direction_to_target(ball_state)

    if np.linalg.norm(direction_vector) < 0.1:  # If close enough, hover or stop
        return np.array([0, 0, 0, 0])  # Hover
    else:
        # Move towards the ball
        vx = direction_vector[0]
        vy = direction_vector[1]
        vz = 0  # Maintain altitude
        yaw_rate = 0  # No rotation

        # Normalize velocities
        magnitude = np.linalg.norm([vx, vy, vz])
        if magnitude > 1:
            vx, vy, vz = [v / magnitude for v in [vx, vy, vz]]

        return np.array([vx, vy, vz, yaw_rate])