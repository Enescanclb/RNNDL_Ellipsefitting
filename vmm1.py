import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from funcs import superellipse_sampler_1

def intersection_points(ellipse_params, angle_degrees, ray_start=np.array([0, 0])):
    centroid = np.array([ellipse_params[0], ellipse_params[1]])
    tau = ellipse_params[4]
    tilt = np.linalg.inv(np.array([[np.cos(tau), -np.sin(tau)],
                     [np.sin(tau), np.cos(tau)]]))

    angle_radians = np.radians(angle_degrees)
    ray_direction = np.array([np.cos(angle_radians), np.sin(angle_radians)])

    ray_start_rotated = np.dot(tilt, ray_start - centroid)
    ray_direction_rotated = np.dot(tilt, ray_direction)

    a, b = ellipse_params[2], ellipse_params[3]
    p, q = ray_start_rotated
    u, v = ray_direction_rotated

    A = u**2 / a**2 + v**2 / b**2
    B = 2 * p * u / a**2 + 2 * q * v / b**2
    C = p**2 / a**2 + q**2 / b**2 - 1
    discriminant = B**2 - 4 * A * C

    
    if discriminant < 0:
        return []

    if discriminant == 0:
        t = -B / (2 * A)
        return np.array([ray_start + t * ray_direction])

    t1 = (-B + np.sqrt(discriminant)) / (2 * A)
    t2 = (-B - np.sqrt(discriminant)) / (2 * A)
    p1= np.array(ray_start + t1 * ray_direction)
    p2= np.array(ray_start + t2 * ray_direction)
    d1= np.sqrt(p1.dot(p1))
    d2= np.sqrt(p2.dot(p2))
    if d1 < d2:
        return p1
    else:
        return p2



ellipse_params=np.array([20,20,10,4,0])
intersection_point=intersection_points(ellipse_params,30)

fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')
ax.set_xlim(0, 75)
ax.set_ylim(0, 75)

centroid_x, centroid_y, width, height, tilt_angle = ellipse_params
ellipse = Ellipse(xy=(centroid_x, centroid_y), width=width*2, height=height*2, angle=np.degrees(tilt_angle),
                      edgecolor='r', fc='None')
ax.add_patch(ellipse)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Ellipse')
ax.grid(True)


for angle_degrees in range(0, 360, 1):
    intersection_point=intersection_points(ellipse_params,angle_degrees)
    if len(intersection_point):
        ax.plot(intersection_point[0], intersection_point[1], 'ro')



plt.xlabel('X')
plt.ylabel('Y')
plt.title('Intersection of Rays and Elliptical Object')
plt.grid(True)
plt.show()