import numpy as np

class EnvQCarRef:
    def __init__(self, threshold: float = 0.01) -> None:
        self.width: float = 0.21 + threshold / 2
        self.length: float = 0.39 + threshold
        self.corners = np.array([
            [self.width/2, self.length/2],
            [-self.width/2, self.length/2],
            [-self.width/2, -self.length/2],
            [self.width/2, -self.length/2]
        ])

    def get_corners(self, orig: np.ndarray, yaw: float) -> np.ndarray:
        rot = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        return orig + np.dot(self.corners, rot.T)
    

def separating_axis_theorem(corners1, corners2) -> bool:
    def get_axes(corners):
        axes = []
        for i in range(len(corners)):
            p1 = corners[i]
            p2 = corners[(i + 1) % len(corners)]
            edge = p1 - p2
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal)
        return np.array(axes)

    axes = np.vstack([get_axes(corners1), get_axes(corners2)])

    for axis in axes:
        projection1 = np.dot(corners1, axis)
        projection2 = np.dot(corners2, axis)
        if max(projection1) < min(projection2) or max(projection2) < min(projection1):
            return False
    return True


def is_collided(orig_1, yaw_1, orig_2, yaw_2):
    corners_1 = EnvQCarRef().get_corners(orig_1, yaw_1)
    corners_2 = EnvQCarRef().get_corners(orig_2, yaw_2)
    return separating_axis_theorem(corners_1, corners_2)


if __name__ == "__main__":
    orig_1 = np.array([0, 0])
    yaw_1 = np.pi/180 * 250
    orig_2 = np.array([0.3, 0])
    yaw_2 = np.pi/4

    collision = is_collided(orig_1, yaw_1, orig_2, yaw_2)
    print("Collision detected:", collision)
