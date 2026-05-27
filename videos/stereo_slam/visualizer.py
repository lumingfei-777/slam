import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from config import OUT_DIR


class Visualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.frame_count = 0

    def init_3d(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')

    def update(self, trajectory, map_points):
        self.frame_count += 1
        if self.fig is None:
            self.init_3d()
        self.ax.cla()
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.set_title(f'Stereo SLAM (frame {self.frame_count})')

        if len(trajectory) > 0:
            self.ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                         'b-', linewidth=1.5, label='trajectory')
            self.ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                            c='r', s=50, marker='o')
        if len(map_points) > 0:
            self.ax.scatter(map_points[:, 0], map_points[:, 1], map_points[:, 2],
                            c='g', s=1, alpha=0.5, label='map points')
        self.ax.legend()

        if len(trajectory) > 0:
            all_pts = trajectory if len(map_points) == 0 else np.vstack([trajectory, map_points])
            center = all_pts.mean(axis=0)
            max_r = max(np.ptp(all_pts, axis=0).max() / 2 + 500, 1000)
            self.ax.set_xlim(center[0] - max_r, center[0] + max_r)
            self.ax.set_ylim(center[1] - max_r, center[1] + max_r)
            self.ax.set_zlim(center[2] - max_r, center[2] + max_r)

        self.ax.view_init(elev=-90, azim=-90)

    def save_frame(self, name=None):
        if self.fig is None:
            return
        if name is None:
            name = f'frame_{self.frame_count:04d}.png'
        path = os.path.join(OUT_DIR, 'viz', name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            self.fig.savefig(path, dpi=80, bbox_inches='tight')
        except Exception as e:
            print(f"      Viz save failed: {e}")

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
