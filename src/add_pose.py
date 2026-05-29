
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    # Odometry from X(3) to X(4):
    # rotate 45 deg, move 2 m, rotate 45 deg again
    odometry = gtsam.Pose2(np.sqrt(2), np.sqrt(2), np.pi / 2)

    # Add odometry factor between X(3) and X(4)
    graph.add(
        gtsam.BetweenFactorPose2(
            X(3),
            X(4),
            odometry,
            ODOMETRY_NOISE
        )
    )

    # Create initial estimate for X(4) based on X(3) and the odometry
    pose_3 = initial_estimate.atPose2(X(3))
    pose_4 = pose_3.compose(odometry)

    # Insert initial guess for X(4)
    initial_estimate.insert(X(4), pose_4)

    return graph, initial_estimate