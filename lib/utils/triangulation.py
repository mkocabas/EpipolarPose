import numpy as np
import cv2

'''
Code borrowed from: https://github.com/Eliasvan/Multiple-Quadrotor-SLAM
'''

def linear_eigen_triangulation(u1, P1, u2, P2, max_coordinate_value=1.e16):
	"""
	Linear Eigenvalue based (using SVD) triangulation.
	Wrapper to OpenCV's "triangulatePoints()" function.
	Relative speed: 1.0

	(u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
	(u2, P2) is the second pair.
	"max_coordinate_value" is a threshold to decide whether points are at infinity

	u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.

	The status-vector is based on the assumption that all 3D points have finite coordinates.
	"""
	x = cv2.triangulatePoints(P1[0:3, 0:4], P2[0:3, 0:4], u1.T, u2.T)  # OpenCV's Linear-Eigen triangl

	x[0:3, :] /= x[3:4, :]  # normalize coordinates
	x_status = (np.max(abs(x[0:3, :]), axis=0) <= max_coordinate_value)  # NaN or Inf will receive status False

	return x[0:3, :].T.astype(output_dtype), x_status


# Initialize consts to be used in linear_LS_triangulation()
linear_LS_triangulation_C = -np.eye(2, 3)


def linear_LS_triangulation(u1, P1, u2, P2):
	"""
	Linear Least Squares based triangulation.
	Relative speed: 0.1

	(u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
	(u2, P2) is the second pair.

	u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.

	The status-vector will be True for all points.
	"""
	A = np.zeros((4, 3))
	b = np.zeros((4, 1))

	# Create array of triangulated points
	x = np.zeros((3, len(u1)))

	# Initialize C matrices
	C1 = np.array(linear_LS_triangulation_C)
	C2 = np.array(linear_LS_triangulation_C)

	for i in range(len(u1)):
		# Derivation of matrices A and b:
		# for each camera following equations hold in case of perfect point matches:
		#     u.x * (P[2,:] * x)     =     P[0,:] * x
		#     u.y * (P[2,:] * x)     =     P[1,:] * x
		# and imposing the constraint:
		#     x = [x.x, x.y, x.z, 1]^T
		# yields:
		#     (u.x * P[2, 0:3] - P[0, 0:3]) * [x.x, x.y, x.z]^T     +     (u.x * P[2, 3] - P[0, 3]) * 1     =     0
		#     (u.y * P[2, 0:3] - P[1, 0:3]) * [x.x, x.y, x.z]^T     +     (u.y * P[2, 3] - P[1, 3]) * 1     =     0
		# and since we have to do this for 2 cameras, and since we imposed the constraint,
		# we have to solve 4 equations in 3 unknowns (in LS sense).

		# Build C matrices, to construct A and b in a concise way
		C1[:, 2] = u1[i, :]
		C2[:, 2] = u2[i, :]

		# Build A matrix:
		# [
		#     [ u1.x * P1[2,0] - P1[0,0],    u1.x * P1[2,1] - P1[0,1],    u1.x * P1[2,2] - P1[0,2] ],
		#     [ u1.y * P1[2,0] - P1[1,0],    u1.y * P1[2,1] - P1[1,1],    u1.y * P1[2,2] - P1[1,2] ],
		#     [ u2.x * P2[2,0] - P2[0,0],    u2.x * P2[2,1] - P2[0,1],    u2.x * P2[2,2] - P2[0,2] ],
		#     [ u2.y * P2[2,0] - P2[1,0],    u2.y * P2[2,1] - P2[1,1],    u2.y * P2[2,2] - P2[1,2] ]
		# ]
		A[0:2, :] = C1.dot(P1[0:3, 0:3])  # C1 * R1
		A[2:4, :] = C2.dot(P2[0:3, 0:3])  # C2 * R2

		# Build b vector:
		# [
		#     [ -(u1.x * P1[2,3] - P1[0,3]) ],
		#     [ -(u1.y * P1[2,3] - P1[1,3]) ],
		#     [ -(u2.x * P2[2,3] - P2[0,3]) ],
		#     [ -(u2.y * P2[2,3] - P2[1,3]) ]
		# ]
		b[0:2, :] = C1.dot(P1[0:3, 3:4])  # C1 * t1
		b[2:4, :] = C2.dot(P2[0:3, 3:4])  # C2 * t2
		b *= -1

		# Solve for x vector
		cv2.solve(A, b, x[:, i:i + 1], cv2.DECOMP_SVD)

	return x.T.astype(output_dtype), np.ones(len(u1), dtype=bool)


# Initialize consts to be used in iterative_LS_triangulation()
iterative_LS_triangulation_C = -np.eye(2, 3)


def iterative_LS_triangulation(u1, P1, u2, P2, tolerance=3.e-5):
	"""
	Iterative (Linear) Least Squares based triangulation.
	From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997.
	Relative speed: 0.025

	(u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
	(u2, P2) is the second pair.
	"tolerance" is the depth convergence tolerance.

	Additionally returns a status-vector to indicate outliers:
		1: inlier, and in front of both cameras
		0: outlier, but in front of both cameras
		-1: only in front of second camera
		-2: only in front of first camera
		-3: not in front of any camera
	Outliers are selected based on non-convergence of depth, and on negativity of depths (=> behind camera(s)).

	u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
	"""
	A = np.zeros((4, 3))
	b = np.zeros((4, 1))

	# Create array of triangulated points
	x = np.empty((4, len(u1)))
	x[3, :].fill(1)  # create empty array of homogenous 3D coordinates
	x_status = np.empty(len(u1), dtype=int)

	# Initialize C matrices
	C1 = np.array(iterative_LS_triangulation_C)
	C2 = np.array(iterative_LS_triangulation_C)

	for xi in range(len(u1)):
		# Build C matrices, to construct A and b in a concise way
		C1[:, 2] = u1[xi, :]
		C2[:, 2] = u2[xi, :]

		# Build A matrix
		A[0:2, :] = C1.dot(P1[0:3, 0:3])  # C1 * R1
		A[2:4, :] = C2.dot(P2[0:3, 0:3])  # C2 * R2

		# Build b vector
		b[0:2, :] = C1.dot(P1[0:3, 3:4])  # C1 * t1
		b[2:4, :] = C2.dot(P2[0:3, 3:4])  # C2 * t2
		b *= -1

		# Init depths
		d1 = d2 = 1.

		for i in range(10):  # Hartley suggests 10 iterations at most
			# Solve for x vector
			cv2.solve(A, b, x[0:3, xi:xi + 1], cv2.DECOMP_SVD)

			# Calculate new depths
			d1_new = P1[2, :].dot(x[:, xi])
			d2_new = P2[2, :].dot(x[:, xi])

			if abs(d1_new - d1) <= tolerance and \
							abs(d2_new - d2) <= tolerance:
				break

			# Re-weight A matrix and b vector with the new depths
			A[0:2, :] *= 1 / d1_new
			A[2:4, :] *= 1 / d2_new
			b[0:2, :] *= 1 / d1_new
			b[2:4, :] *= 1 / d2_new

			# Update depths
			d1 = d1_new
			d2 = d2_new

		# Set status
		x_status[xi] = (i < 10 and  # points should have converged by now
		                (d1_new > 0 and d2_new > 0))  # points should be in front of both cameras
		if d1_new <= 0: x_status[xi] -= 1
		if d2_new <= 0: x_status[xi] -= 2

	return x[0:3, :].T.astype(output_dtype), x_status


def polynomial_triangulation(u1, P1, u2, P2):
	"""
	Polynomial (Optimal) triangulation.
	Uses Linear-Eigen for final triangulation.
	Relative speed: 0.1

	(u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
	(u2, P2) is the second pair.

	u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.

	The status-vector is based on the assumption that all 3D points have finite coordinates.
	"""
	P1_full = np.eye(4)
	P1_full[0:3, :] = P1[0:3, :]  # convert to 4x4
	P2_full = np.eye(4)
	P2_full[0:3, :] = P2[0:3, :]  # convert to 4x4
	P_canon = P2_full.dot(cv2.invert(P1_full)[1])  # find canonical P which satisfies P2 = P_canon * P1

	# "F = [t]_cross * R" [HZ 9.2.4]; transpose is needed for numpy
	F = np.cross(P_canon[0:3, 3], P_canon[0:3, 0:3], axisb=0).T

	# Other way of calculating "F" [HZ (9.2)]
	# op1 = (P2[0:3, 3:4] - P2[0:3, 0:3] .dot (cv2.invert(P1[0:3, 0:3])[1]) .dot (P1[0:3, 3:4]))
	# op2 = P2[0:3, 0:4] .dot (cv2.invert(P1_full)[1][0:4, 0:3])
	# F = np.cross(op1.reshape(-1), op2, axisb=0).T

	# Project 2D matches to closest pair of epipolar lines
	u1_new, u2_new = cv2.correctMatches(F, u1.reshape(1, len(u1), 2), u2.reshape(1, len(u1), 2))

	# For a purely sideways trajectory of 2nd cam, correctMatches() returns NaN for all possible points!
	if np.isnan(u1_new).all() or np.isnan(u2_new).all():
		F = cv2.findFundamentalMat(u1, u2, cv2.FM_8POINT)[0]  # so use a noisy version of the fund mat
		u1_new, u2_new = cv2.correctMatches(F, u1.reshape(1, len(u1), 2), u2.reshape(1, len(u1), 2))

	# Triangulate using the refined image points
	return linear_eigen_triangulation(u1_new[0], P1, u2_new[0], P2)


output_dtype = float


def set_triangl_output_dtype(output_dtype_):
	"""
	Set the datatype of the triangulated 3D point positions.
	(Default is set to "float")
	"""
	global output_dtype
	output_dtype = output_dtype_