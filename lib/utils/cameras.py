import h5py
import numpy as np
import cv2

class Camera():
    def __init__(self, cam_params):
        # R, T, f, c, k, p, name
        self.cam_params = cam_params
        self.R, self.T, self.f, self.c, self.k, self.p, self.name = cam_params

        self.camera_matrix = self.get_intrinsic_matrix()
        if self.k is not None or self.p is not None:
            self.dist_coeffs = self.get_dist_coeffs()

        self.tvec = self.get_tvec()
        self.projection_matrix = self.get_projection_matrix()

    def project_point_radial(self, P):
        """
        Project points from 3d to 2d using camera parameters
        including radial and tangential distortion

        Args
        P: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
        Returns
        Proj: Nx2 points in pixel space
        D: 1xN depth of each point in camera space
        radial: 1xN radial distortion per point
        tan: 1xN tangential distortion per point
        r2: 1xN squared radius of the projected points before distortion
        """

        # P is a matrix of 3-dimensional points
        assert len(P.shape) == 2
        assert P.shape[1] == 3

        N = P.shape[0]
        X = self.R.dot( P.T - self.T ) # rotate and translate
        XX = X[:2,:] / X[2,:]
        r2 = XX[0,:]**2 + XX[1,:]**2

        radial = 1 + np.einsum( 'ij,ij->j', np.tile(self.k,(1, N)), np.array([r2, r2**2, r2**3]) )
        tan = self.p[0]*XX[1,:] + self.p[1]*XX[0,:]

        XXX = XX * np.tile(radial+tan,(2,1)) + \
              np.outer(np.array([self.p[1], self.p[0]]).reshape(-1), r2 )

        Proj = (self.f * XXX) + self.c
        Proj = Proj.T

        D = X[2,]

        return Proj, D, radial, tan, r2

    def unproject_pts(self, pts_uv, pts_d):
        """
        This function converts a set of 2D image coordinates to vectors in pinhole camera space.
        Hereby the intrinsics of the camera are taken into account.
        UV is converted to normalized image space (think frustum with image plane at z=1) then undistored
        adding a z_coordinate of 1 yield vectors pointing from 0,0,0 to the undistored image pixel.
        @return: ndarray with shape=(n, 3)

        """
        pts_uv = np.array(pts_uv)
        num_pts = pts_uv.size / 2
        pts_uv.shape = (int(num_pts), 1, 2)

        pts_uv = cv2.undistortPoints(pts_uv, self.camera_matrix, self.dist_coeffs)

        pts_3d = cv2.convertPointsToHomogeneous(np.float32(pts_uv))
        pts_3d.shape = (int(num_pts),3)

        pts_d.shape = (int(num_pts),1)

        return pts_3d * pts_d

    def world_to_camera_frame(self, P):
        """
        Convert points from world to camera coordinates

        Args
        P: Nx3 3d points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        Returns
        X_cam: Nx3 3d points in camera coordinates
        """

        assert len(P.shape) == 2
        assert P.shape[1] == 3

        X_cam = self.R.dot( P.T - self.T ) # rotate and translate

        return X_cam.T

    def camera_to_world_frame(self, P):
        """Inverse of world_to_camera_frame

        Args
        P: Nx3 points in camera coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        Returns
        X_cam: Nx3 points in world coordinates
        """

        assert len(P.shape) == 2
        assert P.shape[1] == 3

        X_cam = self.R.T.dot( P.T ) + self.T # rotate and translate

        return X_cam.T

    def get_intrinsic_matrix(self):
        fx, fy = self.cam_params[2]
        cx, cy = self.cam_params[3]
        K = np.array([[fx, 0., cx],[0., fy, cy],[0., 0., 1.]]).astype(np.double)
        return K

    def get_projection_matrix(self):
        K = self.get_intrinsic_matrix()
        T = self.tvec
        if len(T.shape) < 2:
            T = np.expand_dims(T, axis=-1)
        return np.dot(K, np.concatenate((self.R, T), axis=1))

    def get_essential_matrix(self, fundamental_mat):
        return np.dot(np.dot(self.camera_matrix.T, fundamental_mat), self.camera_matrix)

    def get_fundamental_matrix(self, u1, u2):
        u1 = np.int32(u1)
        u2 = np.int32(u2)
        F, mask = cv2.findFundamentalMat(u1, u2, cv2.FM_LMEDS)
        # We select only inlier points
        u1 = u1[mask.ravel() == 1]
        u2 = u2[mask.ravel() == 1]
        return F, (u1, u2)

    def get_disp_matrix(self):
        T = self.get_tvec()
        return np.concatenate((self.R, T), axis=1)

    def get_tvec(self):
        return np.dot(self.R, np.negative(self.T))

    def get_dist_coeffs(self):
        return np.array([self.cam_params[4][0],
                         self.cam_params[4][1],
                         self.cam_params[5][0],
                         self.cam_params[5][1],
                         self.cam_params[4][2]])

    def project_points(self, P):
        kps, _ = cv2.projectPoints(objectPoints=P,
                                   rvec=self.R,
                                   tvec=self.tvec,
                                   cameraMatrix=self.camera_matrix,
                                   distCoeffs=self.dist_coeffs)
        return kps

def load_camera_params( hf, path ):
    """Load h36m camera parameters

    Args
    hf: hdf5 open file with h36m cameras data
    path: path or key inside hf to the camera we are interested in
    Returns
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    name: String with camera id
    """

    R = hf[ path.format('R') ][:]
    R = R.T

    T = hf[ path.format('T') ][:]
    f = hf[ path.format('f') ][:]
    c = hf[ path.format('c') ][:]
    k = hf[ path.format('k') ][:]
    p = hf[ path.format('p') ][:]

    name = hf[ path.format('Name') ][:]
    name = "".join( [chr(item) for item in name] )

    return R, T, f, c, k, p, name

def load_cameras(bpath='cameras.h5', subjects=[1,5,6,7,8,9,11]):
    """Loads the cameras of h36m

    Args
    bpath: path to hdf5 file with h36m camera data
    subjects: List of ints representing the subject IDs for which cameras are requested
    Returns
    rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
    """
    rcams = {}

    with h5py.File(bpath,'r') as hf:
        for s in subjects:
            for c in range(4): # There are 4 cameras in human3.6m
                rcams[(s, c+1)] = load_camera_params(hf, 'subject%d/camera%d/{0}' % (s,c+1) )

    return rcams

