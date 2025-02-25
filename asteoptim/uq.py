import xmitgcm
from smartcables import *
from .dataset import open_astedataset

# First pass being written to only consider 2d time varying controls


class SensitivityVector():
    def __init__(
        self,
        run_dir, 
        grid_dir = None, 
        ctrl_vars = [],
        ctrl_weights = [],
        nx = 270,
        subset_time = slice(None, None),
    ):
        self.run_dir = run_dir
        self.grid_dir = grid_dir if grid_dir is not None else self.run_dir
        self.ctrl_vars = ctrl_vars
        self.adxx_vars = [f'adxx_{cv}' for cv in self.ctrl_vars]
        self.ctrl_weights = ctrl_weights
        self.nx = nx
        self.extra_metadata = xmitgcm.utils.get_extra_metadata(domain='aste', nx=self.nx)
        self.subset_time = subset_time
        
        self.load_datasets()

    def load_datasets(self):
        iternum = 0 # feels like an okay assumption for dpp experiments -- no assimilation, just sensitivity
        # load dataset, just the grid
        self.ds = open_astedataset(self.run_dir, nx=self.nx, grid_dir=self.grid_dir)

        for adxx_var, ctrl_weight in zip(self.adxx_vars, self.ctrl_weights):
            adxx_meta = get_aste_file_metadata(self.run_dir + adxx_var, iternum=iternum, dtype=np.dtype('>f4'), extra_metadata=self.extra_metadata)
            adxx_data = read_3d_llc_data(adxx_var, adxx_meta)
            adxx_data *= ctrl_weight
            self.ds[adxx_var] = adxx_data

        self.ds = self.ds.isel(time=self.subset_time)

    def compute_hessian_eigenvectors(self, obs_weights):
        """
        Compute the orthonormal eigenvectors of the (Gauss-Newton approximation of the) Hessian matrix.
    
        Parameters:
            A (numpy.ndarray): Array containing gradients of size nx*ny for each of M observations for each of N controls. Shape is [M, N, nx*ny].
            obs_weights (numpy.ndarray): Array containing observation weights. Length is M.
    
        Returns:
            QV (numpy.ndarray): Array containing orthonormal eigenvectors of size nx*ny for each observation and control.
                           Shape is (M, N, nx*ny).
    
        Description:
            This function computes the orthonormal eigenvectors of the Hessian matrix, which represents the second
            derivatives of simulated observations with respect to controls. It first reshapes the input array A
            to size [N, M, nx*ny] and performs QR decomposition. Then, it constructs a diagonal matrix E using the
            observation weights. The Hessian matrix is approximated using the equation: H = R * inv(E) * R^T, where R
            is the upper triangular factor of the QR decomposition. The eigenvectors of H are computed, and the orthonormal
            eigenvectors are obtained by multiplying Q and V, where Q is the orthogonal matrix from the QR decomposition
            and V contains the eigenvectors. The resulting array QV is reshaped to size (M, N, nx*ny) and returned.
        """

        self.obs_weights = obs_weights

        N, M = (len(self.ctrl_vars), len(self.obs_weights))
        ntile, nx, ny  = self.ds.XC.shape
        ngrid = ntile*nx*ny

        A = np.zeros((M, N, ntile*nx*ny))

        for iobs in range(M):
            for ictrl, adxx_var in enumerate(self.adxx_vars):
                A[iobs, ictrl, :] = self.ds[adxx_var].mean('time').values.ravel()

        
        A = A.reshape(A.shape[:-2] + (-1,)).T
        Q, R = np.linalg.qr(A)
    
        E = np.zeros((M, M))
        
        for m in range(M):
            E[m, m] = self.obs_weights[m]**2
        
        lhs = np.dot(np.dot(R, np.linalg.inv(E)), R.T)
        D, V = np.linalg.eig(lhs)
        
        QV = np.matmul(Q, V)
        QV = np.transpose(QV.reshape(N, ngrid, M), [2, 0, 1])

        for iobs in range(len(self.obs_weights)):
            for ictrl, adxx_var in enumerate(self.adxx_vars):
                self.ds[f'qv_{adxx_var}'] = xr.DataArray(QV[iobs, ictrl, :].reshape(self.ds.XC.shape), dims=self.ds.XC.dims)

class DPP():
    def __init__(
        self,
        sv_qoi,
        sv_obs,
        verbose = True
    ):
        self.sv_qoi = sv_qoi
        self.sv_obs = sv_obs
        self.verbose = verbose
         
        self.compute_dpp(verbose=self.verbose)

    def compute_dpp(self, verbose=True):
        # need to concat all adxx_vars, hard coded rn
        mask = self.sv_qoi.ds.hFacC.isel(k=0)
        
        q = xr.concat(
            [self.sv_qoi.ds.adxx_uwind.mean('time').where(mask), 
             self.sv_qoi.ds.adxx_vwind.mean('time').where(mask)], 
            dim='ictrl'
        ).stack(z=('ictrl', ) + self.sv_qoi.ds.XC.dims)  # Stack only over ictrl and spatial dims
        q /= np.sqrt((q**2).sum())
        
        v1 = xr.concat(
            [self.sv_obs.ds.qv_adxx_uwind.where(mask), 
             self.sv_obs.ds.qv_adxx_vwind.where(mask)], 
            dim='ictrl'
        ).stack(z=('ictrl', ) + self.sv_qoi.ds.XC.dims)
        
        dpp_val = (v1 * q).sum()**2
        if verbose:
            print(f'dpp = {1e2*dpp_val.values:.2f}%')
