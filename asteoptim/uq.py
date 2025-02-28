import xmitgcm
from smartcables import *
from .dataset import open_astedataset

# First pass being written to only consider 2d time varying controls

def l2norm(x):return np.sqrt(np.nansum(np.square(x)))

class SensitivityVector():
    def __init__(
        self,
        run_dir, 
        grid_dir = None, 
        ctrl_vars = [],
        ctrl_weights = [],
        nx = 270,
        subset_time = slice(None, None),
        obs_weights = np.array([1.]),
        time_mean = True,
    ):
        self.run_dir = run_dir
        self.grid_dir = grid_dir if grid_dir is not None else self.run_dir
        self.ctrl_vars = ctrl_vars
        self.adxx_vars = [f'adxx_{cv}' for cv in self.ctrl_vars]
        self.ctrl_weights = ctrl_weights
        self.obs_weights = obs_weights
        self.nx = nx
        self.extra_metadata = xmitgcm.utils.get_extra_metadata(domain='aste', nx=self.nx)
        self.subset_time = subset_time
        self.time_mean = time_mean
 
        
        self.load_sensitivities()

    def load_sensitivities(self):
        iternum = 0 # feels like an okay assumption for dpp experiments -- no assimilation, just sensitivity
        # load dataset, just the grid
        self.ds = open_astedataset(self.run_dir, nx=self.nx, grid_dir=self.grid_dir)

        for adxx_var, ctrl_weight in zip(self.adxx_vars, self.ctrl_weights):
            adxx_meta = get_aste_file_metadata(self.run_dir + adxx_var, iternum=iternum, dtype=np.dtype('>f4'), extra_metadata=self.extra_metadata)
            adxx_data = read_3d_llc_data(adxx_var, adxx_meta)
            adxx_data *= ctrl_weight
            self.ds[adxx_var] = adxx_data

        self.ds = self.ds.isel(time=self.subset_time)
        if self.time_mean:
            self.ds = self.ds.mean('time')

    def _vars_operate(self, operation):
        data_vars = [var for var in self.ds.data_vars if var.startswith('adxx_')]
        q = xr.concat([self.ds[var].where(self.ds.hFacC[0]) for var in data_vars], dim='ictrl')
        q = q.stack(z=('ictrl', ) + self.ds.XC.dims)  # Stack over ictrl and spatial dims
        q = operation(q)  # Apply transformation
        for var, q_transformed in zip(data_vars, q.unstack('z')):
            self.ds[var] = q_transformed.reset_coords(drop=True)
    
    def nondimensionalize(self):
        self._vars_operate(lambda q: q * self.obs_weights)
    
    def normalize(self):
        self._vars_operate(lambda q: q / np.sqrt((q**2).sum()))

    def compute_hessian_eigenvectors(self):
        """
        Compute the orthonormal eigenvectors of the (Gauss-Newton approximation of the) Hessian matrix.
    
        Parameters:
            A (numpy.ndarray): Array containing gradients of size nx*ny for each of M observations for each of N controls. Shape is [M, N, nx*ny].
    
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
        
        if 'iobs' not in self.ds.dims:         
            self.ds = self.ds.expand_dims('iobs')
 
        N, M = (len(self.ctrl_vars), len(self.obs_weights))
        ntile, nx, ny  = self.ds.XC.shape
        ngrid = ntile*nx*ny

        A = np.zeros((M, N, ntile*nx*ny))

        for iobs in range(M):
            for ictrl, adxx_var in enumerate(self.adxx_vars):
                A[iobs, ictrl, :] = self.ds.isel(iobs=iobs)[adxx_var].values.ravel()
        
        A = A.reshape(A.shape[:-2] + (-1,)).T
        Q, R = np.linalg.qr(A)
#        sign_correction = np.sign(np.sum(A * Q, axis=0))  # Ensures alignment with A
#        Q *= sign_correction  # Apply sign correction to Q
#        R *= sign_correction[:, np.newaxis] 
    
        E = np.zeros((M, M))
        
        for m in range(M):
            E[m, m] = self.obs_weights[m]**2
        
        lhs = np.dot(np.dot(R, np.linalg.inv(E)), R.T)
        D, V = np.linalg.eig(lhs)
        
        QV = np.matmul(Q, V)
        QV = np.transpose(QV.reshape(N, ngrid, M), [2, 0, 1])

        dims = ('iobs', ) + self.ds.XC.dims
        shape = (M, ) + self.ds.XC.shape

        for ictrl, adxx_var in enumerate(self.adxx_vars):
            self.ds[f'qv_{adxx_var}'] = xr.DataArray(QV[:, ictrl, :].reshape(shape), dims=dims)


class DPP():
    def __init__(
        self,
        qoi_run_dir,
        obs_run_dirs,
        verbose = True,
        grid_dir = None, 
        ctrl_vars = [],
        ctrl_weights = [],
        nx = 270,
        subset_time = slice(None, None),
        obs_weights = None,
        qoi_weight = None,
    ):

        # set attributes
        self.qoi_run_dir = qoi_run_dir
        self.obs_run_dirs = obs_run_dirs
        self.verbose = verbose

        self.grid_dir = grid_dir if grid_dir is not None else self.run_dir
        self.ctrl_vars = ctrl_vars
        self.adxx_vars = [f'adxx_{cv}' for cv in self.ctrl_vars]
        self.ctrl_weights = ctrl_weights
        self.nx = nx
        self.subset_time = subset_time
        self.extra_metadata = xmitgcm.utils.get_extra_metadata(domain='aste', nx=self.nx)

        self.obs_weights = np.ones(len(obs_run_dirs)) if obs_weights is None else obs_weights
        self.qoi_weight = qoi_weight

        # load qoi and obs datasets         
        self.get_datasets()

        # compute DPP
        self.compute_dpp(verbose=self.verbose)

    def get_datasets(self):
        qoi = SensitivityVector(self.qoi_run_dir, grid_dir=self.grid_dir, ctrl_vars=self.ctrl_vars, ctrl_weights = self.ctrl_weights, obs_weights = self.qoi_weight, nx=self.nx, subset_time=self.subset_time)

        # divide by qoi uncertainty
        qoi._vars_operate(lambda q: q / qoi.obs_weights)
        # multiply by ctrl uncertainty
#        qoi._vars_operate(lambda q: q / self.ctrl_weights) # need to make this work with list of ctrl_weights
        
        # multiply out uncertainty in qoi/obs
        qoi.normalize()

        obs_ds_list = []

        for obs_run_dir, obs_weight in zip(self.obs_run_dirs, self.obs_weights):
            obs = SensitivityVector(obs_run_dir, grid_dir=self.grid_dir, ctrl_vars=self.ctrl_vars, ctrl_weights = self.ctrl_weights, obs_weights=np.array([obs_weight]), nx=self.nx, subset_time=self.subset_time)
            obs_ds_list.append(obs.ds)
        obs_ds = xr.concat(obs_ds_list, dim='iobs')
        setattr(obs, 'ds', obs_ds)
        obs.compute_hessian_eigenvectors()

        self.ds = obs.ds
        for adxx_var in self.adxx_vars:
            self.ds[f'qoi_{adxx_var}'] = qoi.ds[adxx_var]

    def compute_dpp(self, verbose=True):
        # need to concat all adxx_vars, hard coded rn
        mask = self.ds.hFacC[0]

        # stack obs_vars -- already has iobs dimension
        obs_vars = [var for var in self.ds.data_vars if var.startswith('qv')]
        v1 = xr.concat([self.ds[var].where(self.ds.hFacC[0]) for var in obs_vars], dim='ictrl')
        v1 = v1.stack(z=('ictrl', ) + mask.dims)

        # stack qoi_vars -- needs to be tiled iobs times
        qoi_vars = [var for var in self.ds.data_vars if var.startswith('qoi')]
        q = xr.concat([self.ds[var].where(self.ds.hFacC[0]) for var in qoi_vars], dim='ictrl')
        q = q.stack(z=('ictrl', ) + mask.dims)
       
        # tile q iobs times
        if 'iobs' not in q.dims:
            q = q.expand_dims(iobs=len(v1.iobs))
        q = q.transpose('z', 'iobs').values.reshape(-1)
        q = xr.DataArray(q, dims='z')

        # flatten v1
        v1 = v1.transpose('z', 'iobs').values.reshape(-1)
        v1 = xr.DataArray(v1, dims='z')

        # I think it works like so: since the qv vectors are ON, we just repeat 
        # q M times... do we need to renormalize q after repeating?
        # Dont think we do...as long as q and v_i are ON, we're good

        dpp_val = (v1 * q).sum()**2
        if verbose:
            print(f'dpp = {1e2*dpp_val.values:.2f}%')
        self.dpp_val = dpp_val
