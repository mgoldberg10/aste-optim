import xmitgcm
from smartcables import *
from .dataset import open_astedataset
from .tracer import get_aste_tracer, get_aste_tracer_xr_inv
from .plot import aste_orthographic

# First pass being written to only consider 2d time varying controls

def l2norm(x):return np.sqrt(np.nansum(np.square(x)))

def mycos(a, b):
    dims = ('ictrl', 'tile', 'j', 'i')
    dot_product = (a * b).sum(dim=dims)
    a_norm = np.sqrt((a**2).sum(dim=dims))
    b_norm = np.sqrt((b**2).sum(dim=dims))
    cos_angle = dot_product / (a_norm * b_norm)
    angle_radians = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians).values
    return angle_degrees

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
        q = xr.concat([self.ds[var] for var in data_vars], dim='ictrl')
        q = q.stack(z=('ictrl', ) + self.ds.XC.dims)  
    
        # Apply transformation
        q = operation(q)
    
        # Unstack and assign back
        for var, q_transformed in zip(data_vars, q.unstack('z')):
            self.ds[var] = q_transformed.reset_coords(drop=True)
    
    def normalize(self):
        return self._vars_operate(lambda q: q / np.sqrt((q**2).sum()))

    def get_relcons(self, prefix, iobs=0, plot=False):
        isel_dict = dict() if 'iobs' not in self.ds.dims else dict(iobs=iobs)
        vec_vars = [var for var in self.ds.data_vars if var.startswith(prefix)]
        squared_norms = {var: (self.ds.isel(isel_dict)[var] ** 2).sum().values for var in vec_vars}
        total_norm_sq = sum(squared_norms.values())
        self.relative_contributions = {var: squared_norm / total_norm_sq for var, squared_norm in squared_norms.items()}

        if plot: 
            labels, values = zip(*self.relative_contributions.items())
            num_colors = len(labels)
            colors = plt.cm.viridis(np.linspace(0, 1, num_colors))
            # Plot the stacked bar chart
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.bar(['Total'], values, bottom=np.cumsum(values) - values, color=colors, width=0.1)
            
            # Add legend
            legend_labels = [f'{label}: {round(value*100)}%' for label, value in zip(labels, values)]  # Format values as .2f
            for label, color, in zip(legend_labels, colors):
                ax.bar(0, 0, color=color, label=label)  # Invisible bars for legend

            # Set yticks to show percentages
            yticks = np.linspace(0, 1, 6) 
            ax.set_yticks(yticks)
            ax.set_yticklabels([f'{int(tick * 100)}%' for tick in yticks])  # Format as percentage

            ax.set_ylabel('Relative Contribution')
            ax.set_title('Stacked Bar Chart of Control Contributions')

            ax.legend()
            plt.show()

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
 
        N, M = (len(self.ctrl_vars), len(self.ds.iobs))
        ntile, nx, ny  = self.ds.XC.shape
        ngrid = ntile*nx*ny

        A = np.zeros((M, N, ntile*nx*ny))

        for iobs in range(M):
            for ictrl, adxx_var in enumerate(self.adxx_vars):
                A[iobs, ictrl, :] = self.ds.isel(iobs=iobs)[adxx_var].values.ravel()
        
        A = A.reshape(A.shape[:-2] + (-1,)).T
        Q, R = np.linalg.qr(A)
        E = np.zeros((M, M))

        # observational noise, not the same as uncertainty!
        for m in range(M):
#            E[m, m] = self.obs_weights[m]**2
            E[m, m] = 1
        
        lhs = R @ np.linalg.inv(E) @ R.T
        D, V = np.linalg.eig(lhs)
        lhs_reconstr = V @ np.diag(D) @ np.linalg.inv(V)
        assert np.allclose(lhs, lhs_reconstr, atol=1e-6)
        QV = Q @ V
        # ON check
        assert np.allclose(QV.T @ QV, np.eye(QV.shape[1]), atol=1e-6)

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
        self.nobs = len(self.obs_run_dirs)
        self.nctrl = len(self.ctrl_vars)
        self.subset_time = subset_time
        self.extra_metadata = xmitgcm.utils.get_extra_metadata(domain='aste', nx=self.nx)

        if obs_weights is not None:
            if len(obs_weights) < len(self.obs_run_dirs):
                raise ValueError(f'Supply more obs_weights. You gave {len(self.obs_run_dirs)} obs dirs but only {len(obs_weights)} obs weights.')
        else:
            obs_weights = np.ones(self.nobs)

        self.obs_weights = obs_weights
        self.qoi_weight = qoi_weight

        # load qoi and obs datasets         
        self.get_datasets()

        # compute DPP
        self.compute_dpp(verbose=self.verbose)

    def get_datasets(self):
        qoi = SensitivityVector(self.qoi_run_dir, grid_dir=self.grid_dir, ctrl_vars=self.ctrl_vars, ctrl_weights = self.ctrl_weights, obs_weights = self.qoi_weight, nx=self.nx, subset_time=self.subset_time)

        qoi.normalize()

        obs_ds_list = []

        for obs_run_dir, obs_weight in zip(self.obs_run_dirs, self.obs_weights):
            obs = SensitivityVector(obs_run_dir, grid_dir=self.grid_dir, ctrl_vars=self.ctrl_vars, ctrl_weights = self.ctrl_weights, obs_weights=np.array([obs_weight]), nx=self.nx, subset_time=self.subset_time)
            obs_ds_list.append(obs.ds)
        obs_ds = xr.concat(obs_ds_list, dim='iobs')
        setattr(obs, 'ds', obs_ds)
        setattr(obs, 'obs_weights', self.obs_weights)

        obs.compute_hessian_eigenvectors()

        self.ds = obs.ds
        for adxx_var in self.adxx_vars:
            self.ds[f'qoi_{adxx_var}'] = qoi.ds[adxx_var]

        # save sensitivity vector objects
        self.obs = obs
        self.qoi = qoi

    def compute_dpp(self, verbose=True):
        mask = self.ds.hFacC[0]

        # stack obs_vars -- already has iobs dimension
        obs_vars = [var for var in self.ds.data_vars if var.startswith('qv')]
        v1 = xr.concat([self.ds[var].where(self.ds.hFacC[0]) for var in obs_vars], dim='ictrl')

        # stack qoi_vars -- needs to be tiled iobs times
        qoi_vars = [var for var in self.ds.data_vars if var.startswith('qoi')]
        q = xr.concat([self.ds[var].where(self.ds.hFacC[0]) for var in qoi_vars], dim='ictrl')
       
        if 'iobs' not in q.dims:
            q = q.expand_dims(iobs=self.nobs)
        q = q.transpose(*v1.dims)

        dims_to_sum = tuple(dim for dim in v1.dims if dim != "iobs")
        dpp_obs = (q * v1).sum(dim=dims_to_sum)**2  # Sum over all but 'iobs'
         
        dpp_tot = dpp_obs.sum(dim="iobs")

        self.dpp_obs = dpp_obs.values
        self.dpp_tot = dpp_tot.values
        self.v1 = v1
        self.q = q

        if verbose:
            dpp_percent = [f"{1e2 * dpp.values:.0f}%" for dpp in dpp_obs]
            print(" + ".join(f"dpp{i}" for i in range(len(dpp_obs))) + " = " +
                  " + ".join(dpp_percent) + f" = {1e2*self.dpp_tot:.0f}%")


    def get_angles(self, verbose=True):
        dims = ('ictrl', 'tile', 'j', 'i')
        self.angles_degrees = mycos(self.q, self.v1)
        if verbose:
            angles_degrees = [f"{angle:.0f}°" for angle in self.angles_degrees]
            angle_str = " + ".join([f"theta{i}" for i in range(len(angles_degrees))]) + " = "
            angle_values_str = " + ".join(angles_degrees)
            print(f"{angle_str}{angle_values_str}")

    def plot(self, iobs=0, dotprod_fac=100, maskC_fname=None, ao_kwargs={}):
        das = []
        for var in self.adxx_vars:
            qoi = getattr(self.ds, f'qoi_{var}').where(self.ds.hFacC[0])
            qv = getattr(self.ds, f'qv_{var}')[iobs].where(self.ds.hFacC[0])
            dotprod = qoi * qv
            das.extend([qoi, qv, dotprod])
        
        # get sensor/observation coordinates
        if maskC_fname is not None:
            mask_path = glob.glob(self.obs_run_dirs[iobs] + f'*{maskC_fname}')[0]
            mask = get_aste_tracer(read_float32(mask_path).reshape(5*self.nx, self.nx), self.nx)[0]
            mask = xr.DataArray(mask, dims=['J', 'I'])
            mask = get_aste_tracer_xr_inv(mask)
            mask_lon = self.ds.XC.where(mask!=0).stack(z=self.ds.XC.dims).dropna('z').values
            mask_lat = self.ds.YC.where(mask!=0).stack(z=self.ds.XC.dims).dropna('z').values
    
            if len(mask_lat) > 1: 
                mask_lat = mask_lat.mean()
                mask_lon = mask_lon.mean()
        
        fig, axes = aste_orthographic(subplot_n=self.nctrl, subplot_m=3, **ao_kwargs)
        
        titles = [
            r'${\bf q} = \frac{d(v^{\perp}_{ISR})}{d(u_{\mathrm{wind}})}$',
            r'${\bf v}_1 = \frac{d(p_b)}{d(u_{\mathrm{wind}})}$',
            r'${\bf q} \cdot {\bf v}_1$',
            r'${\bf q} = \frac{d(v^{\perp}_{ISR})}{d(v_{\mathrm{wind}})}$',
            r'${\bf v}_1 = \frac{d(p_b)}{d(v_{\mathrm{wind}})}$',
            r'${\bf q} \cdot {\bf v}_1$',
        ]
        
        facs = [1, 1, dotprod_fac] * self.nctrl

        for da, ax, title, fac in zip(das, axes.ravel(), titles, facs):
        
            nlev = 11;vmax = 0.05;vmin = -vmax;cmap=cmap_utils.custom_div_cmap(nlev);levels=np.linspace(vmin, vmax, nlev)
            plot_kwargs = dict(vmin=vmin, vmax=vmax, cmap=cmap, show_cbar=False)#, extend='both', levels=levels)
            if 'cdot' in title:
                title += f' * {fac}'
        
            ax.set_title(title, fontsize=30, pad=30)
        
            _, ax, _, p = self.ds.c.plotpc(da*fac, ax=ax, **plot_kwargs)
        
            if da.name is not None and da.name.startswith('qv'):
                ax.scatter(mask_lon, mask_lat, c='yellow',
                             edgecolors='black', s=100, linewidth=1.5,
                             transform=ccrs.PlateCarree()
                            )
        
        cbar_ax = fig.add_axes([0.15, -0.1, 0.7, 0.04])  # Adjust values as needed
        fig.colorbar(p, cax=cbar_ax, orientation='horizontal')
        cbar_ax.tick_params(size=20, labelsize=40)
        fig.subplots_adjust(wspace=-0.8)  # Reduce horizontal spacing
        fig.tight_layout()
        fig.suptitle(f'DPP={100 * self.dpp_obs[iobs]:.2f}%', fontsize=100, y=1.1)
        return fig, ax

