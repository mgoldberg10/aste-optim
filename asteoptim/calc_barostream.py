from gcmfaces_py import gcmfaces, calc_barostream, convert2array
import numpy as np
from asteoptim.tracer import *
from ecco_v4_py.llc_array_conversion import llc_tiles_to_faces
import xarray as xr

class AsteGrid:
    def __init__(self,
                nx = 270,
                nz = 50,
                nFaces = 5,
                missVal = 0,
                ):
        
        self.nx, self.ny = self.facesExpand = (nx, 3*nx)
        self.nFaces = nFaces
        self.missVal = missVal
        self.facesSize = np.array([[nx, int(nx * 5 / 3)], [0, 0], [nx, nx], [int(nx * 2 / 3), nx], [int(nx * 5 / 3), nx]])

class AsteBarostream:
    def __init__(self,
            aste_ds,
            grid_dir,
            aste_grid = AsteGrid(),
            nx = 270,
            ):

        self.aste_ds = aste_ds
        self.grid_dir = grid_dir
        self.aste_grid = aste_grid
        self.nx = self.aste_grid.nx


    def add_to_mygrid(self, grid_fld_list=['XC', 'YC', 'dxC', 'dyC', 'hFacC']):
        for grid_fld in grid_fld_list:
            print(grid_fld)          
            gf = self.ds[grid_fld].compute()

            if grid_fld == 'hFacC':
                gf = gf.where(gf != 0.)
                gf = gf[0]
                grid_fld = 'mskC0'
            else:
                grid_fld = grid_fld.upper()
                
            gf = llc_tiles_to_faces(gf.values, less_output=True)
            gf = {iF: gf[iF].T for iF in gf}
            gf = gcmfaces(gf)
            setattr(self.aste_grid, grid_fld, gf)

    def get_maskWS(self):
        for suffix in ['W', 'S', 'C']:
            self.aste_ds[f'mask{suffix}'] = np.ceil(self.aste_ds[f'hFac{suffix}'])

    def calc_one_barostream(self, ioptim, time, domain='aste'):
        ds_tmp = self.aste_ds.isel(ioptim=ioptim, time=time)

        if ('maskW' not in ds_tmp.coords) or ('maskS' not in ds_tmp.coords):
            self.get_maskWS()

        fldU = ((ds_tmp.UVELMASS * ds_tmp.dyG * ds_tmp.drF).sum(axis=0) * ds_tmp.maskW[0])
        fldU = fldU.where(fldU != 0., other=np.nan)
        fldV = ((ds_tmp.VVELMASS * ds_tmp.dxG * ds_tmp.drF).sum(axis=0) * ds_tmp.maskS[0])
        fldV = fldV.where(fldV != 0., other=np.nan)


        if domain == 'aste': # replace with rebuild_llc_facets

            def get_aste_faces(fld):
                from xmitgcm.utils import rebuild_llc_facets, get_extra_metadata
                aste_extra_metadata = get_extra_metadata(domain='aste', nx=self.aste_grid.nx)
                fld = rebuild_llc_facets(fld.rename({'tile': 'face'}), extra_metadata=aste_extra_metadata)
                from pdb import set_trace;set_trace()
                
                fld_dict = {iF+1: fld[facet_key].values for iF, facet_key in enumerate(fld)}

                return fld_dict
            fldU = get_aste_faces(fldU)
            fldV = get_aste_faces(fldV)
            # fldU_compact = aste_tracer2compact(fldU.at().values)
            # fldV_compact = aste_tracer2compact(fldV.at().values)
            # # inputs to aste_compact_to_global270 need to have a singleton dimension
            # fldU = aste_compact_to_global270(fldU_compact, self.aste_grid, return_faces=True)
            # fldV = aste_compact_to_global270(fldV_compact, self.aste_grid, return_faces=True)

        elif domain == 'ecco':
            fldU = llc_tiles_to_faces(fldU.values, less_output=True)
            fldV = llc_tiles_to_faces(fldV.values, less_output=True)

        fldU = {iF: fldU[iF].T for iF in fldU}
        fldV = {iF: fldV[iF].T for iF in fldV}

        psi = calc_barostream(fldU, fldV, self.aste_grid, noDiv=False)
        psi = np.squeeze(convert2array(psi, self.aste_grid)).T

        return psi

    def calc_barostreams(self):

        nx_glob = self.nx * 4
        opts = self.aste_ds.ioptim.values
        nopt = len(opts)
        nt = len(self.aste_ds.time)

        psis = np.empty((nopt, nt, nx_glob, nx_glob))

        # antithesis of xarray right here
        print('Computing barostream for all (optim, time)\n')

        for ioptim in range(nopt):
            for time in range(nt):
                print(f'{ioptim},{time} ', end='')
                psis[ioptim, time, :, :] = self.calc_one_barostream(ioptim=ioptim, time=time)

            
        psis_tracer = xr.DataArray(aste_global2tracer(psis), dims=['ioptim','time','J','I'])
        psis_da = get_aste_tracer_xr_inv(psis_tracer)
        self.aste_ds['psi'] = xr.DataArray(psis_da.values, dims=psis_tracer.dims)

