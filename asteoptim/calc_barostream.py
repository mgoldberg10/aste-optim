from gcmfacespy import gcmfaces, calc_barostream, convert2array
import numpy as np
from asteoptim.tracer import *
from ecco_v4_py.llc_array_conversion import llc_tiles_to_faces, llc_faces_to_tiles
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
            aste_grid = AsteGrid(),
            nx = 270,
            domain = 'aste',
            noDiv = False
            ):

        self.aste_ds = aste_ds
        self.aste_grid = aste_grid
        self.nx = self.aste_grid.nx
        self.domain = domain
        self.noDiv = noDiv


    def add_to_mygrid(self, grid_fld_list=['XC', 'YC', 'hFacC']):
        if self.noDiv:
            grid_fld_list += ['dxC', 'dyC']
        for grid_fld in grid_fld_list:
            print(grid_fld)          
            gf = self.aste_ds[grid_fld].compute()

            if grid_fld == 'hFacC':
                gf = gf.where(gf != 0.)
                gf = gf[0]
                grid_fld = 'mskC0'
            else:
                grid_fld = grid_fld.upper()
                
            if self.domain == 'ecco':
                gf = llc_tiles_to_faces(gf.values, less_output=True)
            elif self.domain == 'aste':
                gf = aste_tracer2compact(gf.at().values)
                gf = aste_compact_to_global270(gf, self.aste_grid, return_faces=True)
                if grid_fld == 'mskC0':
                    for iF in gf:
                        gf[iF][gf[iF] == 0.] = np.nan

            gf = {iF: gf[iF].T for iF in gf}
            gf = gcmfaces(gf)
            setattr(self.aste_grid, grid_fld, gf)

    def get_maskWS(self):
        for suffix in ['W', 'S']:
            self.aste_ds[f'mask{suffix}'] = np.ceil(self.aste_ds[f'hFac{suffix}'])

    def calc_one_barostream(self, ioptim, time, domain='aste'):
        ds_tmp = self.aste_ds.isel(ioptim=ioptim, time=time)

        if ('maskW' not in ds_tmp.coords) or ('maskS' not in ds_tmp.coords):
            self.get_maskWS()

        # adding special logic for llc4320 fields, which are constructed ahead of time
        if 'fldU' not in ds_tmp.data_vars.keys():
            fldU = ((ds_tmp.UVELMASS * ds_tmp.dyG * ds_tmp.drF).sum(axis=0) * ds_tmp.maskW[0])
        else:
            fldU = ds_tmp.fldU
        if 'fldV' not in ds_tmp.data_vars.keys():
            fldV = ((ds_tmp.VVELMASS * ds_tmp.dxG * ds_tmp.drF).sum(axis=0) * ds_tmp.maskS[0])
        else:
            fldV = ds_tmp.fldV

        fldU = fldU.where(fldU != 0., other=np.nan)
        fldV = fldV.where(fldV != 0., other=np.nan)


        if self.domain == 'aste': # replace with rebuild_llc_facets
            fldU_compact = aste_tracer2compact(fldU.at().values)
            fldV_compact = aste_tracer2compact(fldV.at().values)
            # inputs to aste_compact_to_global270 need to have a singleton dimension
            fldU = aste_compact_to_global270(fldU_compact, self.aste_grid, return_faces=True)
            fldV = aste_compact_to_global270(fldV_compact, self.aste_grid, return_faces=True)

        elif self.domain == 'ecco':
            fldU = llc_tiles_to_faces(fldU.values, less_output=True)
            fldV = llc_tiles_to_faces(fldV.values, less_output=True)

        fldU = {iF: fldU[iF].T for iF in fldU}
        fldV = {iF: fldV[iF].T for iF in fldV}

        psi = calc_barostream(fldU, fldV, self.aste_grid, noDiv=self.noDiv)

        if self.domain == 'aste':
            psi = np.squeeze(convert2array(psi, self.aste_grid)).T
            psis_tracer = xr.DataArray(aste_global2tracer(psi), dims=['J','I'])
            psi_tiles = get_aste_tracer_xr_inv(psis_tracer).values
        elif self.domain == 'ecco':
            psi = {iF: psi[iF].T for iF in psi}
            psi_tiles = llc_faces_to_tiles(psi, less_output=True)

        return psi_tiles

    def calc_barostreams(self):

        opts = self.aste_ds.ioptim.values
        nopt = len(opts)
        nt = len(self.aste_ds.time)
        ntile = len(self.aste_ds.tile)

        psis = np.empty((nopt, nt, ntile, self.nx, self.nx))

        # antithesis of xarray right here
        print('Computing barostream for all (optim, time)\n')

        for ioptim in range(nopt):
            for time in range(nt):
                print(f'{ioptim},{time} ', end='')
                psis[ioptim, time, :, :, :] = self.calc_one_barostream(ioptim=ioptim, time=time)

        self.aste_ds['psi'] = xr.DataArray(psis, dims=['ioptim', 'time', 'tile', 'j', 'i'])
