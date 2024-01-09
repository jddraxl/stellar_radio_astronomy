import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import math
import json
import pickle
import pyphot
import inspect
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from typing import Sequence
from pyphot import (unit, Filter)
import scipy.interpolate as interp
import astropy.coordinates as coord
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
import scipy.stats as sp

#  
# inspect()
# 

print('''Functions useful for my radio astronomy reasearch''')

filters = {'umag':3543, 'gmag': 4770, 'rmag':6231, 'imag': 7625, 'zmag': 9134,'Ymag':10090,'Jmag':12360,'ch1mag':36000,'ch2mag':44930}
bands   = {'uerr': [3120,4000], 'gerr': [3980,5480], 'rerr':[5680,7160], 'ierr': [7100,8560], 'zerr': [8500,10020],'Yerr':[9530,10650],'Jerr':[11430,13290],'ch1err':[31760,39260],'ch2err':[39880,49980]}


def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def Simbad_desc():
    return('''Simbad.get_votable_fields() indicates what fields are being used
    Simbad.list_votable_fields() is a list of all available fields
    Simbad.add_votable_fields() add fields/columns that will be printed
    Simbad.remove_votable_fields() removes fields to be displayed
    Simbad.get_field_description() describes the field/column
    Some available filternames are B,V,R,I,J,K''')

def duplicates(list_):
    uniques = []
    dupes = []
    for x in list_:
        if x in uniques:dupes.append(x)
        else:uniques.append(x)
    print('There are ' + str(len(dupes)) + ' duplicates')

def coordConv(RA,DEC,RAin='d',RAout='h',DECin='d',DECout='d'):
    RAh = math.floor(RA/15)
    RArem = RA - RAh*15
    RAm = math.floor(RArem*4)
    RArem = RArem*4 - RAm
    RAs = round(RArem*60)

    DECd = math.floor(DEC)
    if DEC<0:
        DECd += 1
    DECrem = np.abs(DEC - DECd)
    DECm = math.floor(DECrem*60)
    DECrem = DECrem*60 - DECm
    DECs = round(DECrem*60)

    string = str(RAh)+RAout+str(RAm)+'m'+str(RAs)+'s '+str(DECd)+DECout+str(DECm)+'m'+str(DECs)+'s'
    
    return string

def RAConv(RA,RAin='d',RAout='h',DECin='d',DECout='d'):
    RAh = math.floor(RA/15)
    RArem = RA - RAh*15
    RAm = math.floor(RArem*4)
    RArem = RArem*4 - RAm
    RAs = round(RArem*60)

    string = str(RAh)+':'+str(RAm)+':'+str(RAs)
    
    return string

def DecConv(DEC,RAin='d',RAout='h',DECin='d',DECout='d'):

    DECd = math.floor(DEC)
    if DEC<0:
        DECd += 1
    DECrem = np.abs(DEC - DECd)
    DECm = math.floor(DECrem*60)
    DECrem = DECrem*60 - DECm
    DECs = round(DECrem*60)

    string = str(DECd)+':'+str(DECm)+':'+str(DECs)
    
    return string

def magAB2flux(mAB,units='Jy'):
    if units=='Jy':
        K = 8.9
    elif units=='cgs':
        K = -48.6
    else:
        raise ValueError('Allowed units are Jy and cgs')

    flux = np.power(10,(mAB-K)/(-2.5))

    return flux


def interpolate_flux_wave(
    waveflux: Sequence, flux: Sequence, wavenew: Sequence):
    """
    filterMag function requires interpolation to different wavelengths.
    Function to interpolate the flux from the stars to the wavegrid we are working on.
    Parameters
    ----------
    waveflux : Sequence
        An array specifying wavelength in units of microns of the given star.
    flux : Sequence
        An array specifying flux density in f_lambda units of the given star.
    wavenew : Sequence
        An array specifying wavelength in units of microns on which the star
        will be interpolated.
    Returns
    -------
    interpolated_flux : Sequence
        An array with the interpolated flux.
    """
    f = interp.interp1d(waveflux, flux, assume_sorted=False, fill_value=0.0)
    return f(wavenew)


def chi_square(flux1, flux2, error1):
    flux1 = np.array(flux1)
    flux2 = np.array(flux2)
    scale = np.nansum((flux1 * flux2) / (error1**2)) / np.nansum((flux2**2) / (error1**2))
    chisquared = np.nansum(
            ((flux1 - (flux2*scale)) ** 2) / (error1**2)
        )
    red_chisquared = chisquared/(len(flux1)-1)
    return red_chisquared


def chi2bestfit(listofstars,starflux,starfluxerror):
    chi_squares=[]
    for i in range(len(listofstars)):     
        chi2 = chi_square(starflux,listofstars[i]['filtersfluxnorm'],starfluxerror)
        chi_squares.append(chi2)
    bestfit = listofstars[np.argmin(chi_squares)]
    bestfit['chi2'] = min(chi_squares)
    return bestfit


def recover_stars(folder_path,wavelimit=14400):
    allstars={}
    # Iterate over folders
    for folder_name in folder_path:
        star_info = []
        # Iterate over the files in the folder
        for filename in os.listdir(folder_name):
            if filename.endswith('.spec.txt'):
                spec_txt_path = os.path.join(folder_name, filename)
                spec_path = os.path.join(folder_name, filename[:-4])
                
                # Extract star name from the filename
                star_name = re.sub(r'\.spec\.txt$', '', filename)
                # print(spec_path)
                
                # Check if the corresponding .spec file exists
                if not os.path.exists(spec_path):
                    print(f'Skipping {star_name}: corresponding .spec file not found')
                    continue
                
                # Read the .spec.txt file
                with open(spec_txt_path, 'r') as file:
                    spec_txt_content = file.read()
                
                # Extract relevant information from the .spec.txt file
                teff_match = re.search(r"Teff\s*=\s*'(\d+)'", spec_txt_content)
                if teff_match:
                    teff = int(teff_match.group(1))
                else:
                    teff = None
                
                logg_match = re.search(r"logg\s*=\s*'\s*(\+*\-*\d+\.*\d*)'", spec_txt_content)
                if logg_match:
                    logg = float(logg_match.group(1))
                else:
                    logg = None
                
                mbol_match = re.search(r"mbol\s*=\s*'\s*(\+*\-*\d+\.*\d*)'", spec_txt_content)
                if mbol_match:
                    mbol = float(mbol_match.group(1).strip())
                else:
                    mbol = None

                metallic_mod_match = re.search(r"metallic_mod\s*=\s*'\s*(\+*\-*\d+\.*\d*)\s*'", spec_txt_content)
                if metallic_mod_match:
                    metallic_mod = float(metallic_mod_match.group(1).strip())
                else:
                    metallic_mod = None
                
                alpha_mod_match = re.search(r"alpha_mod\s*=\s*'\s*(\+*\-*\d+\.*\d*)'", spec_txt_content)
                if alpha_mod_match:
                    alpha_mod = float(alpha_mod_match.group(1).strip())
                else:
                    alpha_mod = None
                
                conv_alpha_match = re.search(r"conv_alpha\s*=\s*'\s*(\+*\-*\d+\.*\d*)'", spec_txt_content)
                if conv_alpha_match:
                    conv_alpha = float(conv_alpha_match.group(1).strip())
                else:
                    conv_alpha = None
                
                turbvel_match = re.search(r"turbvel\s*=\s*'\s*(\+*\-*\d+\.*\d*)'", spec_txt_content)
                if turbvel_match:
                    turbvel = float(turbvel_match.group(1).strip())
                else:
                    turbvel = None
                
                macroturbvel_match = re.search(r"macroturbvel\s*=\s*'\s*(\+*\-*\d+\.*\d*)'", spec_txt_content)
                if macroturbvel_match:
                    macroturbvel = float(macroturbvel_match.group(1).strip())
                else:
                    macroturbvel = None
                
                r_process_mod_match = re.search(r"r_process_mod\s*=\s*'\s*(\+*\-*\d+\.*\d*)'", spec_txt_content)
                if r_process_mod_match:
                    r_process_mod = float(r_process_mod_match.group(1).strip())
                else:
                    r_process_mod = None
                
                s_process_mod_match = re.search(r"s_process_mod\s*=\s*'\s*(\+*\-*\d+\.*\d*)'", spec_txt_content)
                if s_process_mod_match:
                    s_process_mod = float(s_process_mod_match.group(1).strip())
                else:
                    s_process_mod = None
                
                dy_ab_match = re.search(r"Abu_66\s*=\s*'\s*(\+*\-*\d+\.*\d*)'", spec_txt_content)
                if dy_ab_match:
                    dy_ab = float(dy_ab_match.group(1).strip())
                else:
                    dy_ab = None
                
                # Read the .spec file and extract information
                with open(spec_path, 'r') as file:
                    spec_content = file.read()
                
                # Split the content by newline character to get individual lines
                lines = spec_content.split('\n')
                wavelengths = []
                fluxes = []

                # Iterate over the lines and extract values for each column
                for line in lines:
                    if line.strip() == '':
                        continue  # Skip empty lines
                        
                    columns = line.split()  # Split the line by whitespace
                    
                    # Extract values for each column and append to respective lists                    
                    wavelengths.append(float(columns[0]))
                    fluxes.append(float(columns[1]))
                
                cond = np.array(wavelengths)<wavelimit
                wavelengths = list(np.array(wavelengths)[cond])
                fluxes = list(np.array(fluxes)[cond])

                # Store the extracted information for the star
                star_info.append({
                    'star_name': star_name,
                    'teff': teff,
                    'logg': logg,
                    'mbol': mbol,
                    'Fe/H': metallic_mod,
                    'alpha/Fe': alpha_mod,
                    'conv_alpha': conv_alpha,
                    'microturbvel': turbvel,
                    'macroturbvel': macroturbvel,
                    'r_process_mod': r_process_mod,
                    's_process_mod': s_process_mod,
                    'Dy abundance': dy_ab,
                    'wavelength': wavelengths,
                    'flux': fluxes,
                })
        allstars[folder_name]=star_info
    return allstars


def merge_star_data(star_data_list, reverse=True):
    data_list = star_data_list.copy()
    if reverse:
        data_list.reverse()
    merged_data = {}
    
    for star_data in data_list:
        star_name = star_data['star_name']
        wavelength = star_data['wavelength']
        flux = star_data['flux']
        
        # Extract the part of the star name without the ending (_IR, _UV, or _VIS)
        name_without_suffix = star_name.rsplit('_', 1)[0]
        
        if name_without_suffix in merged_data:
            # Merge data with existing entry
            merged_data[name_without_suffix]['wavelength'] += wavelength
            merged_data[name_without_suffix]['flux'] += flux
        else:
            # Create a new entry in the merged_data dictionary
            merged_data[name_without_suffix] = {
                'star_name': name_without_suffix,
                'wavelength': wavelength,
                'flux': flux,
                'teff': star_data['teff'],
                'logg': star_data['logg'],
                'mbol': star_data['mbol'],
                'Fe/H': star_data['Fe/H'],
                'alpha/Fe': star_data['alpha/Fe'],
                'conv_alpha': star_data['conv_alpha'],
                'microturbvel': star_data['microturbvel'],
                'macroturbvel': star_data['macroturbvel'],
                'r_process_mod': star_data['r_process_mod'],
                's_process_mod': star_data['s_process_mod'],
                'Dy abundance': star_data['Dy abundance'],
            }
    
    return list(merged_data.values())


def gettransmition(fname):
    with open('Filters/'+fname+'.res', 'r') as file:
        filter_content = file.read()
    lines = filter_content.split('\n')
    wave = []
    transmit = []
    for line in lines:
        if line.strip() == '':
            continue
        columns = line.split()
        wave.append(float(columns[0]))
        transmit.append(float(columns[1]))
    return np.array(wave), np.array(transmit)


def init_filters():
    wave_u,transmit_u = gettransmition('DECam_u')
    f_u = Filter(wave_u, transmit_u, name='tophat', dtype='photon', unit='Angstrom')
    wave_g,transmit_g = gettransmition('DECam_g')
    f_g = Filter(wave_g, transmit_g, name='tophat', dtype='photon', unit='Angstrom')
    wave_r,transmit_r = gettransmition('DECam_r')
    f_r = Filter(wave_r, transmit_r, name='tophat', dtype='photon', unit='Angstrom')
    wave_i,transmit_i = gettransmition('DECam_i')
    f_i = Filter(wave_i, transmit_i, name='tophat', dtype='photon', unit='Angstrom')
    wave_z,transmit_z = gettransmition('DECam_z')
    f_z = Filter(wave_z, transmit_z, name='tophat', dtype='photon', unit='Angstrom')
    wave_Y,transmit_Y = gettransmition('DECam_Y')
    f_Y = Filter(wave_Y, transmit_Y, name='tophat', dtype='photon', unit='Angstrom')
    wave_J,transmit_J = gettransmition('J_Johnson')
    f_J = Filter(wave_J, transmit_J, name='tophat', dtype='photon', unit='Angstrom')
    return f_u,f_g,f_r,f_i,f_z,f_Y,f_J

wave_u,transmit_u = gettransmition('DECam_u')
f_u = Filter(wave_u, transmit_u, name='tophat', dtype='photon', unit='Angstrom')
wave_g,transmit_g = gettransmition('DECam_g')
f_g = Filter(wave_g, transmit_g, name='tophat', dtype='photon', unit='Angstrom')
wave_r,transmit_r = gettransmition('DECam_r')
f_r = Filter(wave_r, transmit_r, name='tophat', dtype='photon', unit='Angstrom')
wave_i,transmit_i = gettransmition('DECam_i')
f_i = Filter(wave_i, transmit_i, name='tophat', dtype='photon', unit='Angstrom')
wave_z,transmit_z = gettransmition('DECam_z')
f_z = Filter(wave_z, transmit_z, name='tophat', dtype='photon', unit='Angstrom')
wave_Y,transmit_Y = gettransmition('DECam_Y')
f_Y = Filter(wave_Y, transmit_Y, name='tophat', dtype='photon', unit='Angstrom')
wave_J,transmit_J = gettransmition('J_Johnson')
f_J = Filter(wave_J, transmit_J, name='tophat', dtype='photon', unit='Angstrom')


def createmodelslist(path,IR=False):
    models = merge_star_data(recover_stars([path])[path])
    for i in range(len(models)):
        wavelength = np.array(models[i]['wavelength'])
        flux = np.array(models[i]['flux'])
        fluxg = f_g.get_flux(wavelength, flux, axis=-1)
        fluxi = f_i.get_flux(wavelength, flux, axis=-1)
        fluxr = f_r.get_flux(wavelength, flux, axis=-1)
        fluxu = f_u.get_flux(wavelength, flux, axis=-1)
        fluxz = f_z.get_flux(wavelength, flux, axis=-1)
        fluxY = f_Y.get_flux(wavelength, flux, axis=-1)
        filtersflux = [fluxu,fluxg,fluxr,fluxi,fluxz,fluxY]
        if IR:
            fluxJ = f_J.get_flux(wavelength, flux, axis=-1)
            filtersflux.append(fluxJ)
        filtersfluxnorm = np.array(filtersflux)/max(filtersflux)
        models[i]['filtersflux'] = filtersflux
        models[i]['filtersfluxnorm'] = list(filtersfluxnorm)
        remlist=['wavelength','flux','filtersflux','mbol']
        for key in remlist:
            models[i].pop(key,None)
    return models


# def findbestfits(fwave,fnames,ferrornames,modelslist,modelsname,df):
#     nu2lambda = 2.99792458*(10**18)/(np.array(fwave)**2)
#     bestfits=[]
#     chi2s = []
#     for jj in range(len(df)):
#         flux0 = magAB2flux(np.array([df[fn][jj] for fn in fnames]),units='cgs')*nu2lambda
#         fluxerr0 = magAB2flux(np.array([df[fn][jj]-df[fen][jj] for fn,fen in zip(fnames,ferrornames)]),units='cgs')*nu2lambda-flux0
#         flux0norm = np.array(flux0)/max(flux0)
#         fluxerr0norm = np.array(fluxerr0)/max(flux0)
#         bestfit = chi2bestfit(modelslist,flux0norm,fluxerr0norm)
#         bestfits.append(bestfit)
#         chi2s.append(bestfit['chi2'])
#     # if modelsname+'_chi2' not in df.columns:
#     df[modelsname+'_bestfits']=bestfits
#     df[modelsname+'_chi2']=chi2s
#     # else:
#     #     for jj in range(len(df)):
#     #         df[modelsname+'_chi2'][jj] = min([chi2s[jj],df[modelsname+'_chi2'][jj]])
#     #         bf = [bestfits[jj], df[modelsname+'_bestfits'][jj]]
#     #         df[modelsname+'_bestfits'][jj] = bf[np.argmin([chi2s[jj],df[modelsname+'_chi2'][jj]])]
#     return df

# def findbestfits(fwave,fnames,ferrornames,modelslist,modelsname,df):
#     nu2lambda = 2.99792458*(10**18)/(np.array(fwave)**2)

#     df[modelsname+'_bestfits']=[{} for i in range(len(df))]

#     chi2s=[]
#     for jj in range(len(df)):
#         flux0 = magAB2flux(np.array([df[fn][jj] for fn in fnames]),units='cgs')*nu2lambda
#         fluxerr0 = magAB2flux(np.array([df[fn][jj]-df[fen][jj] for fn,fen in zip(fnames,ferrornames)]),units='cgs')*nu2lambda-flux0
#         flux0norm = np.array(flux0)/max(flux0)
#         fluxerr0norm = np.array(fluxerr0)/max(flux0)
#         bestfit = chi2bestfit(modelslist,flux0norm,fluxerr0norm)
#         df[modelsname+'_bestfits'].iloc[jj]=bestfit
#         if jj==30:
#             print(bestfit)
#             print(df[modelsname+'_bestfits'][jj])
#             print(df[modelsname+'_bestfits'][18])
#         chi2s.append(bestfit['chi2'])
#     df[modelsname+'_chi2']=chi2s
#     return df

def findbestfits(fwave,fnames,ferrornames,modelslist,modelsname,df):
    nu2lambda = 2.99792458*(10**18)/(np.array(fwave)**2)
    chi2s=[]
    empty_df=pd.DataFrame(columns=df.columns)
    with open('data/useless.pickle', 'wb') as f:
            pickle.dump(empty_df,f)
    for jj in range(len(df)):
        flux0 = magAB2flux(np.array([df[fn][jj] for fn in fnames]),units='cgs')*nu2lambda
        fluxerr0 = magAB2flux(np.array([df[fn][jj]-df[fen][jj] for fn,fen in zip(fnames,ferrornames)]),units='cgs')*nu2lambda-flux0
        flux0norm = np.array(flux0)/max(flux0)
        fluxerr0norm = np.array(fluxerr0)/max(flux0)
        bestfit = chi2bestfit(modelslist,flux0norm,fluxerr0norm)
        jj_row_df = df.iloc[jj:(jj+1)]
        jj_row_df[modelsname+'_bestfits']=[bestfit]
        with open('data/useless.pickle', 'rb') as f:
            previous_df = pickle.load(f)
        new_df = pd.concat([previous_df, jj_row_df], ignore_index=True)
        with open('data/useless.pickle', 'wb') as f:
            pickle.dump(new_df,f)
        chi2s.append(bestfit['chi2'])
    with open('data/useless.pickle', 'rb') as f:
        new_df = pickle.load(f)
    new_df[modelsname+'_chi2']=chi2s
    return new_df


def getmin(i, lis):
    sortedlist = lis.copy()
    sortedlist.sort()
    log = []
    for op in range(i):
        log.append(sortedlist[op])
    return log

def chi2bestfit_list(listofstars,starflux,starfluxerror,N=10):
    chi_squares=[]
    for i in range(len(listofstars)):     
        chi2 = chi_square(starflux,listofstars[i]['filtersfluxnorm'],starfluxerror)
        chi_squares.append(chi2)
    positions=[i for i, v in enumerate(list(chi_squares)) if v in getmin(N,list(chi_squares))]
    bestfit = np.array(listofstars)[positions]
    for element in bestfit:
        element['chi2'] = chi_square(starflux,element['filtersfluxnorm'],starfluxerror)
    return bestfit

# def findbestfits_list(fwave,fnames,ferrornames,modelslist,modelsname,df,N=10):
#     nu2lambda = 2.99792458*(10**18)/(np.array(fwave)**2)
#     bestfits=[]
#     chi2s = []
#     for jj in range(len(df)):
#         flux0 = magAB2flux(np.array([df[fn][jj] for fn in fnames]),units='cgs')*nu2lambda
#         fluxerr0 = magAB2flux(np.array([df[fn][jj]-df[fen][jj] for fn,fen in zip(fnames,ferrornames)]),units='cgs')*nu2lambda-flux0
#         flux0norm = np.array(flux0)/max(flux0)
#         fluxerr0norm = np.array(fluxerr0)/max(flux0)
#         bestfit = chi2bestfit_list(modelslist,flux0norm,fluxerr0norm,N=N)
#         bestfits.append(bestfit)
#     df[modelsname+'_'+str(N)+'bestfits']=bestfits
#     return df

def findbestfits_list(fwave,fnames,ferrornames,modelslist,modelsname,df,N=10):
    nu2lambda = 2.99792458*(10**18)/(np.array(fwave)**2)
    empty_df=pd.DataFrame(columns=df.columns)
    with open('data/useless.pickle', 'wb') as f:
            pickle.dump(empty_df,f)
    for jj in range(len(df)):
        flux0 = magAB2flux(np.array([df[fn][jj] for fn in fnames]),units='cgs')*nu2lambda
        fluxerr0 = magAB2flux(np.array([df[fn][jj]-df[fen][jj] for fn,fen in zip(fnames,ferrornames)]),units='cgs')*nu2lambda-flux0
        flux0norm = np.array(flux0)/max(flux0)
        fluxerr0norm = np.array(fluxerr0)/max(flux0)
        bestfit = chi2bestfit_list(modelslist,flux0norm,fluxerr0norm,N=N)
        jj_row_df = df.iloc[jj:(jj+1)]
        jj_row_df[modelsname+'_'+str(N)+'bestfits']=[bestfit]
        with open('data/useless.pickle', 'rb') as f:
            previous_df = pickle.load(f)
        new_df = pd.concat([previous_df, jj_row_df], ignore_index=True)
        with open('data/useless.pickle', 'wb') as f:
            pickle.dump(new_df,f)
    return new_df





def plotbestfit(fwave,fnames,models,df,jj=0):
    nu2lambda = 2.99792458*(10**18)/(np.array(fwave)**2)
    flux0 = magAB2flux(np.array([df[fn][jj] for fn in fnames]),units='cgs')*nu2lambda
    flux0norm = np.array(flux0)/max(flux0)
    bestfit = df[models+'_bestfits'][jj]
    string ='best fit btdusty:\nChi2={chi:.3f}'.format(chi=bestfit['chi2'])
    plt.scatter(fwave,bestfit['filtersfluxnorm'],label=string,alpha=0.9)
    plt.scatter(fwave,flux0norm,c='orange',label='my star',zorder=3,alpha=0.7)
    plt.show()






# def smooth_wave(wave, spec, outwave, sigma, nsigma=10, inres=0, in_vel=False,
#                 **extras):
#     """Smooth a spectrum in wavelength space.  This is insanely slow, but
#     general and correct (except for the treatment of the input resolution if it
#     is velocity)

#     Parameters
#     ----------
#     wave : ndarray of shape ``(N_pix,)``
#         Wavelength vector of the input spectrum.
#     spec : ndarray of shape ``(N_pix,)``
#         Flux vector of the input spectrum.
#     outwave : ndarray of shape ``(N_pix_out,)``
#         Desired output wavelength vector.
#     sigma : float or ndarray of shape ``(N_pix,)``
#         Desired resolution (dispersion *not* FWHM) in wavelength units.  This
#         can be a vector of same length as ``wave``, in which case a wavelength
#         dependent broadening is calculated

#     nsigma : float (optional, default=10)
#         Number of sigma away from the output wavelength to consider in the
#         integral.  If less than zero, all wavelengths are used.  Setting this to
#         some positive number decreses the scaling constant in the O(N_out *
#         N_in) algorithm used here.
#     inres : float (optional)
#         Resolution of the input, in either wavelength units or
#         :math:`\lambda/d\lambda \cdot (c/v)`. The spectrum will be smoothed by
#         the quadrature difference between the desired resolution and ``inres``
#     in_vel : bool (optional, default: False)
#         If True, the input spectrum has been smoothed in velocity
#         space, and ``inres`` is assumed to be in lambda/dlambda.

#     Returns
#     -------
#     smoothed_spec : ndarray of shape ``(N_pix_out,)``
#         The smoothed spectrum.
#     """
#     # sigma_eff is in angstroms
#     if inres <= 0:
#         sigma_eff_sq = sigma**2
#     elif in_vel:
#         # Make an approximate correction for the intrinsic wavelength
#         # dependent dispersion.  This sort of maybe works.
#         sigma_eff_sq = sigma**2 - (wave / inres)**2
#     else:
#         sigma_eff_sq = sigma**2 - inres**2
#     if np.any(sigma_eff_sq < 0):
#         raise ValueError("Desired wavelength sigma is lower than the value "
#                          "possible for this input spectrum.")

#     sigma_eff = np.sqrt(sigma_eff_sq)
#     flux = np.zeros(len(outwave))
#     for i, w in enumerate(outwave):
#         x = (wave - w) / sigma_eff
#         if nsigma > 0:
#             good = np.abs(x) < nsigma
#             x = x[good]
#             _spec = spec[good]
#         else:
#             _spec = spec
#         f = np.exp(-0.5 * x**2)
#         flux[i] = np.trapz(f * _spec, x) / np.trapz(f, x)
#     return flux