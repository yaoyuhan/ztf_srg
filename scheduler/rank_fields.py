#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:34:55 2020

@author: yuhanyao
"""
import astropy.constants as const
from copy import deepcopy
import datetime
import os
import pandas as pd
import glob
import numpy as np
from astropy.table import Table
from astropy import units as u
import astropy.io.ascii as asci
from astropy.time import Time
from astral import Astral
from astropy.coordinates import get_sun, Galactic
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from plan import raDec2xy, get_fieldid_srg

import matplotlib
import matplotlib.pyplot as plt
fs= 10
matplotlib.rcParams['font.size']=fs
ms = 5
matplotlib.rcParams['lines.markersize']=ms

    

def load_fields(datestr = "20201209"):
    """
    For a given date, for a given field, calculate rise time, set time, 
    duration above airmass 2.
    
    UTC - 7 = PDT (mid-March to early November)
    UTC - 8 = PST (early November to mid-March)
    
    Use PST in the following computation
    
    Remove fields with Dec > 80
    """
    fields = asci.read("files/ZTF_fields.txt",
                       names = ["ID","RA","Dec","Ebv","Gal Long","Gal Lat",
                                "Ecl Long","Ecl Lat","Entry"])
    fields = fields[fields["ID"]>199]
    fields = fields[fields["ID"]<880]
    #fieldid > 199, fieldid < 880
    myfiles = ["files/field_edges/"+str(i)+".dat" for i in fields["ID"].data]
    myfiles = np.array(myfiles)
    fields["filename"] = myfiles
    fields = fields[fields["Dec"]<80]
    
    palomar = EarthLocation.of_site("Palomar")
    
    ### get UTC sunrise and sunset time
    city_name = 'san diego'
    a = Astral()
    a.solar_depression = 'nautical'  # 12 degree twilight
    #String        Degrees
    #civil            6.0
    #nautical        12.0
    #astronomical    18.0
    city = a[city_name]
    sun = city.sun(date=datetime.date(int(datestr[:4]), int(datestr[4:6]), int(datestr[6:])), 
                   local=False)
    sunrise = sun["sunrise"]  # UTC sunrise
    sunset = sun["sunset"]
    sunrise_h_utc = sunrise.hour + sunrise.minute/60 + sunrise.second/3600 
    sunset_h_utc = sunset.hour + sunset.minute/60 + sunset.second/3600
    midnight_h_utc = (sunrise_h_utc + sunset_h_utc)/2.
    halfnight_h = (sunrise_h_utc - sunset_h_utc)/2.
    
    result = {}
    midnight_utc = Time(datestr[:4]+"-"+datestr[4:6]+'-'+datestr[6:]+'T00:00:00') + midnight_h_utc*u.hour
    result["midnight_utc"] = midnight_utc
    #startnight_utc = midnight_utc - halfnight_h*u.hour
    #endnight_utc = midnight_utc + halfnight_h*u.hour
    #print (datestr, startnight_utc, endnight_utc)
    
    delta_midnight = np.linspace(-halfnight_h, halfnight_h, 200)*u.hour # 4~5 min space
    result["start_h"] = delta_midnight[0].value
    result["end_h"] = delta_midnight[-1].value
    
    frame_tonight = AltAz(obstime=midnight_utc+delta_midnight, 
                          location=palomar)
    
    up_fs = np.ones(len(fields))
    duration_hs = np.zeros(len(fields))
    for i in range(len(fields)):
        ra_f = fields["RA"].data[i]
        dec_f = fields["Dec"].data[i]
        ztf_f = SkyCoord(ra = ra_f, dec = dec_f, unit = 'degree')
    
        ztf_f_altazs_tonight = ztf_f.transform_to(frame_tonight)
        ztf_f_airmasss_tonight = ztf_f_altazs_tonight.secz.value
        ztf_f_alt_tonight = ztf_f_altazs_tonight.alt.deg # in degree
        ix = ztf_f_alt_tonight>0
        if np.sum(ix)==0:
            up_fs[i] = 0
        else:
            delta_midnight_h = delta_midnight[ix].value
            airmass_tonight = ztf_f_airmasss_tonight[ix]
            ind_ = np.where(airmass_tonight<2)
            ind = ind_[0]
            if np.sum(ind)<=1:
                up_fs[i] = 0
            else:
                if np.sum(np.diff(ind)!=1)!=0:
                    temp = np.diff(ind)
                    arg = np.where(temp!=1)[0][0]
                    ind1 = ind[:arg+1]
                    ind2 = ind[arg+1:]
                    rise_time1 = delta_midnight_h[ind1[0]]
                    set_time1 = delta_midnight_h[ind1[-1]]
                    rise_time2 = delta_midnight_h[ind2[0]]
                    set_time2 = delta_midnight_h[ind2[-1]]
                    duration_hs[i] = (set_time1 - rise_time1) + (set_time2 - rise_time2)
                else:
                    rise_time = delta_midnight_h[ind[0]]
                    set_time = delta_midnight_h[ind[-1]]
                    duration_hs[i] = set_time - rise_time
                    #print (rise_time, set_time)
        """
        plt.plot(delta_midnight_h, airmass_tonight)
        plt.ylim(4, 1)
        plt.xlabel('Hours from Midnight')
        plt.ylabel('Airmass [Sec(z)]')
        """
        
    fields["up?"] = up_fs
    fields["duration_h"] = duration_hs
    result["fields"] = fields
    return result


def load_SRG_pointing(datestr):
    csvfiles = glob.glob("./files/srg_plan/*.csv")
    csvfiles = np.array(csvfiles)
    csvfiles = csvfiles[np.argsort(csvfiles)]
    nfiles = len(csvfiles)
    for i in range(nfiles):
        if i==0:
            tb = pd.read_csv(csvfiles[i])
        else:
            tb = pd.concat([tb, pd.read_csv(csvfiles[i])])
    
    tstart = Time(datestr[:4]+"-"+datestr[4:6]+'-'+datestr[6:]+'T00:00:00.0', format='isot')
    #tend = Time(datestr[:4]+"-"+datestr[4:6]+'-'+datestr[6:]+'T23:59:59.9', format='isot')
    ix = (tb["MJD"]>=tstart.mjd)&(tb["MJD"]<(tstart.mjd+1))
    tb = tb[ix]
    alphas = tb["RA"].values
    deltas = tb["Dec"].values
    return alphas, deltas


def add_galactic_belt(ax):
    coo_GP_p30 = SkyCoord(l = np.linspace(0, 360, 500)* u.deg, 
                          b = np.ones(500)*15* u.deg, frame="galactic")
    ra_GP_p30 = coo_GP_p30.icrs.ra.degree
    dec_GP_p30 = coo_GP_p30.icrs.dec.degree
    x_GP_p30, y_GP_p30 = raDec2xy(ra_GP_p30, dec_GP_p30)
    
    coo_GP_n30 = SkyCoord(l = np.linspace(0, 360, 500)* u.deg, 
                          b = np.ones(500)*-15* u.deg, frame="galactic")
    ra_GP_n30 = coo_GP_n30.icrs.ra.degree
    dec_GP_n30 = coo_GP_n30.icrs.dec.degree
    x_GP_n30, y_GP_n30 = raDec2xy(ra_GP_n30, dec_GP_n30)
    ax.plot(x_GP_p30, y_GP_p30, ".", color="deepskyblue", markersize = 3)
    ax.plot(x_GP_n30, y_GP_n30, ".", color="deepskyblue", markersize = 3)
    
    
def add_ecliptic_belt(ax):
    coo_GP_p30 = SkyCoord(lon = np.linspace(0, 360, 500)* u.deg, 
                          lat = np.ones(500)*15* u.deg, frame="barycentricmeanecliptic")
    ra_GP_p30 = coo_GP_p30.icrs.ra.degree
    dec_GP_p30 = coo_GP_p30.icrs.dec.degree
    x_GP_p30, y_GP_p30 = raDec2xy(ra_GP_p30, dec_GP_p30)
    
    coo_GP_n30 = SkyCoord(lon = np.linspace(0, 360, 500)* u.deg, 
                          lat = np.ones(500)*-15* u.deg, frame="barycentricmeanecliptic")
    ra_GP_n30 = coo_GP_n30.icrs.ra.degree
    dec_GP_n30 = coo_GP_n30.icrs.dec.degree
    x_GP_n30, y_GP_n30 = raDec2xy(ra_GP_n30, dec_GP_n30)
    ax.plot(x_GP_p30, y_GP_p30, ".", color="lime", markersize = 3)
    ax.plot(x_GP_n30, y_GP_n30, ".", color="lime", markersize = 3)
    
    
def rescale_flags(flags):
    newflags = np.zeros(len(flags))
    ix1 = (flags>=1)&(flags<5)
    ix2 = (flags>=5)&(flags<10)
    ix3 = (flags>=10)&(flags<15)
    ix4 = (flags>=15)&(flags<20)
    ix5 = flags>=20
    newflags[ix1] = 1
    newflags[ix2] = 2
    newflags[ix3] = 3
    newflags[ix4] = 4
    newflags[ix5] = 5
    return newflags
    


def field_grids_realistic(datestr):
    dirstr = "bydate/"
    outdir = os.path.join(dirstr, datestr)
    if os.path.exists(outdir) == False:
        os.mkdir(outdir)
        
    result = load_fields(datestr = datestr)
    fields = result["fields"]
    x_f, y_f = raDec2xy(fields["RA"], fields["Dec"])
    
    alphas, decs = load_SRG_pointing(datestr)
    flags = get_fieldid_srg(fields, alphas, decs)
    up_durations = fields["duration_h"].data
    flags[up_durations<1] = 0
    flags = rescale_flags(flags)
    x_srg, y_srg = raDec2xy(alphas, decs)
    
    ### plot Galactic plane
    
    
    plt.figure(figsize=(15, 8))
    ax = plt.subplot(111)
    cm = matplotlib.cm.get_cmap('binary')
    sc = ax.scatter(x_f, y_f, c=fields["duration_h"], cmap = cm)
    plt.colorbar(sc, label = "Hour above Airmass = 2")
    plt.plot(x_srg, y_srg, 'o', color = "gold", markersize = 4, alpha=1)
    add_galactic_belt(ax)
    add_ecliptic_belt(ax)

    cmap='Reds'
    cNorm  = matplotlib.colors.Normalize(vmin=0, 
                                         vmax=5)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    sm = scalarMap

    
    x_f, y_f = raDec2xy(fields["RA"], fields["Dec"])
    for i in range(len(fields)):
        rag, decg = np.loadtxt(fields["filename"][i]).T
        x_g, y_g = raDec2xy(rag, decg)
        
        if flags[i]>0:
            color = sm.to_rgba(flags[i])
            zorder = 2
            fsnow = 10
            xoff = -2/180*np.pi
            yoff = -1/180*np.pi
            lw = 1
        else:
            color = "k"
            fsnow = 8
            zorder = 1
            xoff = -2/180*np.pi
            yoff = -1/180*np.pi
            lw = 0.3
        #plt.text(x_f[i]+xoff, y_f[i]+yoff, str(fields["ID"].data[i]), color = color, fontsize = fsnow)
        
        if color!="k":
            ix1 = (rag>0)&(rag<90)
            ix2 = (rag>270)&(rag<360)
            if np.sum(ix1)>=1 and np.sum(ix2)>=1:
                plt.plot(x_g[ix1], y_g[ix1], '-', alpha=1, linewidth = lw, color = color,
                         zorder = zorder)
                plt.plot(x_g[ix2], y_g[ix2], '-', alpha=1, linewidth = lw, color = color,
                         zorder = zorder)
            else:
                plt.plot(x_g, y_g, '-', alpha=1, linewidth = lw, color = color,
                         zorder = zorder)
        
    titlename = datestr+": %d fields"%(np.sum(flags>0))
    print (titlename)
    plt.title(titlename)
    plt.axis('off')
    plt.tight_layout()
    
    figfile = os.path.join(outdir, "fields_%s.png"%datestr)
    plt.savefig(figfile)
    plt.close()
    
    fields_obs = fields["ID"].data[flags>0] 
    flags_obs = flags[flags>0] 
    arg = np.argsort(flags_obs)[::-1]
    fields_obs = fields_obs[arg]
    flags_obs = flags_obs[arg]
    tb = Table([fields_obs, flags_obs],
               names = ["fieldid", "priority"])
    fieldfile = os.path.join(outdir, "fields_srg_%s.txt"%datestr)
    tb.write(fieldfile, overwrite = True, format = "ascii")


if __name__=="__main__":
   
    for datestr in [#"20201209",
                    "20201210", "20201211", "20201212",
                    "20201213", "20201214", "20201215", "20201216",
                    "20201217", "20201218", "20201219", "20201220",
                    "20201221", "20201222", "20201223", "20201224",
                    "20201225", "20201226", "20201227", "20201228"]:
        field_grids_realistic(datestr)
    
    
    
    
    #field_grids_realistic("20201209")
    
    