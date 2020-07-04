#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 08:35:28 2020

@author: yuhanyao
"""
import os
from copy import deepcopy
import numpy as np
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.io.ascii as asci
from astropy.time import Time
from astropy.coordinates import get_sun
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import matplotlib
import matplotlib.pyplot as plt
fs= 10
matplotlib.rcParams['font.size']=fs
ms = 5
matplotlib.rcParams['lines.markersize']=ms


def SRG_pointing(datestr = "20200704"):
    """
    Sun's movement during the year:
        http://community.dur.ac.uk/john.lucey/users/solar_year.html#:~:text=At%20the%20equinoxes%20at%20every,is%20on%20the%20celestial%20equator.&text=(September)%20equinox.-,At%20this%20time%20the%20Sun%20is%20at%20RA,h%2C%20Dec%20%3D%200.0%C2%B0.
    
    SRG real-time tracking:
        http://plan.srg.cosmos.ru/monthplan/tracking
        
    SRG obs strategy:
        1 sq FoV, 1 sq per day drift, one full circle every four hours
    """
    # Palomar midnight = UTC 7am
    tnow = Time(datestr[:4]+"-"+datestr[4:6]+'-'+datestr[6:]+'T07:00:00.0', format='isot')
    sunpos = get_sun(tnow)
    sun_ra_rad = sunpos.ra.rad
    sub_dec_rad = sunpos.dec.rad
    
    sun_theta = np.pi/2 - sub_dec_rad
    sun_phi = sun_ra_rad
    a = np.sin(sun_theta)*np.cos(sun_phi)
    b = np.sin(sun_theta)*np.sin(sun_phi)
    c = np.cos(sun_theta)
    # the plane that is perpendicular to the Sun is 
    # ax + by + cz = 0
    phis = np.linspace(0, 2*np.pi, 360)
    thetas = np.zeros(len(phis))
    for i in range(len(phis)):
        phi = phis[i]
        a_prime = a * np.cos(phi)
        b_prime = b * np.sin(phi)
        theta_ = np.arctan(-1 * c / (a_prime + b_prime))
        if theta_ < 0:
            theta = theta_+np.pi
        else:
            theta = theta_
        thetas[i] = theta
    alphas = phis
    decs = np.pi/2 - thetas
    return alphas/np.pi*180, decs/np.pi*180


def get_projection_rad(ra, dec):
    """
    input in degree
    output in rad (for aitoff projection)
    """
    c = SkyCoord(ra=ra* u.degree, dec=dec* u.degree, frame='icrs')
    ra_f = c.ra.wrap_at(180 * u.deg).radian
    dec_f = c.dec.radian
    return ra_f, dec_f


def _if_in_field(rag, decg, rap, decp):
    """
    rag, decg: grid boundary
    rap, decp: point source
    """
    point = Point(rap, decp)
    bounds = []
    for j in range(len(rag)):
        bounds.append((rag[j], decg[j]))
    polygon = Polygon(bounds)
    return polygon.contains(point)


def if_in_field(rag, decg, rap, decp):
    """
    rag, decg: grid boundary
    rap, decp: point source
    """
    if (max(rag) - min(rag))>100:
        ix = rag<180
        rag1 = deepcopy(rag)
        rag2 = deepcopy(rag)
        decg1 = deepcopy(decg)
        decg2 = deepcopy(decg)
        rag1[ix] += 360
        rag2[~ix] -= 360
        flag1 = _if_in_field(rag1, decg1, rap, decp)
        flag2 = _if_in_field(rag2, decg2, rap, decp)
        flag = flag1 | flag2
    else:
        flag = _if_in_field(rag, decg, rap, decp)
    return flag


def get_fieldid_srg(fields, alphas, decs):
    flags = np.zeros(len(fields))
    for i in range(len(fields)):
        rag, decg = np.loadtxt(fields["filename"][i]).T
        for j in range(len(alphas)):
            rap = alphas[j]
            decp = decs[j]
            ff = if_in_field(rag, decg, rap, decp)
            if ff==True:
                flags[i] += 1
    return flags


def field_grids(datestr):
    dirstr = "bydate/"
    outdir = os.path.join(dirstr, datestr)
    if os.path.exists(outdir) == False:
        os.mkdir(outdir)

    fields = asci.read("files/ZTF_fields.txt",
                       names = ["ID","RA","Dec","Ebv","Gal Long","Gal Lat",
                                "Ecl Long","Ecl Lat","Entry"])
    fields = fields[fields["ID"]>199]
    fields = fields[fields["ID"]<880]
    #fieldid > 199, fieldid < 880
    myfiles = ["files/field_edges/"+str(i)+".dat" for i in fields["ID"].data]
    myfiles = np.array(myfiles)
    fields["filename"] = myfiles
    
    alphas, decs = SRG_pointing(datestr = datestr)
    """
    plt.plot(alphas, decs)
    plt.plot(23.95, 2.33, 'ro')
    plt.xlim(0, 50)
    plt.ylim(0, 5)
    """
    
    flags = get_fieldid_srg(fields, alphas, decs)
    
    ra_srg, dec_srg = get_projection_rad(alphas, decs)
    ra_f, dec_f = get_projection_rad(fields["RA"], fields["Dec"])
    
    
    plt.figure(figsize=(18,10))
    plt.subplot(111, projection="aitoff")
    plt.plot(ra_srg, dec_srg, 'o', color = "gold", markersize = 6, alpha=1)
    
    for i in range(len(fields)):
        rag, decg = np.loadtxt(fields["filename"][i]).T
        ra_g, dec_g = get_projection_rad(rag, decg)
        if flags[i]>0:
            color = "r"
            zorder = 2
            fsnow = 10
            xoff = -2/180*np.pi
            yoff = -1/180*np.pi
        else:
            color = "k"
            fsnow = 8
            zorder = 1
            xoff = -2/180*np.pi
            yoff = -1/180*np.pi
        plt.text(ra_f[i]+xoff, dec_f[i]+yoff, str(fields["ID"].data[i]), color = color, fontsize = fsnow)
        
        ix1 = (rag>90)&(rag<180)
        ix2 = (rag>180)&(rag<270)
        if np.sum(ix1)>=1 and np.sum(ix2)>=1:
            plt.plot(ra_g[ix1], dec_g[ix1], '-', alpha=1, linewidth = 0.5, color = color,
                     zorder = zorder)
            plt.plot(ra_g[ix2], dec_g[ix2], '-', alpha=1, linewidth = 0.5, color = color,
                     zorder = zorder)
        else:
            plt.plot(ra_g, dec_g, '-', alpha=1, linewidth = 0.5, color = color,
                     zorder = zorder)
        
    plt.title(datestr+": %d fields"%(np.sum(flags>0)))
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
    datestr = "20200705"
    field_grids(datestr)
    