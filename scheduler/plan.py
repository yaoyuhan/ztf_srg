#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 08:35:28 2020

@author: yuhanyao

Eric's email: programpi = 'SRG'
"""
import os
import astropy.constants as const
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


def get_theta_given_phi(a, b, c, phi):
    """
    ax + by + cz defines a plane
    point coordinate (theta, phi)
    """
    a_prime = a * np.cos(phi)
    b_prime = b * np.sin(phi)
    theta_ = np.arctan(-1 * c / (a_prime + b_prime))
    if theta_ < 0:
        theta = theta_+np.pi
    else:
        theta = theta_
    return theta


def cos_in_sphere(theta1, phi1, theta2, phi2):
    """
    Here we use te cosine rule in Spherical trigonometry
    https://en.wikipedia.org/wiki/Spherical_trigonometry
    """
    term1 = np.cos(theta2) * np.cos(theta1)
    term2 = np.sin(theta2) * np.sin(theta1) * np.cos(phi1-phi2)
    cos_rad = term1 + term2
    return cos_rad


def SRG_pointing(datestr = "20200709"):
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
    phis = [0]
    thetas = [get_theta_given_phi(a, b, c, phis[0])]
    sep_degs = []
    for i in range(360):
        phi_now = phis[-1]
        theta_now = thetas[-1]
        
        # first of all, we assume that phi is uniformly scanned, but this is not the case
        # so we use this as an initial guess, and adjust thhe step, such that
        # the angle between consecutive points are the same (~1 degree, 360 points)
        
        step_deg = 1
        step_rad = step_deg/180*np.pi
        phi_trial = phi_now + step_rad
        theta_trial = get_theta_given_phi(a, b, c, phi_trial)
        cossep_trial = cos_in_sphere(theta_now, phi_now, theta_trial, phi_trial)
        sep_deg_trial = np.arccos(cossep_trial)/np.pi*180
        while abs(sep_deg_trial-1)>0.001:
            step_deg /= sep_deg_trial
            step_rad = step_deg/180*np.pi
            phi_trial = phi_now + step_rad
            theta_trial = get_theta_given_phi(a, b, c, phi_trial)
            cossep_trial = cos_in_sphere(theta_now, phi_now, theta_trial, phi_trial)
            sep_deg_trial = np.arccos(cossep_trial)/np.pi*180
        phis.append(phi_trial)
        thetas.append(theta_trial)
        sep_degs.append(sep_deg_trial)
    phis = np.array(phis)
    thetas = np.array(thetas)
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


def raDec2xy(ra,dec):
    """
    borrowed from Eric's ztf_sim notebook
    """
    # Using Aitoff projections (from Wiki) returns x-y coordinates on a plane of RA and Dec
    theta = np.deg2rad(dec)
    phi = np.deg2rad(ra)-np.pi #the range is [-pi,pi]
    alpha = np.arccos(np.cos(theta)*np.cos(phi/2))
    x = 2*np.cos(theta)*np.sin(phi/2)/np.sinc(alpha/np.pi) # The python's sinc is normalazid, hence the /pi
    y = np.sin(theta)/np.sinc(alpha/np.pi)
    return x, y


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
    flags = get_fieldid_srg(fields, alphas, decs)
    
    x_srg, y_srg = raDec2xy(alphas, decs)
    
    plt.figure(figsize=(18,10))
    plt.subplot(111)
    plt.plot(x_srg, y_srg, 'o', color = "gold", markersize = 6, alpha=1)
    
    x_f, y_f = raDec2xy(fields["RA"], fields["Dec"])
    for i in range(len(fields)):
        rag, decg = np.loadtxt(fields["filename"][i]).T
        x_g, y_g = raDec2xy(rag, decg)
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
        plt.text(x_f[i]+xoff, y_f[i]+yoff, str(fields["ID"].data[i]), color = color, fontsize = fsnow)
        
        ix1 = (rag>0)&(rag<90)
        ix2 = (rag>270)&(rag<360)
        if np.sum(ix1)>=1 and np.sum(ix2)>=1:
            plt.plot(x_g[ix1], y_g[ix1], '-', alpha=1, linewidth = 0.5, color = color,
                     zorder = zorder)
            plt.plot(x_g[ix2], y_g[ix2], '-', alpha=1, linewidth = 0.5, color = color,
                     zorder = zorder)
        else:
            plt.plot(x_g, y_g, '-', alpha=1, linewidth = 0.5, color = color,
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
    for datestr in ["20200711", "20200712", "20200713", "20200714",
                    "20200715", "20200716", "20200717", "20200718",
                    "20200719", "20200720", "20200721", "20200722",
                    "20200723", "20200724", "20200725", "20200726",
                    "20200727", "20200728", "20200729", "20200730",
                    "20200731", "20200801"]:
        field_grids(datestr)
    