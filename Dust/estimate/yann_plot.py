# %%
# import library
import numpy as np
import matplotlib.pyplot as plt


def show_omega_v3_points(omega_mat, omega_mat_longrid, omega_mat_latgrid, mask_omega_mat=None, 
    omega_name=None, cb_title='IBD', omega_contour=None, omega_counter_levels=None, 
    mask_mat_points=None, omega_mat_lon=None, omega_mat_lat=None, legend_points=None, 
    color_plot='rainbow', marker='s', markersize=3, cmap='Greys_r', vmin=None, vmax=None, 
    alpha=None, title='auto', lonlim=(None, None), latlim=(None, None), Nfig=None, 
    figsize=(8.4,6.8), polar=False, cbar="vertical", grid=True, axis='equal', ajustable=False, 
    xticks_loc=None, xticks_labels=None, yticks_loc=None, yticks_labels=None, savename=None, 
    bg_color=None, dpi=None, fontsize_cb=None, extendrect=True, invert_xaxis=False, 
    invert_yaxis=False):

    """Display an OMEGA/MEx observation with respect of the lat/lon coordinates of the pixels,
    and allows to use a polar projection if desired.

    Parameters
    ==========
    omega_mat: 2D-array (float)
        The OMEGA/MEx observation 2D-array we want to plot.
    omega_mat_longrid: 2D-array (float)
        The OMEGA/MEx observation longitude grid.
    omega_mat_latgrid: 2D-array (float)
        The OMEGA/MEx observation latitude grid.
    mask_omega_mat: 2D-array (bool), optional (default None)
        The OMEGA/MEx observation mask.
    omega_name: str, optional (default None)
        The OMEGA/MEx observation name.
    mask_mat_points: list of 2D-array (bool), optional (default None)
        The masks on the OMEGA/MEx observation to plot the points of interest.
    omega_mat_lon: 2D-array (float)
        The OMEGA/MEx observation longitude (center of the pixels) used to plot points over the map.
    omega_mat_lat: 2D-array (float)
        The OMEGA/MEx observation latitude (center of the pixels) used to plot points over the map.
    legend_points: list, optional (default None)
        The legend(s) of the points plotted.
    marker: str, optional (default 's')
        The matplotlib marker style.
    markersize: str, optional (default 4)
        The matplotlib marker style.
    cmap: str, optional (default 'Greys_r')
        The matplotlib colormap.
    vmin: float or None, optional (default None)
        The lower bound of the coloscale.
    vmax: float or None, optional (default None)
        The upper bound of the colorscale.
    alpha: float or None, optional (default None)
        Opacity of the plot.
    title: str, optional (default 'auto')
        The title of the figure.
    lonlim: tuple of int or None, optional (default (None, None))
        The longitude bounds of the figure.
    latlim: tuple of int or None, optional (default (None, None))
        The latitude bounds of the y-axis of the figure.
    Nfig: int or str or None, optional, default None)
        The target figure ID.
   figsize: tuple, float (default : (8.4,6.8))
        The size figure in inches of the plot.
    polar: bool, optional (default False)
        If True -> Use a polar projection for the plot.
    cbar: str, optional (default "vertical")
        If "vertical" -> diplay the colorbar vertically on the righthand of the figure.
        If "horizontal" -> diplay the colorbar horizontally on the bottom of the figure.
    grid: bool, optional (default True)
        Enable the display of the lat/lon grid.
    axis: str, optional (default 'equal')
        The set scaling of the plot.
    ajustable: bool, optional (default False)
        If True -> Adjuste figure on the cube.
    global_map: bool, optional (default False)
        If True -> Adjuste plot features for a global map.
    savename: str or None, optional (default None)
        The filename to save the figure.
    bg_color: str, optional (default None)
        The name of the background color of the figure.
    fontsize_cb: int, optional (default None)
        The size of the colorbar label font in points.
    extendrect: bool, optional (default True)
        If False the minimum and maximum colorbar extensions will be triangular (the default).
    invert_xaxis: bool, optional (default False)
        If True the x axis direction is inverted.
    invert_yaxis: bool, optional (default False)
        If True the y axis direction is inverted.
    """
    if  omega_name is not None:
        title_auto = ('OMEGA/MEx observation {0}'.format(omega_name))
    else:
        title_auto = None
    fig = plt.figure(Nfig, figsize=figsize)
    Nfig = fig.number   # get the actual figure number if Nfig=None
    
    omega_mat_copy = copy.deepcopy(omega_mat)
    if mask_omega_mat is not None:
        omega_mat_copy[np.logical_not(mask_omega_mat)] = np.nan
    if polar:
        ax = plt.axes(polar=True)
        plt.pcolormesh(omega_mat_longrid*np.pi/180, omega_mat_latgrid, omega_mat, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        ax.set_yticklabels([])  # remove the latitude values in the plot
        ax.set_theta_offset(-np.pi/2)   # longitude origin at the bottom
    else:
        ######### Modif Yann

        plt.pcolormesh(omega_mat_longrid, omega_mat_latgrid, omega_mat_copy, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        if mask_mat_points is not None:
            plt.pcolormesh(omega_mat_longrid, omega_mat_latgrid, omega_mat_copy, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
            nb_dataset = int(np.shape(mask_mat_points)[0])
            n_color = nb_dataset + 1
            if color_plot == 'rainbow':
                color=iter(cm.gist_rainbow(np.linspace(0,1,n_color)))
            elif color_plot == 'gray':
                color=iter(cm.gray(np.linspace(0,1,n_color)))
            else:
                ValueError("'color_plot' must be : 'rainbow' (default) or 'gray'.")
            for i in range(nb_dataset):
                if legend_points is not None:
                    c=next(color); plt.plot(omega_mat_lon[mask_mat_points[i]], omega_mat_lat[mask_mat_points[i]], c=c, marker=marker, markersize=markersize, fillstyle='none', lw=0, zorder=2, label=legend_points[i])
                else:
                    c=next(color); plt.plot(omega_mat_lon[mask_mat_points[i]], omega_mat_lat[mask_mat_points[i]], c=c, marker=marker, markersize=markersize, fillstyle='none', lw=0, zorder=2)
        #########
        plt.gca().axis(axis)
        plt.xlabel('Longitude [°E]', fontsize=15)
        plt.ylabel('Latitude [°]', fontsize=15)
    
    if cbar == "vertical":
        cb = plt.colorbar()
    elif cbar == "horizontal":
        cb = plt.colorbar(orientation="horizontal", location="bottom") #extendrect=extendrect
    
    if fontsize_cb is None:
        fontsize_cb = 15
    cb.set_label(cb_title, fontsize=fontsize_cb)
    if grid:
        ax = plt.figure(Nfig, figsize=figsize).get_axes()[0] #modif Yann
        if ajustable == True:
            ax.set_adjustable('box')

        plt.grid()
        if polar:
            latlim = ax.get_ylim()
            lat_sgn = np.sign(latlim[1] - latlim[0])
            lat_grid = np.arange(np.round(latlim[0]/5)*5, np.round(latlim[1]/5)*5+lat_sgn, 5 * lat_sgn)   # 5° grid in latitude
            ax.set_rticks(lat_grid)
    if omega_contour is not None:
        plt.contour(omega_contour[0], omega_contour[1], omega_contour[2], omega_counter_levels)

    plt.xticks(xticks_loc, xticks_labels, fontsize=13) #modif Yann
    plt.yticks(yticks_loc, yticks_labels, fontsize=13) #modif Yann
    if title == 'auto':
        title = title_auto
    plt.title(title, fontsize=13)
    if legend_points is not None:
        plt.legend(fontsize=11) #modif Yann
    
    if bg_color is not None:
        facecolorRGBA = mc.to_rgba(bg_color)
        ax.patch.set_facecolor(facecolorRGBA[:-1])
        ax.patch.set_alpha(facecolorRGBA[-1])
    plt.xlim(lonlim)
    plt.ylim(latlim)
    if invert_xaxis is True:
        plt.gca().invert_xaxis()
    if invert_yaxis is True:
        plt.gca().invert_yaxis()
    if savename is None:
        plt.tight_layout()
    else:
        plt.tight_layout()
        plt.savefig(savename, dpi=dpi)