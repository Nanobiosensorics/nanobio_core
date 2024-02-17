import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

def animate_well(data, filename, nframes, vmin, vmax, text):
    '''
        Animate a single well image sequence.
        
        Parameters
        ----------
        data        -    (t, 80, 80) matrix containing the well data
        filename    -    name of the output file
        nframes     -    the length of the sequence you wish to animate
        vmin        -    the lower boundary of the pixel values
        vmax        -    the upper boundary of the pixel values
        text        -    the title of the animated plot
    '''
    fig = plt.figure(figsize=(20, 10), dpi=300)
    im = plt.imshow(data[-1, :, :],interpolation='none', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(text)


    # animation function.  This is called sequentially
    def animate(i):
        print(f'{i}', end='\r')
        im.set_data(data[i, :, :])
        return [im]

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, 
                                   frames=nframes, interval=200, blit=True)
    anim.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])
    
def animate_well_by_ID(wells, text):
    '''
        Select a well based on its ID in the well map and animate it.
        For example: "B2" is the well from the 2nd row and 2nd column.
        
        Parameters
        ----------
        text - well ID
    '''
    row = ord(text[0]) - 65
    col = ord(text[1]) - 49
    well = wells[row, col, :, :, :]
    m = np.median(well[well > 820])
    s = np.std(well[well > 820])
    animate_well(well, f'./{text}.mp4', wells.shape[2], m - s, m + s, f'{text} well')

def animate_well_maxima(data, coords, lines, time, filename, nframes, vmin, vmax, text):
    '''
        Animate a single well image sequence with its selected maximum points.
        
        Parameters
        ----------
        data        -    (t, 80, 80) matrix containing the well data
        coords      -    list that contains the coordinates of the maximum points
        filename    -    name of the output file
        nframes     -    the length of the sequence you wish to animate
        vmin        -    the lower boundary of the pixel values
        vmax        -    the upper boundary of the pixel values
        text        -    the title of the animated plot
    '''
    fig, ax = plt.subplots(1,3, figsize=(15, 5), dpi=200)
    elements = []
    im = ax[0].imshow(data[-1, :, :],interpolation='none', vmin=vmin, vmax=vmax)
    elements.append(im)
    _ = ax[0].scatter(coords[:, 0], coords[:, 1] , s=1, c='r')
    # elements.append(line)
    # plt.colorbar()
    ax[0].set_title(text)
    
    mn = np.mean(lines, axis=0)
    sd = np.std(lines, axis=0)
    
    ax[1].fill_between(time, mn - sd, mn + sd, alpha=.2)
    ax[1].plot(time, mn - sd, c='b')
    ax[1].plot(time, mn + sd, c='b')
    ax[1].plot(time, mn, c='b')
    ax[1].set_ylim(np.min(lines), np.max(lines))
    tempo1 = ax[1].axvline(0, c='r')
    elements.append(tempo1)

    ax[2].plot(time, lines.T)
    ax[2].set_ylim(np.min(lines), np.max(lines))
    tempo2 = ax[2].axvline(0, c='r')
    elements.append(tempo2)

    # animation function.  This is called sequentially
    def animate(i):
        print(f'Progress {round(i / data.shape[0], 2)}% {i}', end='\r')
        elements[0].set_data(data[i, :, :])
        elements[1].set_xdata(time[i])
        elements[2].set_xdata(time[i])
        return elements

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, 
                                   frames=nframes, interval=1, blit=True)
    anim.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])