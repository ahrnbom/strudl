""" A module which provides an easy interface for matplotlib for simple plots """

import matplotlib as mpl
mpl.use('Agg') # This is necessary since we're running inside Docker, without a graphical interface to draw on
import matplotlib.pyplot as plt

def multi_plot(xs, ys, filepath, xlabel=None, ylabel=None, linewidth=2, xlim=None, ylim=None, style=None, styles=None, title=None, legend=None):
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    
    lines = []
    if styles is None:
        for x, y in zip(xs, ys):
            if style is None:
                # Who thought it was a good idea to return a tuple from ax.plot?
                # This syntax is error-prone
                curr_line, = ax.plot(x,y, linewidth=linewidth)
            else:
                curr_line, = ax.plot(x,y, style, linewidth=linewidth)
                
            lines.append(curr_line)
    else:
        for x, y, style in zip(xs, ys, styles):
            curr_line, = ax.plot(x, y, style, linewidth=linewidth)
            lines.append(curr_line)
            
    if not (xlabel is None):
        plt.xlabel(xlabel)
    
    if not (ylabel is None):
        plt.ylabel(ylabel)

    if not (xlim is None):
        plt.xlim(xlim)
    
    if not (ylim is None):
        plt.ylim(ylim)    
        
    if not (title is None):
        plt.title(title)
        
    if not (legend is None):
        plt.legend(lines, legend)
            
    plt.savefig(filepath, dpi=300) # dpi sets the size of the output image
    plt.close(fig)

def simple_plot(x, y, filepath, xlabel=None, ylabel=None, linewidth=2, xlim=None, ylim=None):
    """ Takes two lists, x and y, and draws a simple plot. """
    multi_plot([x], [y], filepath, xlabel=xlabel, ylabel=ylabel, linewidth=linewidth, xlim=xlim, ylim=ylim)
