#from mayavi import mlab
#import mayavi.mlab as mlab
from matplotlib import animation
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
def visualize_points(points_before:np.ndarray,points:np.ndarray, vis_num_tresh = 500000):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #set colour uniform mapping
    color_map = plt.get_cmap('cividis')
    #initialize plot by plotting all points
    def init():
        ax.scatter(points_before[:, 0], points_before[:, 1], points_before[:, 2], c=points_before[:,3],cmap=color_map, s=0.02, alpha=0.1)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', s=0.02, alpha=1)
        ax.legend(['before', 'after'])
        return fig,
    #create animation by rotating the plot
    def animate(i):
        print(i)
        ax.view_init(elev=10, azim=(i * 30) % 360)
        return fig,
    #set parameters for animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=12, interval=3000, blit=True)
    #save animation
    anim.save('animation.gif', writer="pillow", fps=1)
