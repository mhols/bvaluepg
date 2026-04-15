import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class QuadTree:

    def __init__(self, xmin, xmax, ymin, ymax, x, y, countmax, maxdepth):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.x = x
        self.y = y
        self.countmax = countmax
        self.maxdepth = maxdepth

        self.lop = []

        self.quadtree()


    def quadtree(self):

        def _quadtree(xmin, xmax, ymin, ymax, x, y, depth):

            if len(x)<=self.countmax or depth >= self.maxdepth:
                self.lop.append(
                    {
                        'xmin': xmin,
                        'xmax': xmax,
                        'ymin': ymin,
                        'ymax': ymax,
                        'count' : len(x) 
                    }
                )
                return
            
            # else...


             # Mittelpunkte berechnen
            midx = 0.5 * (xmin + xmax)
            midy = 0.5 * (ymin + ymax)
            # Punkte den vier Quadranten zuordnen
            # Südwesten (SW):
            I = (x < midx) & (y< midy)
            x_sw = x[I]
            y_sw = y[I]

            # Südosten (SE): 
            I = (x>=midx) & (y<midy)
            x_se = x[I]
            y_se = y[I]

            # Nordwesten (NW): 
            I = (x<midx) & (y>=midy)
            x_nw = x[I]
            y_nw = y[I]

            # Nordosten (NE): 
            I = (x>=midx) & (y>=midy)
            x_ne = x[I]
            y_ne = y[I]

            # Rekursion für jeden Unterbereich
            _quadtree(xmin, midx, ymin, midy, x_sw, y_sw, depth+1)
            
            _quadtree(midx, xmax, ymin, midy, x_se, y_se, depth+1)
            
            _quadtree(xmin, midx, midy, ymax, x_nw, y_nw, depth+1)
            
            _quadtree(midx, xmax, midy, ymax, x_ne, y_ne, depth+1)

        _quadtree(self.xmin, self.xmax, self.ymin, self.ymax, self.x, self.y, 0)


    def plot(self, ax=None, show_points=True, cmap='viridis', color_by_count=True, alpha=0.6):
        """
        Plot des QuadTrees als Rechtecke.

        Parameters
        ----------
        ax : matplotlib axis, optional
            Falls None, wird eine neue Figure/Achse erzeugt.
        show_points : bool
            Falls True, werden die Originalpunkte zusätzlich geplottet.
        cmap : str
            Matplotlib-Colormap für die Einfärbung der Zellen.
        color_by_count : bool
            Falls True, werden die Zellen nach ihrer Punktzahl eingefärbt.
            Falls False, werden nur die Ränder gezeichnet.
        alpha : float
            Transparenz der Rechtecke.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure

        counts = np.array([cell['count'] for cell in self.lop], dtype=float)

        if len(counts) == 0:
            raise ValueError("QuadTree enthält keine Zellen zum Plotten.")

        if color_by_count:
            norm = plt.Normalize(vmin=counts.min(), vmax=counts.max())
            cmap_obj = plt.get_cmap(cmap)
        else:
            norm = None
            cmap_obj = None

        for cell in self.lop:
            width = cell['xmax'] - cell['xmin']
            height = cell['ymax'] - cell['ymin']

            if color_by_count:
                facecolor = cmap_obj(norm(cell['count']))
            else:
                facecolor = 'none'

            rect = Rectangle(
                (cell['xmin'], cell['ymin']),
                width,
                height,
                facecolor=facecolor,
                edgecolor='black',
                linewidth=0.5,
                alpha=alpha if color_by_count else 1.0,
            )
            ax.add_patch(rect)

        if show_points:
            ax.scatter(self.x, self.y, s=4, c='red', alpha=0.5)

        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_aspect('equal')
        ax.set_title('QuadTree')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        if color_by_count:
            sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label='count')

        return ax



    def cell_areas(self):
        """
        Calculates the area of each QuadTree cell/patch.

        Returns
        -------
        np.ndarray
            Array of cell areas (in coordinate units).
        """
        return np.array([
            (cell['xmax'] - cell['xmin']) * (cell['ymax'] - cell['ymin'])
            for cell in self.lop
        ], dtype=float)

if __name__ == '__main__':

    xs = np.random.uniform(size=1000)
    ys = np.random.uniform(size=1000)


    Q = QuadTree(0, 1, 0, 1, xs, ys, 25, 10)

    n = [ l['count'] for l in Q.lop]

    print("Total number of points:", sum(n))
    areas = Q.cell_areas()

    print("Number of cells:", len(areas))
    print("First 5 cell areas:", areas[:5])

    Q.plot()
    plt.show()
