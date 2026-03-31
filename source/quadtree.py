import numpy as np


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


if __name__ == '__main__':

    xs = np.random.uniform(size=10000)
    ys = np.random.uniform(size=10000)


    Q = QuadTree(0, 1, 0, 1, xs, ys, 25, 10)

    n = [ l['count'] for l in Q.lop]

    print(sum(n))


