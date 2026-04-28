import roboticstoolbox as rtb
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import FillingCirclesBar
from tqdm import tqdm
from spatialmath import SE2


def modified_scanmap(posegraph, occgrid, maxrange=None, M=10):
    """
    Generates a binary occupancy grid where a cell has the value 0
    if more than M rays found it unoccupied. Otherwise, the cell has
    the value 1.

    :param posegraph: A roboticstoolbox.PoseGraph object containing the
        relevant lidar scans.

    :param occgrid:
        The occupancy grid the lidar scans should be mapped onto before
        being converted to a binary occupancy grid.

    :param maxrange: The maximum value the lidar returns

    :param M: The minimum number of rays that need to find a cell unoccupied
        for it to be counted as unoccupied
    """
    # note about maxrange timing
    pg = posegraph

    bar = FillingCirclesBar(
        "Converting", max=pg.graph.n, suffix="%(percent).1f%% - %(eta)ds"
    )

    grid1d = occgrid.ravel
    for i in range(pg.graph.n):

        xy = pg.scanxy(i)
        r, theta = pg.scan(i)
        if maxrange is not None:
            toofar = np.where(r > maxrange)[0]
            xy = np.delete(xy, toofar, axis=1)
        xyt = pg.vindex[i].coord

        xy = SE2(xyt) * xy

        # start of each ray
        p1 = occgrid.w2g(xyt[:2])

        for p2 in occgrid.w2g(xy.T):

            # all cells along the ray
            try:
                k = occgrid._line(p1, p2)
            except ValueError:
                # silently ignore rays to points outside the grid map
                continue

            # increment cells along the ray, these are free space
            grid1d[k[:-1]] += 1

        bar.next()
    bar.finish()

    grid1d[grid1d < M] = 0
    grid1d[grid1d >= M] = 1
    bar.finish()
    return rtb.BinaryOccupancyGrid(
        grid=occgrid.grid,
        cellsize=0.1
    )


def plotoccgrid(occgrid):
    occgrid.plot()
    plt.show()


def prmplanning(occgrid, npoints=300):
    """
    A method to perform PRM-planning on an occupancy grid

    :param occgrid: The occupancy grid
    :param npoints: The amount of points generated in PRM-planning
    """
    prm = rtb.PRMPlanner(occgrid=occgrid, seed=0)
    prm.plan(npoints=npoints)
    prm.plot()
    plt.show()


def dstarplanning(occgrid):
    dstar = rtb.DstarPlanner(occgrid=occgrid)
    print("Planning")
    dstar.plan(goal=(120, 190))
    print("Finding path")
    path, status = dstar.query(start=(65, 150))
    dstar.plot(path)
    plt.show()


def astarplanning(occgrid: rtb.OccupancyGrid, start, goal):
    # Dict format:
    # {Coordinate: (g(n), h(n), f(n), parent_coordinate)}
    start = tuple(occgrid.w2g(start).tolist())  # Convert to grid coords
    goal = tuple(occgrid.w2g(goal).tolist())  # Convert to grid coords
    g_start = 0  # The cost to move from start to start is zero
    # Manhattan distance heuristic
    h_start = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    f_start = g_start + h_start  # Estimated travel cost to target
    open_nodes = {start: (g_start, h_start, f_start, None)}
    closed_nodes = {}

    pbar = tqdm(desc="Planning")  # See that iterations happen

    while open_nodes:
        current = None
        lowestf = float('inf')
        # Makes the current node the one with the lowest
        # heuristically estimated distance to target
        for coord, values in open_nodes.items():
            node = (coord,) + values
            if node[3] < lowestf and occgrid.grid[coord] == 0:
                lowestf = node[3]
                current = node

        # Reached goal, constructs path by recursively going
        # back to start through node parents
        if current[0] == goal:
            pbar.close()
            path = [current[0]]
            parent = current[-1]
            while parent is not None:
                path.append(occgrid.g2w(parent))
                parent = closed_nodes[parent][-1]
            path.reverse()
            return path

        # Moves current node from open to closed
        closed_nodes[current[0]] = current[1:]
        del open_nodes[current[0]]

        x, y = current[0]
        # Does not include diagonals
        nghcoords = [
            (x, y+1),
            (x, y-1),
            (x+1, y),
            (x-1, y)
        ]

        # Goes through all neighbour cells, and adds them to open
        # if they are not already evaluated. If a cell is already in
        # open, and is deemed to be a worse path, it is ignored for now
        for nghx, nghy in nghcoords:
            if (nghx, nghy) in closed_nodes:
                # Already evaluated
                continue
            g = current[1] + 1
            if (nghx, nghy) in open_nodes:
                if g >= open_nodes[nghx, nghy][0]:
                    # Not a better path
                    continue
            # Manhattan distance heuristic
            h = abs(nghx - goal[0]) + abs(nghy - goal[1])
            f = g + h
            open_nodes[(nghx, nghy)] = (g, h, f, current[0])
        pbar.update(1)
    # Never reached goal
    print("Path not found")
    return None


def main():
    og = rtb.OccupancyGrid(
        workspace=np.array([-100, 250, -100, 250]),
        cellsize=0.1,
        value=np.int32(0)
        )
    pg = rtb.PoseGraph("data/killian.g2o.zip", lidar=True)
    killian = modified_scanmap(pg, occgrid=og, maxrange=50, M=10)
    path = astarplanning(killian, (65, 150), (55, 160))
    killian.plot()
    plt.plot(path)
    plt.show()


if __name__ == "__main__":
    main()
