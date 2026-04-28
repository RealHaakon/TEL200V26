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
    goal_x, goal_y = goal
    start_x, start_y = start
    g_start = 0  # The cost to move from start to start is zero
    # Manhattan distance heuristic
    h_start = abs(start_x - goal_x) + abs(start_y - goal_y)
    f_start = g_start + h_start  # Estimated travel cost to target
    open_nodes = {start: (g_start, h_start, f_start, None)}
    closed_nodes = {}

    # See that iterations happe and how close the current node is to goal
    pbar = tqdm(total=f_start, desc="Planning, Distance to target: ")

    while open_nodes:
        coord = None
        lowestf = float('inf')
        # Makes the current node the one with the lowest
        # heuristically estimated distance to target
        for tentative_coord, values in open_nodes.items():
            g, h, f, parent = values
            if f < lowestf and occgrid.grid[tentative_coord] == 0:
                lowestf = f
                coord = tentative_coord

        g, h, f, parent = open_nodes[coord]

        # Reached goal, constructs path by recursively going
        # back to start through node parents
        if coord == goal:
            pbar.close()
            path = [coord]
            while parent is not None:
                path.append(occgrid.g2w(parent))
                parent = closed_nodes[parent][-1]
            path.reverse()
            return path

        # Moves current node from open to closed
        closed_nodes[coord] = (g, h, f, parent)
        del open_nodes[coord]

        x, y = coord
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
            ngh_g = g + 1
            if (nghx, nghy) in open_nodes:
                tent_g, tent_h, tent_f, tent_parent = open_nodes[(nghx, nghy)]
                if ngh_g >= tent_g:
                    # Not a better path
                    continue
            # Manhattan distance heuristic
            ngh_h = abs(nghx - goal_x) + abs(nghy - goal_y)
            ngh_f = ngh_g + ngh_h
            open_nodes[(nghx, nghy)] = (ngh_g, ngh_h, ngh_f, coord)
        pbar.reset()
        pbar.update(f)
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
    # killian.plot()
    plt.plot(path)
    plt.show()


if __name__ == "__main__":
    main()
