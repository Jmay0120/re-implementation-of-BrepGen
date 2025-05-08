from plyfile import PlyData
from mayavi import mlab

ply = PlyData.read("./abc_test_pcd/00657710_8a44129e11ca47db73115d55_step_100.ply")
vtx = ply['vertex']
mlab.points3d(vtx['x'], vtx['y'], vtx['z'])
mlab.show()
