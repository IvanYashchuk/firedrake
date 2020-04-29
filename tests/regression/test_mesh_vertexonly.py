from firedrake import *
import pytest
import numpy as np
from mpi4py import MPI

# Utility Functions

def cell_midpoints(m):
    """Get the coordinates of the midpoints of every cell in mesh `m`.
    The mesh may be distributed, but the midpoints are returned for the
    entire mesh as though it were not distributed."""
    m.init()
    V = VectorFunctionSpace(m, "DG", 0)
    f = Function(V).interpolate(m.coordinates)
    # since mesh may be distributed, the number of cells on the MPI rank
    # may not be the same on all ranks (note we exclude ghost cells
    # hence using num_cells_local = m.cell_set.size). Below local means
    # MPI rank local.
    num_cells_local = m.cell_set.size
    num_cells = MPI.COMM_WORLD.allreduce(num_cells_local, op=MPI.SUM)
    local_midpoints = f.dat.data_ro
    local_midpoints_size = np.array(local_midpoints.size)
    local_midpoints_sizes = np.empty(MPI.COMM_WORLD.size, dtype=int)
    MPI.COMM_WORLD.Allgatherv(local_midpoints_size, local_midpoints_sizes)
    midpoints = np.empty((num_cells, m.cell_dimension()), dtype=float)
    MPI.COMM_WORLD.Allgatherv(local_midpoints, (midpoints, local_midpoints_sizes))
    assert len(np.unique(midpoints, axis=0)) == len(midpoints)
    return midpoints

"""Parent meshes used in tests"""
parentmeshes = [
    pytest.param(UnitIntervalMesh(1), marks=pytest.mark.xfail(reason="swarm not implemented in 1d")),
    UnitSquareMesh(1,1),
    UnitCubeMesh(1,1,1)
]

# pic swarm tests

@pytest.mark.parametrize("parentmesh", parentmeshes)
def test_pic_swarm_in_plex(parentmesh):
    """Generate points in cell midpoints of mesh `parentmesh` and check correct
    swarm is created in plex."""

    # Setup
    
    parentmesh.init()
    pointcoords = cell_midpoints(parentmesh)
    plex = parentmesh.topology._plex
    swarm = mesh._pic_swarm_in_plex(plex, pointcoords)
    # Get point coords on current MPI rank
    localpointcoords = np.copy(swarm.getField("DMSwarmPIC_coor"))
    swarm.restoreField("DMSwarmPIC_coor")
    if len(pointcoords.shape) > 1:
        localpointcoords = np.reshape(localpointcoords, (-1, pointcoords.shape[1]))
    # Turn this into a number of points locally and MPI globally before 
    # doing any tests to avoid making tests hang should a failure occur 
    # on not all MPI ranks
    nptslocal = len(localpointcoords)
    nptsglobal = MPI.COMM_WORLD.allreduce(nptslocal, op=MPI.SUM)

    # Tests

    # Check comm sizes match
    assert plex.comm.size == swarm.comm.size
    # check local points are found in list of points
    for p in localpointcoords:
        assert np.any(np.isclose(p, pointcoords))
    # Check methods for checking number of points on current MPI rank
    assert len(localpointcoords) == swarm.getLocalSize()
    # Check there are as many local points as there are local cells
    # (excluding ghost cells in the halo)
    assert len(localpointcoords) == parentmesh.cell_set.size
    # Check total number of points on all MPI ranks is correct
    # (excluding ghost cells in the halo)
    assert nptsglobal == len(pointcoords)
    assert nptsglobal == swarm.getSize()
    # Check each cell has the correct point associated with it
    #TODO

@pytest.mark.parallel
@pytest.mark.parametrize("parentmesh", parentmeshes)
def test_pic_swarm_in_plex_parallel(parentmesh):
    test_pic_swarm_in_plex(parentmesh)

@pytest.mark.parallel(nprocs=2) # nprocs == total number of mesh cells
def test_pic_swarm_in_plex_2d_2procs():
    test_pic_swarm_in_plex(UnitSquareMesh(1,1))

@pytest.mark.parallel(nprocs=3) ## nprocs > total number of mesh cells
def test_pic_swarm_in_plex_2d_3procs():
    test_pic_swarm_in_plex(UnitSquareMesh(1,1))

# Mesh Generation Tests

def verify_vertexonly_mesh(m, vm, vertexcoords, gdim):
    """Assumes all vertexcoords are all in vm"""
    assert m.geometric_dimension() == gdim
    # Correct dims
    assert vm.geometric_dimension() == gdim
    assert vm.topological_dimension() == 0
    # Can initialise
    vm.init()
    # Correct coordinates
    assert np.shape(vm.coordinates.dat.data_ro) == np.shape(vertexcoords)
    assert np.all(np.isin(vm.coordinates.dat.data_ro, vertexcoords))
    # Correct parent topology
    assert vm._parent_mesh is m.topology
    # Check other properties
    assert np.shape(vm.cell_closure) == (len(vertexcoords), 1)
    with pytest.raises(AttributeError):
        vm.cell_to_facets
    assert vm.num_cells() == len(vertexcoords)
    assert vm.num_facets() == 0
    assert vm.num_faces() == vm.num_entities(2) == 0
    assert vm.num_edges() == vm.num_entities(1) == 0
    assert vm.num_vertices() == vm.num_entities(0) == vm.num_cells()


@pytest.mark.parametrize("parentmesh", parentmeshes)
def test_generate(parentmesh):
    vertexcoords = cell_midpoints(parentmesh)
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    verify_vertexonly_mesh(parentmesh, vm, vertexcoords, parentmesh.geometric_dimension())

@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize("parentmesh", parentmeshes)
def test_generate_parallel(parentmesh):
    test_generate(parentmesh)

# Mesh use tests

def _test_functionspace(vm, family, degree):
    # Can create function space
    V = FunctionSpace(vm, family, degree)
    # Can create function on function spaces
    f = Function(V)
    # Can interpolate onto functions
    gdim = vm.geometric_dimension()
    if gdim == 1:
        x, = SpatialCoordinate(vm)
        f.interpolate(x)
    elif gdim == 2:
        x, y = SpatialCoordinate(vm)
        f.interpolate(x+y)
    elif gdim == 3:
        x, y, z = SpatialCoordinate(vm)
        f.interpolate(x+y+z)
    # Get exact values at coordinates
    assert np.shape(f.dat.data_ro)[0] == np.shape(vm.coordinates.dat.data_ro)[0]
    # assert f.dat.data_ro == sum(vm.coordinates.dat.data_ro , #over correct index - should be in correct order - also use allclose)
    for coord in vm.coordinates.dat.data_ro:
        # .at doesn't work on immersed manifolds
        # assert f.at(coord) == sum(coord)
        assert np.isin(sum(coord), f.dat.data_ro)

def _test_vectorfunctionspace(vm, family, degree):
    # Can create function space
    V = VectorFunctionSpace(vm, family, degree)
    # Can create function on function spaces
    f = Function(V)
    # Can interpolate onto functions
    gdim = vm.geometric_dimension()
    x = SpatialCoordinate(vm)
    f.interpolate(2*as_vector(x))
    # Get exact values at coordinates
    assert np.shape(f.dat.data_ro)[0] == np.shape(vm.coordinates.dat.data_ro)[0]
    for coord in vm.coordinates.dat.data_ro:
        # .at doesn't work on immersed manifolds
        # assert f.at(coord) == sum(coord)
        assert np.all(np.isin(2*coord, f.dat.data_ro))

"""Families and degrees to test function spaces on VertexOnlyMesh"""
families_and_degrees = [
    ("DG", 0),
    pytest.param("DG", 1, marks=pytest.mark.xfail(reason="unsupported degree")),
    pytest.param("CG", 1, marks=pytest.mark.xfail(reason="unsupported family and degree"))
]

@pytest.mark.parametrize("parentmesh", parentmeshes)
@pytest.mark.parametrize(("family", "degree"), families_and_degrees)
def test_functionspaces(parentmesh, family, degree):
    vertexcoords = cell_midpoints(parentmesh)
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    _test_functionspace(vm, family, degree)
    _test_vectorfunctionspace(vm, family, degree)

@pytest.mark.parallel
@pytest.mark.parametrize("parentmesh", parentmeshes)
@pytest.mark.parametrize(("family", "degree"), families_and_degrees)
def test_functionspaces_parallel(parentmesh, family, degree):
    test_functionspaces(parentmesh, family, degree)

# remove this before final merge
if __name__ == "__main__":
    test_generate_2d()
    import pytest, sys
    pytest.main([sys.argv[0]])
