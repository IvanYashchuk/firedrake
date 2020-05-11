from firedrake import *
import pytest
import numpy as np
from mpi4py import MPI

# Utility Functions


def cell_midpoints(m):
    """Get the coordinates of the midpoints of every cell in mesh `m`.

    :param m: The mesh to generate cell midpoints for.

    :returns: A tuple of numpy arrays `(midpoints, local_midpoints)` where
    `midpoints` are the midpoints for the entire mesh even if the mesh is
    distributed and `local_midpoints` are the midpoints of only the
    rank-local non-ghost cells."""
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
    return midpoints, local_midpoints


"""Parent meshes used in tests"""
parentmeshes = [
    pytest.param(UnitIntervalMesh(1), marks=pytest.mark.xfail(reason="swarm not implemented in 1d")),
    UnitSquareMesh(1, 1),
    UnitCubeMesh(1, 1, 1)
]


# pic swarm tests

@pytest.mark.parametrize("parentmesh", parentmeshes)
def test_pic_swarm_in_plex(parentmesh):
    """Generate points in cell midpoints of mesh `parentmesh` and check correct
    swarm is created in plex."""

    # Setup

    parentmesh.init()
    inputpointcoords, inputlocalpointcoords = cell_midpoints(parentmesh)
    plex = parentmesh.topology._plex
    swarm = mesh._pic_swarm_in_plex(plex, inputpointcoords)
    # Get point coords on current MPI rank
    localpointcoords = np.copy(swarm.getField("DMSwarmPIC_coor"))
    swarm.restoreField("DMSwarmPIC_coor")
    if len(inputpointcoords.shape) > 1:
        localpointcoords = np.reshape(localpointcoords, (-1, inputpointcoords.shape[1]))
    # Turn this into a number of points locally and MPI globally before
    # doing any tests to avoid making tests hang should a failure occur
    # on not all MPI ranks
    nptslocal = len(localpointcoords)
    nptsglobal = MPI.COMM_WORLD.allreduce(nptslocal, op=MPI.SUM)
    # Get parent PETSc cell indices on current MPI rank
    localparentcellindices = np.copy(swarm.getField("DMSwarm_cellid"))
    swarm.restoreField("DMSwarm_cellid")

    # Tests

    # Check comm sizes match
    assert plex.comm.size == swarm.comm.size
    # Check coordinate list and parent cell indices match
    assert len(localpointcoords) == len(localparentcellindices)
    # check local points are found in list of input points
    for p in localpointcoords:
        assert np.any(np.isclose(p, inputpointcoords))
    # check local points are correct local points given mesh
    # partitioning (but don't require ordering to be maintained)
    assert len(localpointcoords) == len(inputlocalpointcoords)
    assert np.all(np.isin(inputlocalpointcoords, localpointcoords))
    # Check methods for checking number of points on current MPI rank
    assert len(localpointcoords) == swarm.getLocalSize()
    # Check there are as many local points as there are local cells
    # (excluding ghost cells in the halo)
    assert len(localpointcoords) == parentmesh.cell_set.size
    # Check total number of points on all MPI ranks is correct
    # (excluding ghost cells in the halo)
    assert nptsglobal == len(inputpointcoords)
    assert nptsglobal == swarm.getSize()
    # Check the parent cell indexes match those in the parent mesh
    cell_indexes = parentmesh.cell_closure[:, -1]
    for index in localparentcellindices:
        assert np.any(index == cell_indexes)


@pytest.mark.parallel
@pytest.mark.parametrize("parentmesh", parentmeshes)
def test_pic_swarm_in_plex_parallel(parentmesh):
    test_pic_swarm_in_plex(parentmesh)


@pytest.mark.parallel(nprocs=2)  # nprocs == total number of mesh cells
def test_pic_swarm_in_plex_2d_2procs():
    test_pic_swarm_in_plex(UnitSquareMesh(1, 1))


@pytest.mark.parallel(nprocs=3)  # nprocs > total number of mesh cells
def test_pic_swarm_in_plex_2d_3procs():
    test_pic_swarm_in_plex(UnitSquareMesh(1, 1))


# Mesh Generation Tests


def verify_vertexonly_mesh(m, vm, inputvertexcoords, inputvertexcoordslocal, gdim):
    """Assumes all inputvertexcoordslocal are the correct process local
    vertex coords in the vm"""
    assert m.geometric_dimension() == gdim
    # Correct dims
    assert vm.geometric_dimension() == gdim
    assert vm.topological_dimension() == 0
    # Can initialise
    vm.init()
    # Correct coordinates (though not guaranteed to be in same order)
    assert np.shape(vm.coordinates.dat.data_ro) == np.shape(inputvertexcoordslocal)
    assert np.all(np.isin(inputvertexcoordslocal, vm.coordinates.dat.data_ro))
    # Coordinates located in correct cells of parent mesh
    V = VectorFunctionSpace(m, "DG", 0)
    f = Function(V).interpolate(m.coordinates)
    for i in range(len(vm.coordinates.dat.data_ro)):
        assert all(f.dat.data_ro[m.locate_cell(vm.coordinates.dat.data_ro[i])] == vm.coordinates.dat.data_ro[i])
    # Correct parent topology
    assert vm._parent_mesh is m.topology
    # Check other properties
    assert np.shape(vm.cell_closure) == (len(inputvertexcoordslocal), 1)
    with pytest.raises(AttributeError):
        vm.cell_to_facets
    assert vm.num_cells() == len(inputvertexcoordslocal) == vm.cell_set.size
    assert vm.num_facets() == 0
    assert vm.num_faces() == vm.num_entities(2) == 0
    assert vm.num_edges() == vm.num_entities(1) == 0
    assert vm.num_vertices() == vm.num_entities(0) == vm.num_cells()


@pytest.mark.parametrize("parentmesh", parentmeshes)
def test_generate(parentmesh):
    inputcoords, inputcoordslocal = cell_midpoints(parentmesh)
    vm = VertexOnlyMesh(parentmesh, inputcoords)
    verify_vertexonly_mesh(parentmesh, vm, inputcoords, inputcoordslocal, parentmesh.geometric_dimension())


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize("parentmesh", parentmeshes)
def test_generate_parallel(parentmesh):
    test_generate(parentmesh)


@pytest.mark.parametrize("parentmesh", parentmeshes)
@pytest.mark.xfail(raises=NotImplementedError)
def test_extrude(parentmesh):
    inputcoords, inputcoordslocal = cell_midpoints(parentmesh)
    vm = VertexOnlyMesh(parentmesh, inputcoords)
    ExtrudedMesh(vm, 1)


# Mesh usage tests


def functionspace_tests(vm, family, degree):
    # Prep: Get number of cells
    num_cells_mpi_global = MPI.COMM_WORLD.allreduce(vm.num_cells(), op=MPI.SUM)
    # Can create function space
    V = FunctionSpace(vm, family, degree)
    # Can create function on function spaces
    f = Function(V)
    g = Function(V)
    # Can interpolate and Galerkin project onto functions
    gdim = vm.geometric_dimension()
    if gdim == 1:
        x, = SpatialCoordinate(vm)
        f.interpolate(x)
        g.project(x)
    elif gdim == 2:
        x, y = SpatialCoordinate(vm)
        f.interpolate(x+y)
        g.project(x+y)
    elif gdim == 3:
        x, y, z = SpatialCoordinate(vm)
        f.interpolate(x+y+z)
        g.project(x+y+z)
    # Get exact values at coordinates with maintained ordering
    assert np.shape(f.dat.data_ro)[0] == np.shape(vm.coordinates.dat.data_ro)[0]
    assert np.allclose(f.dat.data_ro, np.sum(vm.coordinates.dat.data_ro, 1))
    # Projection is the same as interpolation
    assert np.allclose(f.dat.data_ro, g.dat.data_ro)
    # Assembly works as expected
    f.interpolate(Constant(2))
    assert np.isclose(assemble(f*dx), 2*num_cells_mpi_global)


def vectorfunctionspace_tests(vm, family, degree):
    # Can create function space
    V = VectorFunctionSpace(vm, family, degree)
    # Can create function on function spaces
    f = Function(V)
    # Can interpolate onto functions
    x = SpatialCoordinate(vm)
    f.interpolate(2*as_vector(x))
    # Get exact values at coordinates with maintained ordering
    assert np.shape(f.dat.data_ro)[0] == np.shape(vm.coordinates.dat.data_ro)[0]
    assert np.allclose(f.dat.data_ro, 2*vm.coordinates.dat.data_ro)
    # TODO add assembly and Galerkin projection


"""Families and degrees to test function spaces on VertexOnlyMesh"""
families_and_degrees = [
    ("DG", 0),
    pytest.param("DG", 1, marks=pytest.mark.xfail(reason="unsupported degree")),
    pytest.param("CG", 1, marks=pytest.mark.xfail(reason="unsupported family and degree"))
]


@pytest.mark.parametrize("parentmesh", parentmeshes)
@pytest.mark.parametrize(("family", "degree"), families_and_degrees)
def test_functionspaces(parentmesh, family, degree):
    vertexcoords, vertexcoordslocal = cell_midpoints(parentmesh)
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    functionspace_tests(vm, family, degree)
    vectorfunctionspace_tests(vm, family, degree)


@pytest.mark.parallel
@pytest.mark.parametrize("parentmesh", parentmeshes)
@pytest.mark.parametrize(("family", "degree"), families_and_degrees)
def test_functionspaces_parallel(parentmesh, family, degree):
    test_functionspaces(parentmesh, family, degree)
