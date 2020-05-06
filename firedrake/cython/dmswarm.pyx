# cython: language_level=3

# Utility functions to derive global and local numbering from DMSwarm
import cython
import numpy as np
from firedrake.petsc import PETSc
from mpi4py import MPI
from pyop2.datatypes import IntType
from libc.string cimport memset
from libc.stdlib cimport qsort
cimport numpy as np
cimport mpi4py.MPI as MPI
cimport petsc4py.PETSc as PETSc

np.import_array()

include "petschdr.pxi"

@cython.boundscheck(False)
@cython.wraparound(False)
def mark_entity_classes(PETSc.DM swarm):
    """Mark all points in a given DMSwarm according to the PyOP2 entity
    classes:

    core   : owned and not in send halo
    owned  : owned and in send halo
    ghost  : in halo

    Note that this is a copy and paste job from the equivalent function
    in dmplex.pyx with the relevant modifications made for a DMSwarm.

    :arg swarm: The DMSwarm object encapsulating the mesh topology
    """
    cdef:
        PetscInt pStart, pEnd, cStart, cEnd
        PetscInt c, ci, p
        PetscInt nleaves
        const PetscInt *ilocal = NULL
        PetscBool non_exec
        const PetscSFNode *iremote = NULL
        PETSc.SF point_sf = None
        PetscBool is_ghost, is_owned
        DMLabel lbl_core, lbl_owned, lbl_ghost

    pStart = 0
    pEnd = swarm.getSize()

    swarm.createLabel("pyop2_core")
    swarm.createLabel("pyop2_owned")
    swarm.createLabel("pyop2_ghost")

    CHKERR(DMGetLabel(swarm.dm, b"pyop2_core", &lbl_core))
    CHKERR(DMGetLabel(swarm.dm, b"pyop2_owned", &lbl_owned))
    CHKERR(DMGetLabel(swarm.dm, b"pyop2_ghost", &lbl_ghost))

    if swarm.comm.size > 1:
        # Mark ghosts from point overlap SF
        point_sf = swarm.getPointSF()
        CHKERR(PetscSFGetGraph(point_sf.sf, NULL, &nleaves, &ilocal, NULL))
        for p in range(nleaves):
            CHKERR(DMLabelSetValue(lbl_ghost, ilocal[p], 1))
    else:
        # If sequential mark all points as core
        for p in range(pStart, pEnd):
            CHKERR(DMLabelSetValue(lbl_core, p, 1))
        return

    CHKERR(DMLabelCreateIndex(lbl_ghost, pStart, pEnd))

    # Mark all remaining points as core
    CHKERR(DMLabelCreateIndex(lbl_owned, pStart, pEnd))
    for p in range(pStart, pEnd):
        CHKERR(DMLabelHasPoint(lbl_owned, p, &is_owned))
        CHKERR(DMLabelHasPoint(lbl_ghost, p, &is_ghost))
        if not is_ghost and not is_owned:
            CHKERR(DMLabelSetValue(lbl_core, p, 1))
    CHKERR(DMLabelDestroyIndex(lbl_owned))
    CHKERR(DMLabelDestroyIndex(lbl_ghost))

@cython.boundscheck(False)
@cython.wraparound(False)
def get_entity_classes(PETSc.DM swarm):
    """Builds PyOP2 entity class offsets for all entity levels.

    Note that this is a copy and paste job from the equivalent function
    in dmplex.pyx with the relevant modifications made for a DMSwarm.

    :arg swarm: The DMSwarm object encapsulating the mesh topology
    """
    cdef:
        np.ndarray[PetscInt, ndim=2, mode="c"] entity_class_sizes
        np.ndarray[PetscInt, mode="c"] eStart, eEnd
        PetscInt depth, d, i, ci, class_size, start, end
        const PetscInt *indices = NULL
        PETSc.IS class_is

    depth = 1 # by definition since a swarm is point cloud
    entity_class_sizes = np.zeros((depth, 3), dtype=IntType)
    eStart = np.zeros(depth, dtype=IntType)
    eEnd = np.zeros(depth, dtype=IntType)
    for d in range(depth):
        start = 0 # by definition since a swarm is point cloud
        CHKERR(DMSwarmGetSize(swarm.dm, &end)) # by definition since a swarm is point cloud
        eStart[d] = start
        eEnd[d] = end

    for i, op2class in enumerate([b"pyop2_core",
                                  b"pyop2_owned",
                                  b"pyop2_ghost"]):
        class_is = swarm.getStratumIS(op2class, 1)
        class_size = swarm.getStratumSize(op2class, 1)
        if class_size > 0:
            CHKERR(ISGetIndices(class_is.iset, &indices))
            for ci in range(class_size):
                for d in range(depth):
                    if eStart[d] <= indices[ci] < eEnd[d]:
                        entity_class_sizes[d, i] += 1
                        break
            CHKERR(ISRestoreIndices(class_is.iset, &indices))

    # PyOP2 entity class indices are additive
    for d in range(depth):
        for i in range(1, 3):
            entity_class_sizes[d, i] += entity_class_sizes[d, i-1]
    return entity_class_sizes

def create_section(mesh, nodes_per_entity, on_base=False):
    """Create the section describing a global numbering.

    Note that this is a copy and paste job from the equivalent function
    in dmplex.pyx with the relevant modifications made for a DMSwarm.

    :arg mesh: The mesh.
    :arg nodes_per_entity: Number of nodes on each type of topological
        entity of the mesh.
    :arg on_base: If True, assume extruded space is actually Foo x Real.

    :returns: A PETSc Section providing the number of dofs, and offset
        of each dof, on each mesh point.
    """
    # We don't use DMPlexCreateSection because we only ever put one
    # field in each section.
    cdef:
        PETSc.DM dm
        PETSc.Section section
        # PETSc.IS renumbering
        PetscInt i, p, layers, pStart, pEnd
        PetscInt dimension, ndof
        np.ndarray[PetscInt, ndim=2, mode="c"] nodes
        np.ndarray[PetscInt, ndim=2, mode="c"] layer_extents
        bint variable, extruded, on_base_

    variable = mesh.variable_layers
    extruded = mesh.cell_set._extruded
    on_base_ = on_base
    nodes_per_entity = np.asarray(nodes_per_entity, dtype=IntType)
    if variable:
        layer_extents = mesh.layer_extents
    elif extruded:
        if on_base:
            nodes_per_entity = sum(nodes_per_entity[:, i] for i in range(2))
        else:
            nodes_per_entity = sum(nodes_per_entity[:, i]*(mesh.layers - i) for i in range(2))

    dm = mesh._swarm
    # renumbering = mesh._swarm_renumbering
    section = PETSc.Section().create(comm=mesh.comm)
    pStart = 0 # by definition since point cloud
    pEnd = dm.getSize() # by definition since point cloud
    section.setChart(pStart, pEnd)
    # CHKERR(PetscSectionSetPermutation(section.sec, renumbering.iset))
    dimension = 0 # by definition since point cloud

    nodes = nodes_per_entity.reshape(dimension + 1, -1)

    for i in range(dimension + 1):
        pStart = 0 # by definition since a swarm is point cloud
        CHKERR(DMSwarmGetSize(dm.dm, &pEnd)) # by definition since a swarm is point cloud
        if not variable:
            ndof = nodes[i, 0]
        for p in range(pStart, pEnd):
            if variable:
                if on_base_:
                    ndof = nodes[i, 1]
                else:
                    layers = layer_extents[p, 1] - layer_extents[p, 0]
                    ndof = layers*nodes[i, 0] + (layers - 1)*nodes[i, 1]
            CHKERR(PetscSectionSetDof(section.sec, p, ndof))
    section.setUp()
    return section

@cython.boundscheck(False)
@cython.wraparound(False)
def closure_ordering(PETSc.DM swarm,
                     PETSc.Section vertex_numbering,
                     PETSc.Section cell_numbering,
                     np.ndarray[PetscInt, ndim=1, mode="c"] entity_per_cell):
    """Apply Fenics local numbering to a cell closure.

    Note that this is a copy and paste job from the equivalent function
    in dmplex.pyx with the relevant modifications made for a DMSwarm.

    :arg swarm: The DMSwarm object encapsulating the vertex-only mesh topology
    :arg vertex_numbering: Section describing the universal vertex numbering
    :arg cell_numbering: Section describing the global cell numbering
    :arg entity_per_cell: List of the number of entity points in each dimension

    Vertices    := Ordered according to global/universal
                   vertex numbering
    Edges/faces := Ordered according to lexicographical
                   ordering of non-incident vertices
    """
    cdef:
        PetscInt c, cStart, cEnd, v, vStart, vEnd
        PetscInt f, fStart, fEnd, e, eStart, eEnd
        PetscInt dim, vi, ci, fi, v_per_cell, cell
        PetscInt offset, cell_offset, nfaces, nfacets
        PetscInt *vertices = NULL
        PetscInt *v_global = NULL
        PetscInt closure
        PetscInt *facets = NULL
        PetscInt *faces = NULL
        PetscInt *face_indices = NULL
        const PetscInt *face_vertices = NULL
        PetscInt *facet_vertices = NULL
        np.ndarray[PetscInt, ndim=2, mode="c"] cell_closure

    dim = 0 # by definition since point cloud
    cStart = 0
    cEnd = swarm.getSize()
    vStart = cStart
    vEnd = cEnd
    v_per_cell = entity_per_cell[0]
    cell_offset = sum(entity_per_cell) - 1

    CHKERR(PetscMalloc1(v_per_cell, &vertices))
    CHKERR(PetscMalloc1(v_per_cell, &v_global))
    cell_closure = np.empty((cEnd - cStart, sum(entity_per_cell)), dtype=IntType)

    for c in range(cStart, cEnd):
        CHKERR(PetscSectionGetOffset(cell_numbering.sec, c, &cell))
        closure = c # by definition since just the vertex

        # Find vertices and translate universal numbers
        vi = 0
        if vStart <= closure < vEnd:
            vertices[vi] = closure
            CHKERR(PetscSectionGetOffset(vertex_numbering.sec,
                                            closure, &v))
            # Correct -ve offsets for non-owned entities
            if v >= 0:
                v_global[vi] = v
            else:
                v_global[vi] = -(v+1)
            vi += 1

        # Sort vertices by universal number
        CHKERR(PetscSortIntWithArray(v_per_cell,v_global,vertices))
        for vi in range(v_per_cell):
            cell_closure[cell, vi] = vertices[vi]
        offset = v_per_cell

        # The cell itself is always the first entry in the Swarm closure
        cell_closure[cell, cell_offset] = closure

    CHKERR(PetscFree(vertices))
    CHKERR(PetscFree(v_global))

    return cell_closure

@cython.boundscheck(False)
@cython.wraparound(False)
def get_cell_nodes(mesh,
                   PETSc.Section global_numbering,
                   entity_dofs,
                   np.ndarray[PetscInt, ndim=1, mode="c"] offset):
    """
    Builds the DoF mapping.

    Note that this is a copy and paste job from the equivalent function
    in dmplex.pyx with the relevant modifications made for a DMSwarm.

    :arg mesh: The mesh
    :arg global_numbering: Section describing the global DoF numbering
    :arg entity_dofs: FInAT element entity dofs for the cell
    :arg offset: offsets for each entity dof walking up a column.

    Preconditions: This function assumes that cell_closures contains mesh
    entities ordered by dimension, i.e. vertices first, then edges, faces, and
    finally the cell. For quadrilateral meshes, edges corresponding to
    dimension (0, 1) in the FInAT element must precede edges corresponding to
    dimension (1, 0) in the FInAT element.
    """
    cdef:
        int *ceil_ndofs = NULL
        int *flat_index = NULL
        PetscInt nclosure, dofs_per_cell
        PetscInt c, i, j, k, cStart, cEnd, cell
        PetscInt entity, ndofs, off
        PETSc.Section cell_numbering
        np.ndarray[PetscInt, ndim=2, mode="c"] cell_nodes
        np.ndarray[PetscInt, ndim=2, mode="c"] layer_extents
        np.ndarray[PetscInt, ndim=2, mode="c"] cell_closures
        bint variable

    variable = mesh.variable_layers
    cell_closures = mesh.cell_closure
    if variable:
        layer_extents = mesh.layer_extents
        if offset is None:
            raise ValueError("Offset cannot be None with variable layer extents")

    nclosure = cell_closures.shape[1]

    # Extract ordering from FInAT element entity DoFs
    ndofs_list = []
    flat_index_list = []

    for dim in sorted(entity_dofs.keys()):
        for entity_num in xrange(len(entity_dofs[dim])):
            dofs = entity_dofs[dim][entity_num]

            ndofs_list.append(len(dofs))
            flat_index_list.extend(dofs)

    # Coerce lists into C arrays
    assert nclosure == len(ndofs_list)
    dofs_per_cell = len(flat_index_list)

    CHKERR(PetscMalloc1(nclosure, &ceil_ndofs))
    CHKERR(PetscMalloc1(dofs_per_cell, &flat_index))

    for i in range(nclosure):
        ceil_ndofs[i] = ndofs_list[i]
    for i in range(dofs_per_cell):
        flat_index[i] = flat_index_list[i]

    # Fill cell nodes
    cStart = 0
    cEnd = mesh._swarm.getSize()
    cell_nodes = np.empty((cEnd - cStart, dofs_per_cell), dtype=IntType)
    cell_numbering = mesh._cell_numbering
    for c in range(cStart, cEnd):
        k = 0
        CHKERR(PetscSectionGetOffset(cell_numbering.sec, c, &cell))
        for i in range(nclosure):
            entity = cell_closures[cell, i]
            CHKERR(PetscSectionGetDof(global_numbering.sec, entity, &ndofs))
            if ndofs > 0:
                CHKERR(PetscSectionGetOffset(global_numbering.sec, entity, &off))
                # The cell we're looking at the entity through is
                # higher than the lowest cell the column touches, so
                # we need to offset by the difference from the bottom.
                if variable:
                    off += offset[flat_index[k]]*(layer_extents[c, 0] - layer_extents[entity, 0])
                for j in range(ceil_ndofs[i]):
                    cell_nodes[cell, flat_index[k]] = off + j
                    k += 1

    CHKERR(PetscFree(ceil_ndofs))
    CHKERR(PetscFree(flat_index))
    return cell_nodes

@cython.boundscheck(False)
@cython.wraparound(False)
def reordered_coords(PETSc.DM swarm, PETSc.Section global_numbering, shape):
    """Return coordinates for the swarm, reordered according to the
    global numbering permutation for the coordinate function space.

    Note that this is a copy and paste job from the equivalent function
    in dmplex.pyx with the relevant modifications made for a DMSwarm.

    Shape is a tuple of (mesh.num_vertices(), geometric_dim)."""
    cdef:
        PetscInt v, vStart, vEnd, offset
        PetscInt i, dim = shape[1]
        np.ndarray[PetscReal, ndim=2, mode="c"] swarm_coords, coords

    # get coords field - NOTE it isn't copied so could have GC issues!
    swarm_coords = swarm.getField("DMSwarmPIC_coor").reshape(shape)
    coords = np.empty_like(swarm_coords)
    vStart = 0
    vEnd = swarm.getSize()

    for v in range(vStart, vEnd):
        CHKERR(PetscSectionGetOffset(global_numbering.sec, v, &offset))
        for i in range(dim):
            coords[offset, i] = swarm_coords[v - vStart, i]

    # have to restore coords field once accessed to allow access again
    swarm.restoreField("DMSwarmPIC_coor")

    return coords

@cython.boundscheck(False)
@cython.wraparound(False)
def remove_ghosts_pic(PETSc.DM swarm, PETSc.DM plex):
    """Remove DMSwarm PICs which are in ghost cells of a distributed
    DMPlex.

    :arg swarm: The DMSWARM which has been associated with the input
        DMPlex `plex` using PETSc `DMSwarmSetCellDM`.
    :arg plex: The DMPlex which is associated with the input DMSWARM
        `swarm`
    """
    cdef:
        PetscInt cStart, cEnd, ncells, i, npics
        PETSc.SF sf
        PetscInt nroots, nleaves
        const PetscInt *ilocal = NULL
        const PetscSFNode *iremote = NULL
        np.ndarray[PetscInt, ndim=1, mode="c"] pic_cell_indices
        np.ndarray[PetscInt, ndim=1, mode="c"] ghost_cell_indices

    # check doesn't work right now
    # assert plex is swarm.getCellDM().dm

    if plex.comm.size > 1:

        cStart, cEnd = plex.getHeightStratum(0)
        ncells = cEnd - cStart

        # Get full list of cell indices for particles
        pic_cell_indices = np.copy(swarm.getField("DMSwarm_cellid"))
        swarm.restoreField("DMSwarm_cellid")
        npics = len(pic_cell_indices)

        # Initialise with zeros since these can't be valid ranks or cell ids
        ghost_cell_indices = np.full(ncells, -1, dtype=IntType)

        # Search for ghost cell indices (spooky!)
        sf = plex.getPointSF()
        CHKERR(PetscSFGetGraph(sf.sf, &nroots, &nleaves, &ilocal, &iremote))
        for i in range(nleaves):
            if cStart <= ilocal[i] < cEnd:
                # NOTE need to check this is correct index. Can I check the labels some how?
                ghost_cell_indices[ilocal[i] - cStart] = ilocal[i]

        # trim -1's to reduce searching needed
        ghost_cell_indices = ghost_cell_indices[ghost_cell_indices != -1]

        # remove swarm pic parent cell indices which match ghost cell indices
        for i in range(npics-1, -1, -1):
            if np.isin(pic_cell_indices[i], ghost_cell_indices):
                # removePointAtIndex shift cell numbers down by 1
                swarm.removePointAtIndex(i)

    return