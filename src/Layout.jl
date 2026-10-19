################################################################################
# Voxel memory layout
################################################################################
#
# Every voxel-indexed array -- the optical properties, the tally, the derivative
# tally -- is addressed at a position that is uncorrelated with what the other
# threads in the warp are doing, so nothing here can improve coalescing. What it
# can improve is *temporal* locality: one ray's consecutive steps land in
# neighbouring voxels, and under a plain column-major layout a step along the
# slowest axis moves `nx*ny` elements away and misses cache every time.
#
# Storing voxels in B x B x B bricks makes a brick contiguous, so a +-1 move in
# any axis stays inside the same brick roughly (1 - 1/B)^3 of the time.
#
# `B == 1` reduces to ordinary column-major indexing exactly, so it is the
# current layout with the current arithmetic rather than a separate code path.

"""
    BrickLayout{B}(dims, nb)

Map a voxel grid of extent `dims` onto a flat index, storing voxels in contiguous
`B x B x B` bricks. Build one with [`brick_layout`](@ref) rather than calling
this directly.

`B` is a type parameter so the shifts and masks in [`voxel_index`](@ref) are
compile-time constants, and so the struct is `isbits` and passes to a kernel by
value.

`dims` is rounded up to whole bricks. The overhanging cells are allocated but
never addressed -- `propagate_ray!` clamps every index to `dims` -- so their
contents are irrelevant.

This is a memory layout and nothing more: it knows the extent of the array it
addresses, not what the array means. In particular the guard cells `pad_array`
adds are ordinary voxels as far as it is concerned; stripping them is the
caller's business, handled by passing the wanted extent to
[`unbrick_unpad`](@ref).
"""
struct BrickLayout{B}
    dims::NTuple{3,Int}     # extent being addressed
    nb::NTuple{3,Int}       # brick counts
end

"""
Brick edge lengths `brick_layout` will build. Powers of two only: `voxel_index`
divides by `B` with a shift.
"""
const SUPPORTED_BLOCKS = (1, 2, 4, 8, 16)

@noinline unsupported_block(block) =
    throw(ArgumentError("block must be one of $SUPPORTED_BLOCKS, got $block"))

# The comparison ladder is generated from `SUPPORTED_BLOCKS`, so adding a size to
# the tuple gives it a branch and the two cannot disagree about what is
# supported. A runtime loop would not do: `B` has to reach `BrickLayout{B}` as a
# compile-time constant, which is what makes `voxel_index`'s shifts foldable and
# keeps the struct isbits. Enumerating the sizes also means the return type is a
# small `Union` Julia can split rather than one the caller cannot infer.
@eval begin
    """
        brick_layout(block::Integer, dims) -> BrickLayout

    Layout addressing a `dims` grid in bricks of edge `block` (one of
    `$(SUPPORTED_BLOCKS)`).
    """
    function brick_layout(block::Integer, dims::NTuple{3,Integer})
        $([:(block == $B && return _brick_layout(Val($B), dims))
           for B in SUPPORTED_BLOCKS]...)
        unsupported_block(block)
    end
end

function _brick_layout(::Val{B}, dims::NTuple{3,Integer}) where B
    d = Int.(dims)
    return BrickLayout{B}(d, cld.(d, B))
end

"""
    blocksize(L::BrickLayout)
The brick edge length `B`, recovered from the type.
"""
blocksize(::BrickLayout{B}) where B = B

"""
    length(L::BrickLayout)
Number of slots a voxel array needs, including the cells that overhang the grid.
"""
Base.length(L::BrickLayout{B}) where B = prod(L.nb) * B^3

"""
    overhang(L::BrickLayout)
Ratio of allocated slots to addressable voxels -- the memory the brick rounding
costs. `1.0` when the grid divides evenly into bricks, and always `1.0` at
`B == 1`.
"""
overhang(L::BrickLayout) = length(L) / prod(L.dims)

"""
    voxel_index(L::BrickLayout, i, j, l)

Flat 1-based index of voxel `(i, j, l)`, in padded coordinates.

For `B == 1` every offset is zero, `nb == dims` and this is exactly
`i + nx*((j-1) + ny*(l-1))` -- the column-major index. `test_layout.jl` pins that
against `LinearIndices`.
"""
@inline function voxel_index(L::BrickLayout{B}, i, j, l) where B
    s = trailing_zeros(B)
    bi, oi = (i - 1) >> s, (i - 1) & (B - 1)
    bj, oj = (j - 1) >> s, (j - 1) & (B - 1)
    bl, ol = (l - 1) >> s, (l - 1) & (B - 1)
    brick = bi + L.nb[1] * (bj + L.nb[2] * bl)
    return brick * B^3 + (oi + B * (oj + B * ol)) + 1
end

"""
    brick_permute(A, L::BrickLayout)

Reorder a padded host volume into `L`'s flat layout.

A 3D `A` becomes a `Vector` of `length(L)`; a 4D `A` keeps its leading axis and
becomes a `(size(A, 1), length(L))` `Matrix`. That leading axis is the derivative
direction `k` for `dμ_a`/`dμ_s`/`dE`, and the four interleaved quantities for the
property array -- in both cases it stays the fastest-varying axis, as it is
without bricking.

At `B == 1` the permutation is the identity, so this is a reshape and no element
is touched. That matters because `block = 1` is the default: without the short
cut every run would pay a single-threaded pass over `K` times the grid -- about
a billion scalar writes at `K = 8` on the production geometry -- to reproduce
the array it already had.

Slots belonging to bricks that overhang the grid are left zero; they are never
read.
"""
function brick_permute(A::AbstractArray{T,4}, L::BrickLayout{B}) where {T,B}
    size(A)[2:4] == L.dims ||
        throw(DimensionMismatch("array is $(size(A)[2:4]), layout expects $(L.dims)"))
    K = size(A, 1)
    B == 1 && return reshape(A, K, :)
    out = zeros(T, K, length(L))
    @inbounds for l = 1:L.dims[3], j = 1:L.dims[2], i = 1:L.dims[1]
        n = voxel_index(L, i, j, l)
        for k = 1:K
            out[k, n] = A[k, i, j, l]
        end
    end
    return out
end

# The 3D case is the K = 1 case with the leading axis squeezed away; `reshape`
# and `vec` are views, so this adds no copy.
brick_permute(A::AbstractArray{<:Any,3}, L::BrickLayout) =
    vec(brick_permute(reshape(A, 1, size(A)...), L))

"""
    unbrick_kernel!(dst, src, L, guard)

Undo [`brick_permute`](@ref) and drop the outer `guard` cells in one pass: one
thread per voxel of `dst`, gathering from `src`.

Fusing the two avoids materialising the padded volume, which for `K = 8` on a
production grid is close to a gigabyte that would only be sliced away again.
"""
function unbrick_kernel!(dst::CuDeviceArray{<:Any,3}, src, L, guard)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    nx, ny, nz = size(dst)
    idx > nx * ny * nz && return nothing
    i, j, l = source_voxel(idx, nx, ny, guard)
    @inbounds dst[idx] = src[voxel_index(L, i, j, l)]
    return nothing
end

function unbrick_kernel!(dst::CuDeviceArray{<:Any,4}, src, L, guard)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    K, nx, ny, nz = size(dst)
    idx > nx * ny * nz && return nothing
    i, j, l = source_voxel(idx, nx, ny, guard)
    n = voxel_index(L, i, j, l)
    @inbounds for k = 1:K
        dst[k, i - guard[1], j - guard[2], l - guard[3]] = src[k, n]
    end
    return nothing
end

"""
    source_voxel(idx, nx, ny, guard)

Turn a 1-based linear index over the *output* grid into the source coordinates
`voxel_index` expects. The `+ 1` converts the 0-based remainder back to a 1-based
index; `guard` steps over the cells being dropped.
"""
@inline function source_voxel(idx, nx, ny, guard)
    q = idx - 1
    i = q % nx; q = q ÷ nx
    j = q % ny; q = q ÷ ny
    return (i + 1 + guard[1], j + 1 + guard[2], q + 1 + guard[3])
end

"""
    unbrick_unpad(src, L::BrickLayout, inner) -> Array

Bring a flat device tally back to the host as an ordinary array of extent
`inner`, dropping the cells the layout addresses but the caller did not ask for.

`src` is either a vector (`length(L)`) or a `(K, length(L))` matrix. The guard
width is the difference between what `L` addresses and what is wanted back, so
nothing here assumes how thick the padding is -- and a caller that wants the
padded volume passes `L.dims`.
"""
unbrick_unpad(src::AbstractVector, L::BrickLayout, inner::NTuple{3,Integer}) =
    gather_interior(src, L, inner, inner)
unbrick_unpad(src::AbstractMatrix, L::BrickLayout, inner::NTuple{3,Integer}) =
    gather_interior(src, L, (size(src, 1), inner...), inner)

function gather_interior(src, L::BrickLayout, shape, inner)
    guard = (L.dims .- inner) .÷ 2
    all(guard .>= 0) ||
        throw(DimensionMismatch("layout addresses $(L.dims), cannot return $inner"))
    dst = CuArray{eltype(src)}(undef, shape)
    threads = 256
    @cuda threads = threads blocks = cld(prod(inner), threads) unbrick_kernel!(dst, src, L, guard)
    return Array(dst)
end

################################################################################
# Host-side padding and property packing
################################################################################
#
# What the voxel arrays look like before `brick_permute` reorders them: a
# replicated guard cell on every face, and the three optical properties
# interleaved into one 16-byte-aligned record per voxel.

"""
    pad_array(A::Array)
Add a single pixel at each boundary of `A`. Those added pixels are filled with the value of their interior neighbour.
"""
function pad_array(A::Array)
    A_padded = Array{eltype(A)}(undef, (size(A) .+ 2)...)

    # Inner Volume
    A_padded[2:(end-1), 2:(end-1), 2:(end-1)] .= A

    # Faces
    A_padded[1, 2:(end-1), 2:(end-1)] .= A[1, :, :]
    A_padded[2:(end-1), 1, 2:(end-1)] .= A[:, 1, :]
    A_padded[2:(end-1), 2:(end-1), 1] .= A[:, :, 1]
    A_padded[end, 2:(end-1), 2:(end-1)] .= A[end, :, :]
    A_padded[2:(end-1), end, 2:(end-1)] .= A[:, end, :]
    A_padded[2:(end-1), 2:(end-1), end] .= A[:, :, end]

    # Edges
    A_padded[1, 1, 2:(end-1)] .= A[1,1,:]
    A_padded[1, 2:(end-1), 1] .= A[1,:,1]
    A_padded[2:(end-1), 1, 1] .= A[:,1,1]
    A_padded[1, end, 2:(end-1)] .= A[1,end,:]
    A_padded[1, 2:(end-1), end] .= A[1,:,end]
    A_padded[end, 1, 2:(end-1)] .= A[end,1,:]
    A_padded[end, 2:(end-1), 1] .= A[end,:,1]
    A_padded[2:(end-1), 1, end] .= A[:,1,end]
    A_padded[2:(end-1), end, 1] .= A[:,end,1]
    A_padded[end, end, 2:(end-1)] .= A[end,end,:]
    A_padded[end, 2:(end-1), end] .= A[end,:,end]
    A_padded[2:(end-1), end, end] .= A[:,end,end]

    # Corners
    A_padded[1,1,1] = A[1,1,1]
    A_padded[1,1,end] = A[1,1,end]
    A_padded[1,end,1] = A[1,end,1]
    A_padded[end,1,1] = A[end,1,1]
    A_padded[1,end,end] = A[1,end,end]
    A_padded[end,1,end] = A[end,1,end]
    A_padded[end,end,1] = A[end,end,1]
    A_padded[end,end,end] = A[end,end,end]

    return A_padded
end

"""
    pad_stack(A::AbstractArray{<:Any,4})
Apply [`pad_array`](@ref) to each `A[k, :, :, :]` slice and stack the results back
along the leading derivative axis. Used for the multi-direction sensitivity inputs.
"""
function pad_stack(A::AbstractArray{<:Any,4})
    K = size(A, 1)
    first_slice = pad_array(Array(A[1, :, :, :]))
    out = Array{eltype(first_slice)}(undef, K, size(first_slice)...)
    out[1, :, :, :] .= first_slice
    for k = 2:K
        out[k, :, :, :] .= pad_array(Array(A[k, :, :, :]))
    end
    return out
end

"""
    pack_properties(T, μ_a, μ_s, g)

Pad each property volume with a replicated boundary cell and interleave them into
a single `(4, nx+2, ny+2, nz+2)` host array laid out as `[μ_a, μ_s, g, μ_a + μ_s]`
per voxel.

`propagate_ray!` reads all three properties at the same voxel, and because each
thread's ray sits at an unrelated position those are three independent
random-access loads -- three cache lines, three memory transactions -- per step.
Interleaved, they occupy 16 contiguous bytes and cost one transaction. The fourth
slot holds the precomputed extinction `μ_a + μ_s`: it is free (same cache line),
and padding to 4 floats is what makes the load 16-byte aligned in the first place.
"""
function pack_properties(::Type{T}, μ_a, μ_s, g) where T
    a, s, gg = map(x -> pad_array(Array(x)), (μ_a, μ_s, g))
    out = Array{T}(undef, 4, size(a)...)
    @views begin
        out[1, :, :, :] .= a
        out[2, :, :, :] .= s
        out[3, :, :, :] .= gg
        out[4, :, :, :] .= a .+ s
    end
    return out
end
