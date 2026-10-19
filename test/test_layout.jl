"""
CPU tests for `BrickLayout`: the voxel-to-memory map is injective, and at
`B == 1` it is plain column-major indexing.
"""

using Test
using Moca

const LAYOUT_DIMS = ((8, 8, 8),          # exact multiple of every B
                     (10, 7, 13),        # odd in every axis
                     (1, 1, 1),          # degenerate
                     (66, 3, 130))       # very anisotropic, like a padded slab

@testset "voxel layout" begin

    @testset "B = 1 is column-major" begin
        for dims in LAYOUT_DIMS
            L = Moca.brick_layout(1, dims)
            @test length(L) == prod(dims)
            lin = LinearIndices(dims)
            @test all(Moca.voxel_index(L, i, j, l) == lin[i, j, l]
                      for i = 1:dims[1], j = 1:dims[2], l = 1:dims[3])
        end
    end

    @testset "injective and in range" begin
        for dims in LAYOUT_DIMS, B in Moca.SUPPORTED_BLOCKS
            L = Moca.brick_layout(B, dims)
            seen = falses(length(L))
            in_range = true
            collisions = 0
            for l = 1:dims[3], j = 1:dims[2], i = 1:dims[1]
                n = Moca.voxel_index(L, i, j, l)
                in_range &= 1 <= n <= length(L)
                in_range || break
                collisions += seen[n]
                seen[n] = true
            end
            @test in_range
            @test collisions == 0
            @test count(seen) == prod(dims)
            @test length(L) == prod(cld.(dims, B) .* B)
        end
    end
end
