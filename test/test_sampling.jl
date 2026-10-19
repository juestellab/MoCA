"""
CPU tests for `sampler_from_weight_array`, the beam-profile sampler: samples
land in the cell they were drawn for, and cell frequencies match the weights.
"""

using Test
using Moca
using Random

using Moca: sampler_from_weight_array

"""Map coordinates back to the 1-based cell index they must have come from."""
cell_of(p, dims, distances) = ntuple(
    i -> floor(Int, (p[i] + dims[i] * distances[i] / 2) / distances[i]) + 1, length(dims))

@testset "sampler_from_weight_array" begin
    @testset "coordinate map is centred on the array" begin
        # Cell (2, 3) of a 4x5 grid with edge 0.5 spans x in [-0.5, 0) and
        # y in [-0.25, 0.25).
        w = zeros(Float32, 4, 5)
        w[2, 3] = 1
        p = sampler_from_weight_array(w, (0.5f0, 0.5f0))(2000)

        @test size(p) == (2, 2000)
        @test eltype(p) == Float32
        @test all(-0.5f0 .<= p[1, :] .< 0.0f0)
        @test all(-0.25f0 .<= p[2, :] .< 0.25f0)
        @test isapprox(sum(p[1, :]) / 2000, -0.25f0; atol=0.02)
        @test isapprox(sum(p[2, :]) / 2000, 0.0f0; atol=0.02)
    end

    @testset "cell frequencies match the weights" begin
        Random.seed!(7)
        dims, d = (4, 5), (0.5f0, 0.25f0)
        w = rand(Float32, dims...)
        w[1, 1] = 0                            # a zero-weight cell must never be drawn
        n = 400_000
        p = sampler_from_weight_array(w, d)(n)

        counts = zeros(Int, dims...)
        for j in 1:n
            counts[cell_of(view(p, :, j), dims, d)...] += 1
        end
        @test sum(counts) == n
        @test counts[1, 1] == 0
        @test maximum(abs.(counts ./ n .- w ./ sum(w))) < 4 / sqrt(n)
    end
end
