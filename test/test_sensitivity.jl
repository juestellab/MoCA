"""
GPU tests for the directional derivative `dE` from the five-argument
`light_transport_gpu`, using identities and closed forms rather than finite
differences (those live in validation/derivatives.jl).
"""

using Test
using Moca
using CUDA
using Statistics

include(joinpath(@__DIR__, "sources.jl"))

const NT, NB = 64, 64

"""Closed medium (24 diffusion lengths to the grid face), so sum(E) = 1 for any
optical properties and sum(dE) = 0 for any direction."""
function closed_run(dμ_a, dμ_s; N=4_000_000, dim=(40, 40, 40), voxel=0.5f0, kwargs...)
    gs = Float32.(dim .* voxel)
    quiet_transport() do
        light_transport_gpu(isotropic_beam(gs ./ 2), fill(1f0, dim...), fill(10f0, dim...),
                            dμ_a, dμ_s, fill(0.9f0, dim...), gs, ((0f0, 0f0, 0f0), gs),
                            N, 400; n_threads=NT, n_blocks=NB, kwargs...)
    end
end

@testset "sensitivity" begin

    # K directions on one path ensemble share every random number, so
    # linearity holds to float rounding, not MC noise. Direction 3 = 1 + 2,
    # direction 4 = 0. |sum(dE)| < 5e-3 of the launched energy is ~5 sigma.
    @testset "conservation and linearity over K directions" begin
        dim = (40, 40, 40)
        dμ_a = zeros(Float32, 4, dim...)
        dμ_s = zeros(Float32, 4, dim...)
        dμ_a[1, :, :, :] .= 0.5f0
        dμ_s[2, :, :, :] .= 2f0
        dμ_a[3, :, :, :] .= 0.5f0;  dμ_s[3, :, :, :] .= 2f0

        E, dE = closed_run(dμ_a, dμ_s)

        @test size(dE) == (4, dim...)
        @test isapprox(sum(E), 1.0; atol=1e-3)
        for k in 1:3
            @test abs(sum(dE[k, :, :, :])) < 5e-3
        end
        rhs = dE[1, :, :, :] .+ dE[2, :, :, :]
        @test maximum(abs.(dE[3, :, :, :] .- rhs)) < 1e-5 * maximum(abs.(rhs))
        @test all(iszero, dE[4, :, :, :])
    end

    # dE and dμ use the same brick layout as the tally.
    @testset "blocked layout" begin
        dim = (40, 40, 40)
        E, dE = closed_run(fill(0.5f0, dim...), fill(2f0, dim...); block=4)
        @test size(dE) == dim
        @test isapprox(sum(E), 1.0; atol=1e-3)
        @test abs(sum(dE)) < 5e-3
    end

    # With mu_s = 0: dE_k/dmu_a = z_k exp(-mu_a z_k) - z_{k-1} exp(-mu_a z_{k-1}).
    @testset "analytic Beer-Lambert derivative" begin
        nz, dz, mua = 32, 0.5f0, 0.05f0
        dim = (4, 4, nz)
        gs = Float32.(dim .* dz)
        x0 = 1.5f0 * dz
        E, dE = quiet_transport() do
            light_transport_gpu(pencil_beam([x0, x0, 0f0], [0f0, 0f0, 1f0]),
                                fill(mua, dim...), zeros(Float32, dim...),
                                ones(Float32, dim...), zeros(Float32, dim...),
                                zeros(Float32, dim...), gs, ((0f0, 0f0, 0f0), gs),
                                8_000_000, 8; n_threads=NT, n_blocks=NB)
        end

        colE = vec(sum(E, dims=(1, 2)))
        coldE = vec(sum(dE, dims=(1, 2)))
        z = collect(0:nz) .* dz
        aE = exp.(-mua .* z[1:end-1]) .- exp.(-mua .* z[2:end])
        adE = z[2:end] .* exp.(-mua .* z[2:end]) .- z[1:end-1] .* exp.(-mua .* z[1:end-1])

        @test maximum(abs.(colE .- aE)) / maximum(aE) < 0.05
        @test maximum(abs.(coldE .- adE)) / maximum(abs.(adE)) < 0.10
        # The derivative changes sign with depth.
        @test sign(coldE[1]) == sign(adE[1])
        @test sign(coldE[end]) == sign(adE[end])
    end
end
