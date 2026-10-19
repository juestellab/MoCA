"""
GPU transport tests against analytic references: normalisation, voxel indexing,
energy conservation, and invariance under the scheduling knobs.
"""

using Test
using Moca
using CUDA
using Statistics

include(joinpath(@__DIR__, "sources.jl"))

# Small CUDA grid keeps n_batch low relative to the ray counts used here.
const NT, NB = 64, 64

"""Beer-Lambert column: absorbed fraction per z-voxel for a pencil beam with mu_s = 0."""
function beer_lambert_run(; block=1, nz=32, dz=0.5f0, mua=0.05f0, N=4_000_000)
    dim = (4, 4, nz)
    gs = Float32.(dim .* dz)
    x0 = 1.5f0 * dz                       # centre of voxel (2,2,*)
    E = quiet_transport() do
        light_transport_gpu(pencil_beam([x0, x0, 0f0], [0f0, 0f0, 1f0]),
                            fill(mua, dim...), zeros(Float32, dim...), zeros(Float32, dim...),
                            gs, ((0f0, 0f0, 0f0), gs), N, 8;
                            n_threads=NT, n_blocks=NB, block)
    end
    z = collect(0:nz) .* dz
    E, exp.(-mua .* z[1:end-1]) .- exp.(-mua .* z[2:end]), N
end

"""Closed medium: mu_eff = 2.45/mm, so the 10 mm half-grid is 24 diffusion
lengths and sum(E) = 1 to ~1e-9. Bounds equal the grid so nothing is lost."""
function closed_medium(; dim=(40, 40, 40), voxel=0.5f0)
    gs = Float32.(dim .* voxel)
    (; μ_a=fill(1f0, dim...), μ_s=fill(10f0, dim...), g=fill(0.9f0, dim...),
       gs, bounds=((0f0, 0f0, 0f0), gs), beam=isotropic_beam(gs ./ 2))
end

@testset "transport" begin

    # With mu_s = 0 voxel k absorbs exactly exp(-mu_a z_{k-1}) - exp(-mu_a z_k).
    # Pins per-ray normalisation, the voxel index map and source placement.
    @testset "Beer-Lambert absorption profile" begin
        E, analytic, N = beer_lambert_run()
        col = vec(sum(E, dims=(1, 2)))
        @test sum(E[2, 2, :]) ≈ sum(col) rtol = 1e-6     # no lateral spread
        for k in 1:length(col)
            @test isapprox(col[k], analytic[k]; rtol=5 / sqrt(N * analytic[k]))
        end
        @test isapprox(sum(col), sum(analytic); rtol=2e-3)
    end

    # Voxel (i,j,k) spans [(i-1)d, i d) etc.; run_mcx.py relies on this.
    @testset "voxel index convention" begin
        dim = (8, 8, 8)
        gs = Float32.(dim)
        E = quiet_transport() do
            light_transport_gpu(pencil_beam([2.5f0, 3.5f0, 0.1f0], [0f0, 0f0, 1f0]),
                                fill(1f2, dim...), zeros(Float32, dim...),
                                zeros(Float32, dim...), gs, ((0f0, 0f0, 0f0), gs),
                                200_000, 4; n_threads=NT, n_blocks=NB)
        end
        @test argmax(E) == CartesianIndex(3, 4, 1)
        @test E[3, 4, 1] > 0.9 * sum(E)
    end

    # A permuted brick layout would scramble the per-voxel profile.
    @testset "blocked layout is transparent" begin
        E, analytic, N = beer_lambert_run(block=4)
        col = vec(sum(E, dims=(1, 2)))
        @test sum(E[2, 2, :]) ≈ sum(col) rtol = 1e-6
        for k in 1:length(col)
            @test isapprox(col[k], analytic[k]; rtol=5 / sqrt(N * analytic[k]))
        end
    end

    @testset "energy conservation in a closed medium" begin
        m = closed_medium()
        E = quiet_transport() do
            light_transport_gpu(m.beam, m.μ_a, m.μ_s, m.g, m.gs, m.bounds, 4_000_000, 400;
                                n_threads=NT, n_blocks=NB)
        end
        @test isapprox(sum(E), 1.0; atol=1e-3)
    end

    # With iterations_bunch below the ~120 collisions a ray needs, rays are in
    # flight when the loop exits; without the drain they inflate E by ~1%.
    @testset "drain removes the in-flight bias" begin
        m = closed_medium()
        total(; kwargs...) = sum(quiet_transport() do
            light_transport_gpu(m.beam, m.μ_a, m.μ_s, m.g, m.gs, m.bounds, 1_000_000, 20;
                                kwargs...)
        end)
        @test isapprox(total(drain = true), 1.0; atol = 1e-3)
        @test total(drain = false) > 1.005
    end

    # Float32 atomics drop deposits below one ulp of the voxel total, losing
    # energy as N grows; the Float64 default must conserve it.
    @testset "Float64 tally conserves energy where Float32 does not" begin
        m = closed_medium(dim=(24, 24, 24), voxel=1f0)
        tally(acc) = quiet_transport() do
            light_transport_gpu(m.beam, m.μ_a, m.μ_s, m.g, m.gs, m.bounds, 20_000_000, 400;
                                n_threads=NT, n_blocks=NB, acc=acc)
        end
        @test isapprox(sum(tally(Float64)), 1.0; atol=0.01)
        @test sum(tally(Float32)) < 0.98
    end

    # Rays that leave simulation_bounds mid-batch must not keep scattering in
    # the padded halo and re-enter. Layered medium, bounds larger than the grid.
    @testset "result does not depend on iterations_bunch" begin
        nx, ny, nz = 40, 20, 40
        gs = (0.4, 0.2, 0.4)
        bounds = ((-0.4, -0.4, -0.4), (0.8, 0.6, 0.8))
        surface = 10

        μ_a = fill(0.01f0, nx, ny, nz); μ_a[surface:end, :, :] .= 0.3f0
        μ_s = fill(5.0f0, nx, ny, nz);  μ_s[surface:end, :, :] .= 250.0f0
        g   = fill(0.5f0, nx, ny, nz);  g[surface:end, :, :]   .= 0.9f0

        beam = pencil_beam([0.2f0, 0.1f0, 0.0f0], [0f0, 0f0, 1f0])
        total(ib) = sum(quiet_transport() do
            light_transport_gpu(beam, μ_a, μ_s, g, gs, bounds, 2_000_000, ib;
                                n_threads=NT, n_blocks=NB)
        end)
        @test total(5) ≈ total(500) rtol = 3e-2
    end
end
