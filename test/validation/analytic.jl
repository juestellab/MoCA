#!/usr/bin/env julia
"""
Moca against diffusion theory, independent of MCX. Writes results/analytic.md.

  A3  Diffusion limit: isotropic point source in an infinite medium; for
      r >> 1/mu_s' the fluence approaches exp(-mu_eff r) / (4 pi D r).
  A4  Similarity relation: g = 0.9 (mu_s = 10) and g = 0 (mu_s = 1) share
      mu_s' = 1/mm and must give the same diffusive far field.

Run:  julia --project=test/validation test/validation/analytic.jl
"""

using Moca, CUDA, Printf, CairoMakie
using CairoMakie: Figure, Axis, lines!, axislegend, save

include(joinpath(@__DIR__, "..", "sources.jl"))

const RESULTS = joinpath(@__DIR__, "results")
mkpath(RESULTS)

const report = IOBuffer()
say(args...) = (println(args...); println(report, args...))

"""Diffusion-approximation fluence [1/mm^2] at distance r from a unit isotropic source."""
function diffusion_fluence(μ_a, μ_s, g, r)
    D = 1 / (3 * (μ_a + μ_s * (1 - g)))
    μ_eff = sqrt(μ_a / D)
    @. exp(-μ_eff * r) / (4π * D * r)
end

"""Shell-averaged fluence E / (mu_a dV) around the grid centre."""
function radial_fluence(E, μ_a, voxel)
    n = size(E, 1)
    c = n * voxel / 2
    edges = 0:voxel:c
    sums = zeros(length(edges) - 1); cnts = zeros(Int, length(edges) - 1)
    for I in CartesianIndices(E)
        b = searchsortedlast(edges, sqrt(sum(abs2, (Tuple(I) .- 0.5) .* voxel .- c)))
        (b < 1 || b > length(sums)) && continue
        sums[b] += E[I] / (μ_a * voxel^3)
        cnts[b] += 1
    end
    (edges[1:end-1] .+ edges[2:end]) ./ 2, sums ./ max.(cnts, 1)
end

const MUA = 0.02f0
const DIM, VOXEL, PAD, N = (120, 120, 120), 0.5f0, 20f0, 40_000_000
media = (("g=0.9, mu_s=10", 10f0, 0.9f0, 2000),
         ("g=0, mu_s=1",    1f0,  0f0,   400))

curves = map(media) do (nm, mus, gv, bunch)
    gs = Float32.(DIM .* VOXEL)
    E = quiet_transport() do
        light_transport_gpu(isotropic_beam(gs ./ 2), fill(MUA, DIM...), fill(mus, DIM...),
                            fill(gv, DIM...), gs, ((-PAD, -PAD, -PAD), gs .+ PAD), N, bunch)
    end
    (nm, radial_fluence(E, MUA, VOXEL)...)
end

say("# Moca vs diffusion theory\n")
say("Isotropic point source, infinite medium, mu_a = $MUA/mm, mu_s' = 1/mm. ",
    "Fluence is `E / (mu_a dV)`, shell-averaged.\n")
say("| r (mm) | phi g=0.9 | phi g=0 | phi diffusion | ratio g=0.9 | ratio g=0 |")
say("|---:|---:|---:|---:|---:|---:|")
let (_, r, f1) = curves[1], (_, _, f2) = curves[2]
    for i in eachindex(r)
        (r[i] < 2 || r[i] > 25 || i % 8 != 0) && continue
        φd = diffusion_fluence(MUA, 1.0, 0.0, r[i])
        say(@sprintf("| %.2f | %.4e | %.4e | %.4e | %.3f | %.3f |",
                     r[i], f1[i], f2[i], φd, f1[i] / φd, f2[i] / φd))
    end
end

let fig = Figure(size=(900, 380))
    ax = Axis(fig[1, 1], xlabel="r (mm)", ylabel="fluence (1/mm²)", yscale=log10,
              title="isotropic point source, infinite medium")
    ax2 = Axis(fig[1, 2], xlabel="r (mm)", ylabel="Moca / diffusion",
               title="ratio to diffusion (curves must coincide)")
    for (nm, r, φ) in curves
        ok = (φ .> 0) .& (r .> 1)
        lines!(ax, r[ok], φ[ok], label="Moca $nm")
        lines!(ax2, r[ok], φ[ok] ./ diffusion_fluence(MUA, 1.0, 0.0, r[ok]), label=nm)
    end
    r = curves[1][2]
    lines!(ax, r[r .> 1], diffusion_fluence(MUA, 1.0, 0.0, r[r .> 1]),
           linestyle=:dash, color=:black, label="diffusion")
    axislegend(ax, position=:rt)
    axislegend(ax2, position=:rb)
    save(joinpath(RESULTS, "analytic.png"), fig)
end
say("\n![analytic](analytic.png)")

write(joinpath(RESULTS, "analytic.md"), String(take!(report)))