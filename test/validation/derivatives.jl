#!/usr/bin/env julia
"""
Moca's directional derivative `dE` against central finite differences, in a
closed heterogeneous medium. Writes results/derivatives.md.

MC finite differences have noise ~ sigma/eps, so the usual cbrt(eps) step is far
too small; the best eps is percent-scale (truncation is only O(eps^2)). We sweep
eps and compare on aggregate functionals, since per-voxel FD is noise-dominated.

Run:  julia --project=test/validation test/validation/derivatives.jl [--quick]
"""

using Moca, CUDA, Statistics, Printf, CairoMakie
using CairoMakie: Figure, Axis, lines!, scatter!, axislegend, save

include(joinpath(@__DIR__, "..", "sources.jl"))

const RESULTS = joinpath(@__DIR__, "results")
mkpath(RESULTS)
const QUICK = "--quick" in ARGS
const REPS = QUICK ? 2 : 4
const NRAY = QUICK ? 5_000_000 : 100_000_000

const report = IOBuffer()
say(args...) = (println(args...); println(report, args...))

ramp(dim, lo, hi, axis) = begin
    v = Float32.(range(lo, hi, length=dim[axis]))
    shape = ntuple(i -> i == axis ? dim[axis] : 1, 3)
    Float32.(repeat(reshape(v, shape), ntuple(i -> i == axis ? 1 : dim[i], 3)...))
end

# Closed medium (>25 diffusion lengths to the grid face). mu_a ramps along x,
# mu_s along z; the perturbation directions ramp along the other axes.
const DIM, VOXEL = (64, 64, 64), 0.5f0
const GS = Float32.(DIM .* VOXEL)
const F = (μ_a = ramp(DIM, 0.3, 1.0, 1), μ_s = ramp(DIM, 12.0, 5.0, 3),
           dμ_a = ramp(DIM, 0.3, 1.0, 3), dμ_s = ramp(DIM, 12.0, 5.0, 1),
           g = fill(0.5f0, DIM...))

transport(args...) = quiet_transport() do
    light_transport_gpu(isotropic_beam(GS ./ 2), args..., GS, ((0f0, 0f0, 0f0), GS),
                        NRAY, 400; n_threads=256, n_blocks=256)
end

reductions(A) = (
    total = [sum(A)],
    depth = vec(sum(A, dims=(1, 2))),
    xprof = vec(sum(A, dims=(2, 3))),
    octant = [sum(@view A[ix, iy, iz]) for ix in (1:32, 33:64), iy in (1:32, 33:64),
                                           iz in (1:32, 33:64)] |> vec,
)

"""mean and standard error over replicate runs of a reduction"""
function replicate_stats(f, reps)
    outs = [f() for _ in 1:reps]
    NamedTuple{keys(outs[1])}(map(keys(outs[1])) do k
        v = [o[k] for o in outs]
        μ = reduce(+, v) ./ reps
        (mean=μ, se=sqrt.(reduce(+, (abs2.(x .- μ) for x in v)) ./ (reps * (reps - 1))))
    end)
end

base = replicate_stats(REPS) do
    reductions(transport(F.μ_a, F.μ_s, F.dμ_a, F.dμ_s, F.g)[2])
end

epsilons = [0.003, 0.01, 0.03, 0.1, 0.2]
fd_rows = map(epsilons) do ε
    e = Float32(ε)
    fd = replicate_stats(REPS) do
        Ep = transport(F.μ_a .+ e .* F.dμ_a, F.μ_s .+ e .* F.dμ_s, F.g)
        Em = transport(F.μ_a .- e .* F.dμ_a, F.μ_s .- e .* F.dμ_s, F.g)
        reductions((Ep .- Em) ./ (2ε))
    end
    (; ε, fd)
end

# rms of per-element z is 1 under the null; the max is an extreme-value statistic.
say("# Moca derivative vs finite differences\n")
say("Relative step eps; z = (FD - dE) / combined replicate s.e. ",
    "rms z = 1 means agreement within MC noise.\n")
say("| functional | eps | max\\|FD - dE\\| | rms z |")
say("|---|---:|---:|---:|")
best = Dict{Symbol,Float64}()
for k in (:total, :depth, :xprof, :octant), row in fd_rows
    f = row.fd[k]; b = base[k]
    zs = abs.(f.mean .- b.mean) ./ max.(sqrt.(f.se .^ 2 .+ b.se .^ 2), eps())
    rmsz = sqrt(mean(abs2, zs))
    say(@sprintf("| %s | %.3f | %.3e | %.2f |", k, row.ε, maximum(abs.(f.mean .- b.mean)), rmsz))
    best[k] = min(get(best, k, Inf), rmsz)
end
say("\nBest rms z over the sweep: ",
    join([@sprintf("%s %.2f", k, best[k]) for k in (:total, :depth, :xprof, :octant)], ", "))

let fig = Figure(size=(900, 380))
    for (n, k) in enumerate((:depth, :octant))
        ax = Axis(fig[1, n], xscale=log10, yscale=log10, xlabel="relative eps",
                  ylabel="max |FD - dE|", title="$(k) functional")
        ys = [maximum(abs.(row.fd[k].mean .- base[k].mean)) for row in fd_rows]
        ns = [sqrt(maximum(row.fd[k].se)^2 + maximum(base[k].se)^2) for row in fd_rows]
        lines!(ax, epsilons, max.(ys, 1e-30), label="|FD - dE|")
        scatter!(ax, epsilons, max.(ys, 1e-30))
        lines!(ax, epsilons, max.(ns, 1e-30), linestyle=:dash, label="MC noise floor")
        axislegend(ax, position=:lt)
    end
    save(joinpath(RESULTS, "derivative_eps_sweep.png"), fig)
end
say("\n![eps sweep](derivative_eps_sweep.png)")

write(joinpath(RESULTS, "derivatives.md"), String(take!(report)))