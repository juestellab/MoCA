#!/usr/bin/env julia
"""
Compare Moca and MCX absorbed-energy fields and write results/report.md.

Per voxel, z = (E_moca - E_mcx) / sqrt(se_moca^2 + se_mcx^2) from replicate
standard errors. The verdict uses the mean z (a bias shows up as an offset) and
the weighted relative bias, stratified by decade below the peak.

Run:  julia --project=test/validation test/validation/compare.jl [--case NAME]
"""

using HDF5, Statistics, Printf, CairoMakie
using CairoMakie: Figure, Axis, lines!, heatmap!, hist!, axislegend, Colorbar, save

include(joinpath(@__DIR__, "common.jl"))
using .ValidationCommon

const RESULTS = joinpath(@__DIR__, "results")

"""Mean and standard error of the mean across replicates."""
function load_replicates(prefix, case, dataset; permute=false)
    files = sort(filter(f -> startswith(f, "$(prefix)_$(case)_rep") && endswith(f, ".h5"),
                        readdir(RESULTS)))
    isempty(files) && error("no $prefix replicates for $case in $RESULTS")
    arrs = map(files) do f
        a = h5open(h -> read(h[dataset]), joinpath(RESULTS, f), "r")
        permute ? permutedims(a, (3, 2, 1)) : a
    end
    n = length(arrs)
    μ = reduce(+, arrs) ./ n
    n == 1 && return μ, fill(NaN, size(μ)), 1
    var = reduce(+, (abs2.(a .- μ) for a in arrs)) ./ (n - 1)
    return μ, sqrt.(var ./ n), n
end

"""MCX arrays come back with reversed axes; the label volumes of both codes must
match exactly after permuting, which catches any ordering or geometry mismatch."""
function load_case(p)
    Em, sem, nm = load_replicates("moca", p.name, "E")
    Ex, sex, nx = load_replicates("mcx", p.name, "absorbed"; permute=true)

    lab_m = h5open(h -> read(h["label"]), joinpath(RESULTS, "moca_$(p.name)_rep0.h5"), "r")
    lab_x = permutedims(h5open(h -> read(h["label"]),
                               joinpath(RESULTS, "mcx_$(p.name)_rep0.h5"), "r"), (3, 2, 1))
    lab_m == lab_x || error("$(p.name): label volumes differ -- grids are not aligned")

    (; Em, sem, nm, Ex, sex, nx, label=lab_m)
end

function make_figures(p, d, mask)
    v = p.voxel_mm
    dim = Tuple(Int.(p.dim))
    src = Float64.(p.source.pos_mm)
    fig = Figure(size=(1200, 900))

    # depth profile through the source column
    ix = clamp(round(Int, src[1] / v) + 1, 1, dim[1])
    iy = clamp(round(Int, src[2] / v) + 1, 1, dim[2])
    zc = ((1:dim[3]) .- 0.5) .* v
    ax1 = Axis(fig[1, 1], xlabel="z (mm)", ylabel="absorbed fraction / voxel",
               yscale=log10, title="on-axis depth profile")
    lines!(ax1, zc, max.(d.Em[ix, iy, :], 1e-30), label="Moca")
    lines!(ax1, zc, max.(d.Ex[ix, iy, :], 1e-30), label="MCX", linestyle=:dash)
    axislegend(ax1, position=:rt)

    # radial profile at the source plane
    ax2 = Axis(fig[1, 2], xlabel="radius (mm)", ylabel="absorbed fraction / voxel",
               yscale=log10, title="radial profile")
    kz = clamp(round(Int, src[3] / v) + 1, 1, dim[3])
    rr = Float64[]; vm = Float64[]; vx = Float64[]
    for j in 1:dim[2], i in 1:dim[1]
        push!(rr, hypot((i - 0.5) * v - src[1], (j - 0.5) * v - src[2]))
        push!(vm, d.Em[i, j, kz]); push!(vx, d.Ex[i, j, kz])
    end
    nb = 40
    edges = range(0, maximum(rr), length=nb + 1)
    bm = [mean(vm[(rr .>= edges[k]) .& (rr .< edges[k+1])]) for k in 1:nb]
    bx = [mean(vx[(rr .>= edges[k]) .& (rr .< edges[k+1])]) for k in 1:nb]
    ctr = (edges[1:end-1] .+ edges[2:end]) ./ 2
    ok = isfinite.(bm) .& isfinite.(bx) .& (bm .> 0) .& (bx .> 0)
    lines!(ax2, ctr[ok], bm[ok], label="Moca")
    lines!(ax2, ctr[ok], bx[ok], label="MCX", linestyle=:dash)
    axislegend(ax2, position=:rt)

    # log10 slices side by side
    sl_m = log10.(max.(d.Em[:, iy, :], 1e-30))
    sl_x = log10.(max.(d.Ex[:, iy, :], 1e-30))
    cr = (maximum(sl_x) - 6, maximum(sl_x))
    xc = ((1:dim[1]) .- 0.5) .* v
    heatmap!(Axis(fig[2, 1], title="Moca  log10 absorbed", xlabel="x (mm)", ylabel="z (mm)"),
             xc, zc, sl_m, colorrange=cr)
    hm = heatmap!(Axis(fig[2, 2], title="MCX  log10 absorbed", xlabel="x (mm)", ylabel="z (mm)"),
                  xc, zc, sl_x, colorrange=cr)
    Colorbar(fig[2, 3], hm)

    # z-score histogram vs unit normal
    idx = findall(mask .& isfinite.(d.sem) .& isfinite.(d.sex) .& (d.sem .> 0) .& (d.sex .> 0))
    z = filter(isfinite, [(d.Em[i] - d.Ex[i]) / sqrt(d.sem[i]^2 + d.sex[i]^2) for i in idx])
    ax5 = Axis(fig[3, 1:2], xlabel="z", ylabel="density",
               title="z-score distribution (dashed = unit normal)")
    if !isempty(z)
        hist!(ax5, clamp.(z, -6, 6), bins=80, normalization=:pdf)
        xs = range(-6, 6, length=200)
        lines!(ax5, xs, exp.(-xs .^ 2 ./ 2) ./ sqrt(2π), linestyle=:dash, color=:black)
    end

    path = joinpath(RESULTS, "compare_$(p.name).png")
    save(path, fig)
    return path
end

function report_case(io, p)
    d = load_case(p)
    mask = compare_mask(p, d.label)
    ov = zscore_report(d.Em, d.sem, d.Ex, d.sex; mask=mask)

    println(io, "\n## $(p.name)\n\n$(p.description)\n")
    @printf(io, "- %d compared voxels, %d/%d replicates (Moca/MCX)\n", count(mask), d.nm, d.nx)
    @printf(io, "- total absorbed: Moca %.6f, MCX %.6f (ratio %.5f)\n",
            sum(d.Em), sum(d.Ex), sum(d.Em) / sum(d.Ex))
    @printf(io, "- overall: mean z = %+.3f, weighted rel. bias = %+.3f%%\n",
            ov.mean_z, 100 * ov.rel_bias)

    println(io, "\n| decade below peak | voxels | mean z | rel. bias |")
    println(io, "|---|---:|---:|---:|")
    for r in decade_table(d.Em, d.sem, d.Ex, d.sex; mask=mask, ref=d.Ex)
        @printf(io, "| %s | %d | %+.3f | %+.3f%% |\n", r.decade, r.n, r.mean_z, 100 * r.rel_bias)
    end
    println(io, "\n![comparison]($(basename(make_figures(p, d, mask))))")

    verdict = if d.nm < 2 || d.nx < 2 || isnan(ov.mean_z)
        "INCONCLUSIVE (needs >= 2 replicates per code)"
    elseif abs(ov.mean_z) < 0.5 && abs(ov.rel_bias) < 0.01
        "PASS"
    else
        "FAIL"
    end
    return (; case=p.name, ov..., verdict)
end

function main()
    ps = load_phantoms()
    wanted = [ARGS[i+1] for i in 1:length(ARGS)-1 if ARGS[i] == "--case"]
    cases = isempty(wanted) ? collect(ps) : [find_case(ps, w) for w in wanted]
    have(p) = isfile(joinpath(RESULTS, "moca_$(p.name)_rep0.h5")) &&
              isfile(joinpath(RESULTS, "mcx_$(p.name)_rep0.h5"))
    cases = filter(have, cases)
    isempty(cases) && error("no case has both Moca and MCX results in $RESULTS")

    io = IOBuffer()
    println(io, "# Moca vs MCX\n")
    println(io, "Absorbed fraction per voxel per launched photon. ",
                "PASS: |mean z| < 0.5 and |rel. bias| < 1%.")
    summary = [report_case(io, p) for p in cases]

    println(io, "\n## Summary\n\n| case | mean z | rel. bias | verdict |\n|---|---:|---:|---|")
    for s in summary
        @printf(io, "| %s | %+.3f | %+.3f%% | %s |\n", s.case, s.mean_z, 100 * s.rel_bias, s.verdict)
    end

    write(joinpath(RESULTS, "report.md"), String(take!(io)))
    for s in summary
        @printf("  %-20s mean z=%+.3f  rel.bias=%+.3f%%  %s\n",
                s.case, s.mean_z, 100 * s.rel_bias, s.verdict)
    end
end

main()