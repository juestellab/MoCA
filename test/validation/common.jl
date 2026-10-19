"""
Shared helpers for the Moca <-> MCX cross-validation: phantom loading and
volume construction (mirrors `run_mcx.py::build_volume`), a Moca runner, and
the z-score statistics used by compare.jl.
"""

module ValidationCommon

using Moca
using CUDA
using JSON3
using Statistics

export pencil_beam, isotropic_beam, quiet_transport,
       load_phantoms, find_case, build_volume, compare_mask, moca_run,
       zscore_report, decade_table

include(joinpath(@__DIR__, "..", "sources.jl"))

load_phantoms(path=joinpath(@__DIR__, "phantoms.json")) = JSON3.read(read(path, String)).cases

find_case(ps, name) = ps[findfirst(p -> p.name == name, ps)]

dims(p) = Tuple(Int.(p.dim))
grid_size(p) = Tuple(Float32.(dims(p) .* p.voxel_mm))

"""
    build_volume(p) -> (μ_a, μ_s, g, label)

Lay down the z-layers (0-based half-open ranges), then zero an exterior shell of
`shell_voxels` on all faces. `label` matches the integer volume MCX is given.
"""
function build_volume(p)
    d = dims(p)
    μ_a = zeros(Float32, d...)
    μ_s = zeros(Float32, d...)
    g = zeros(Float32, d...)
    label = zeros(UInt8, d...)
    for (i, l) in enumerate(p.layers)
        zr = (l.z_from + 1):l.z_to
        μ_a[:, :, zr] .= Float32(l.mua)
        μ_s[:, :, zr] .= Float32(l.mus)
        g[:, :, zr] .= Float32(l.g)
        label[:, :, zr] .= UInt8(i)
    end
    s = p.shell_voxels
    if s > 0
        for (A, z) in ((μ_a, 0f0), (μ_s, 0f0), (g, 0f0), (label, UInt8(0)))
            A[1:s, :, :] .= z;  A[end-s+1:end, :, :] .= z
            A[:, 1:s, :] .= z;  A[:, end-s+1:end, :] .= z
            A[:, :, 1:s] .= z;  A[:, :, end-s+1:end] .= z
        end
    end
    return μ_a, μ_s, g, label
end

"""
    compare_mask(p, label) -> BitArray

"label": every medium voxel. "radius": a ball around the source, which keeps
C1's comparison away from the grid faces where the two codes kill photons.
"""
function compare_mask(p, label)
    mode = p.compare.mode
    mode == "label" && return label .!= 0
    mode == "radius" || error("unknown compare mode $mode")
    c = Float64.(p.compare.center_mm)
    rmax = Float64(p.compare.max_mm)
    v = p.voxel_mm
    m = falses(dims(p)...)
    for I in CartesianIndices(m)
        m[I] = sum(abs2, (Tuple(I) .- 0.5) .* v .- c) <= rmax^2
    end
    return m .& (label .!= 0)
end

"""`simulation_bounds` extended `pad_mm` past the grid on every side."""
function sim_bounds(p)
    gs = grid_size(p)
    pad = Float32(p.pad_mm)
    ((-pad, -pad, -pad), gs .+ pad)
end

function build_beam(p)
    s = p.source
    pos = Float32.(s.pos_mm)
    s.type == "pencil" && return pencil_beam(pos, Float32.(s.dir))
    s.type == "isotropic" && return isotropic_beam(pos)
    error("unknown source type $(s.type)")
end

"""Run Moca on phantom `p`, silenced and timed."""
function moca_run(p; nray, n_threads=256, n_blocks=256)
    μ_a, μ_s, g = build_volume(p)
    t0 = time()
    E = quiet_transport() do
        light_transport_gpu(build_beam(p), μ_a, μ_s, g, grid_size(p), sim_bounds(p),
                            nray, p.bunch; n_threads, n_blocks)
    end
    dt = time() - t0
    return (; E, seconds=dt, rays_per_s=nray / dt)
end

"""
    zscore_report(a, sa, b, sb; mask) -> NamedTuple

Compare two MC fields with replicate standard errors. A nonzero mean z is what
a bias looks like; voxels with zero variance on either side are dropped.
"""
function zscore_report(a, sa, b, sb; mask=trues(size(a)))
    mask = mask .& isfinite.(sa) .& isfinite.(sb) .& (sa .> 0) .& (sb .> 0)
    idx = findall(mask)
    isempty(idx) && return (; n=0, mean_z=NaN, std_z=NaN, frac_out=NaN,
                            rel_bias=NaN, med_rel=NaN)
    z = [(a[i] - b[i]) / sqrt(sa[i]^2 + sb[i]^2) for i in idx]
    rel = [(a[i] - b[i]) / b[i] for i in idx if b[i] != 0]
    w = [b[i] for i in idx if b[i] != 0]
    (; n=length(idx),
       mean_z=mean(z), std_z=std(z),
       frac_out=count(x -> abs(x) > 3, z) / length(z),
       rel_bias=isempty(rel) ? NaN : sum(rel .* w) / sum(w),
       med_rel=isempty(rel) ? NaN : median(rel))
end

"""
    decade_table(a, sa, b, sb; mask, ref, max_relerr, min_count)

`zscore_report` per decade of `ref` below its peak inside `mask`, so the bright
voxels near the source cannot carry the verdict. Voxels with relative MC error
above `max_relerr` on either side are excluded.
"""
function decade_table(a, sa, b, sb; mask=trues(size(a)), ref=b,
                      max_relerr=0.05, min_count=1000)
    peak = maximum(ref[mask])
    rows = NamedTuple[]
    for d in 0:7
        lo, hi = peak * 10.0^(-(d + 1)), peak * 10.0^(-d)
        m = mask .& (ref .> lo) .& (ref .<= hi) .&
            (sa .<= max_relerr .* max.(a, eps())) .&
            (sb .<= max_relerr .* max.(b, eps()))
        count(m) < min_count && continue
        r = zscore_report(a, sa, b, sb; mask=m)
        push!(rows, (; decade="1e-$(d)..1e-$(d+1)", r...))
    end
    rows
end

end # module