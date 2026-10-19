#!/usr/bin/env julia
"""
Moca side of the cross-validation: writes results/moca_<case>_rep<k>.h5 with
`E` (absorbed fraction per voxel per launched ray) and the label volume.

Run:  julia --project=test/validation test/validation/run_moca.jl --all [--quick]
      julia --project=test/validation test/validation/run_moca.jl --case C3_semiinf
"""

using Moca, CUDA, HDF5, Printf

include(joinpath(@__DIR__, "common.jl"))
using .ValidationCommon

const RESULTS = joinpath(@__DIR__, "results")
const NRAY = 400_000_000
const REPLICATES = 10

function run_case(p; quick=false)
    nray = quick ? 2_000_000 : NRAY
    reps = quick ? 1 : REPLICATES
    mkpath(RESULTS)
    println("\n=== $(p.name): $(p.description)")
    label = build_volume(p)[4]

    for rep in 0:(reps-1)
        r = moca_run(p; nray)
        @printf("    rep %d: %6.1fs  %6.2f Mray/s  sum(E)=%.6f\n",
                rep, r.seconds, r.rays_per_s / 1e6, sum(r.E))
        h5open(joinpath(RESULTS, "moca_$(p.name)_rep$(rep).h5"), "w") do f
            f["E"] = Array{Float64}(r.E)
            f["label"] = label
            HDF5.attributes(f)["nray"] = nray
        end
    end
end

function main()
    quick = "--quick" in ARGS
    wanted = [ARGS[i+1] for i in 1:length(ARGS)-1 if ARGS[i] == "--case"]
    ps = load_phantoms()
    !("--all" in ARGS) && isempty(wanted) && error("pass --all or --case NAME")
    cases = "--all" in ARGS ? ps : [find_case(ps, w) for w in wanted]
    foreach(p -> run_case(p; quick), cases)
end

main()