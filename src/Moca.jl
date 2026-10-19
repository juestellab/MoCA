module Moca

export
    BeamSampler,
    light_transport_gpu,
    gauss_profile_sampler,
    sampler_from_weight_array,
    create_xy_beam_sampler,
    initialize_beam_sampler



using Distributed
using StatsBase
using ProgressMeter
using SharedArrays
using CUDA
using StaticArrays
using HDF5


include("utils.jl")        # @per_direction, direction bookkeeping, launch config
include("Layout.jl")       # voxel memory layout, padding, property packing
include("BeamSampling.jl") # light source definition (CPU only)
include("RaySupply.jl")    # staging rays onto the GPU and refilling them
include("Transport.jl")    # the device kernel and the physics it runs
include("MonteCarlo.jl")   # light_transport_gpu: setup, main loop, teardown
include("sugar.jl")        # ready-made beam samplers

end # module
