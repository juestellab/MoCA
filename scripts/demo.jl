using Moca
using CUDA
include("visualization.jl")

n_rays = 100_000_000
grid_size = (0.04, 0.04, 0.005)
grid_points = (1600, 1600, 1)

sim_bounds = ((-0.02, -0.02, -0.0025), (0.06, 0.06, 0.0075))

ua = 22.3 * ones(Float32, grid_points...) .+ 1.5 * randn(Float32, grid_points...)
us = 7000.0 * ones(Float32, grid_points...) .+ 1000.0 * randn(Float32, grid_points...)
g = 0.9 * ones(Float32, grid_points...)

angle_sampler = Moca.normal_angle_sampler(Float32, 0.087, 0.0)
res = 0.0019f0
profile = Float32.(0.00227 * ones(Float32, 21, 21))
shifter = Moca.membrane_shifter(0.0f0, 0.02f0, 0.0025f0)
beam_sampler = BeamSampler(profile, (res, res), angle_sampler, shifter)

E = light_transport_gpu(beam_sampler, Float32.(ua), Float32.(us), Float32.(g), 
grid_size, sim_bounds, n_rays, 200, max_loops=188*n_rays)

f = interactive_figure(E, 5)

return nothing
