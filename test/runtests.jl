"""
Moca test suite. Transport and sensitivity tests need a CUDA device; the MCX
cross-validation lives in `test/validation/` (see its README).
"""

using Test
using CUDA

@testset "Moca" begin
    include("test_scatter.jl")
    include("test_sampling.jl")
    include("test_layout.jl")

    if CUDA.functional()
        include("test_transport.jl")
        include("test_sensitivity.jl")
    else
        @warn "CUDA is not functional -- skipping GPU transport and sensitivity tests"
    end
end
