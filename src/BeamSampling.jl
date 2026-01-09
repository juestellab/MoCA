################################################################################
# Hamilton-Monte-Carlo sampling
################################################################################
"""
    BeamLogDensity(dim, log_density)
Beam profile with dimension `dim` and logarithmic energy distribution `log_density`. 
`log_density` must be a function that accepts `dim`-dimensional vectors
(See also [LogDensityProblems](https://github.com/tpapp/LogDensityProblems.jl)).
"""
struct BeamLogDensity
    dim::Int
    log_density::Function
end

LogDensityProblems.logdensity(p::BeamLogDensity, θ) = p.log_density(θ)
LogDensityProblems.dimension(p::BeamLogDensity) = p.dim
LogDensityProblems.capabilities(::Type{BeamLogDensity}) = LogDensityProblems.LogDensityOrder{0}()


"""
    GenericSampler{T}
HMC sampler with return type `T`
"""
struct GenericSampler{T}
    "Hamiltonian of the log_density"
    hamiltonian
    "Sampling Alogrithm (defaults to GeneralisedNoUTurn)"
    proposal
    "Inital sample"
    θ₀
    "HMC adaptor (defaults to MassMatrixAdaptor(DiagEuclideanMetric) and StepSizeAdaptor(0.8))"
    adaptor
end

"""
    GenericSampler(log_density::BeamLogDensity,  T=Float32)
GenericSampler from log_density
"""
function GenericSampler(log_density::BeamLogDensity,  T=Float32, n_adapts=500)
    metric = DiagEuclideanMetric(ones(T, log_density.dim))
    hamiltonian = Hamiltonian(metric, log_density, Zygote)
    
    θ₀ = rand(T, log_density.dim)
    integrator = Leapfrog(find_good_stepsize(hamiltonian, θ₀))
    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(T(0.8), integrator))

    samples, _ = AdvancedHMC.sample(hamiltonian, proposal, θ₀, 1, adaptor, n_adapts, verbose=false)
    θ₀ = last(samples)
    return GenericSampler{T}(hamiltonian, proposal, θ₀, adaptor)
end

"""
    (dist::GenericSampler{T})(N::Integer)
Draw `N` samples from `dist`
"""
function (dist::GenericSampler{T})(N::Integer) where {T}
    samples, _ = AdvancedHMC.sample(dist.hamiltonian, dist.proposal, dist.θ₀, N, dist.adaptor,
                        drop_warmup=true, verbose=false)
    return samples
end

################################################################################
# Wrappers for list of precomputed samples and direct samplers
################################################################################
"""
    SampleList{T}
List of precomputed samples.
"""
struct SampleList{T}
    list::Vector{Vector{T}}
    index_array::Vector
end

SampleList{T}(density::BeamLogDensity, N) where T = begin
    sampler = GenericSampler(density, T)
    SampleList{T}(sampler(N), collect(1:N))
end

function (s::SampleList)(N::Integer)
    return s.list[rand(s.index_array, N)]
end

"""
    DirectSampler{T}
Wrapper for a function, that 
"""
struct DirectSampler{T}
    sampler::Function
end
(s::DirectSampler{T})(x) where T = return T.(s.sampler(x))

################################################################################
# GenericBeam
################################################################################
"""
    BeamSampler{T}
A struct that can sample the location of origin and inital direction for rays of
some illumination source.

Illumination distributions P(x, d), where x is the position and d the direction,
have to be of the form
&
P(x, d) = P(x) * P(x | d)
P(x, d) = P(f(ξ)) * P(ξ | d), x = f(ξ)
&
A location_transform f can be used to simplify the direction sampling


## Fields
 - ρ_p                  Sampler for location of origin
 - ρ_d                  Sampler for initial direction
 - location_transform   Transformation applied to location samples after direction is sampled
                        the respective direction of such a ray.
"""
struct BeamSampler{T}
    ρ_p::Union{SampleList{T}, DirectSampler{T}}
    ρ_d::DirectSampler{T}
    location_transform::Function
end
BeamSampler(ρ_p::GenericSampler{T}, ρ_d::DirectSampler{T}, transform, N_unique_samples) where T = begin
    sample_list = SampleList{T}(ρ_p(N_unique_samples), collect(1:N_unique_samples))
    BeamSampler{T}(sample_list, ρ_d, transform)
end

"""
    BeamSampler(weights::Array{T, N}, distances::NTuple{N, T}, ρ_d::DirectSampler{T}, transform)
Initialize a BeamSampler object from a weighted array over spatial pixels and a direct sampler for inital angles.
"""
BeamSampler(weights::Array{T, N}, distances::NTuple{N, T}, ρ_d::DirectSampler{T}, transform) where {T, N} = begin
    w = Weights(weights[:])
    ρ_p(N) = begin
        idx = StatsBase.sample(1:length(weights), w, N)
        s1, s2 = size(weights)
        _p = Array{T, 2}(undef, 2, N)
        @. _p[1, :] = ((idx - 1) % s1 + 1 + $rand(T, N)) * distances[1] - s1/2 * distances[1]
        @. _p[2, :] = ((idx - 1) ÷ s1 + 1 + $rand(T, N)) * distances[2] - s2/2 * distances[2]
        return _p
    end
    BeamSampler{T}(DirectSampler{T}(ρ_p), ρ_d, transform)
end

################################################################################
# Iteration method
################################################################################
"""
    next(beam::BeamSampler{T}, N::Integer)
Draw N samples from `beam`.
"""
function next(beam::BeamSampler{T}, N::Integer) where T
    _p = beam.ρ_p(N)
    d = beam.ρ_d(_p)
    p = T.(beam.location_transform(_p))
    W = ones(T, N)
    return p, d, W
end
