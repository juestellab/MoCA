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
 - ρ_p                  `N -> position matrix`, samples `N` locations of origin
 - ρ_d                  `p -> direction matrix`, samples one direction per column of `p`
 - location_transform   Transformation applied to location samples after direction is sampled
                        the respective direction of such a ray.

`ρ_p`, `ρ_d` and `location_transform` are plain functions; whatever element type
they return is converted to `T` by [`next`](@ref). Since nothing else fixes the
element type, `T` must be given explicitly: `BeamSampler{Float32}(ρ_p, ρ_d, f)`.
"""
struct BeamSampler{T}
    ρ_p::Function
    ρ_d::Function
    location_transform::Function
end

"""
    BeamSampler(weights::Array{T, N}, distances::NTuple{N, T}, ρ_d, transform)
Initialize a BeamSampler object from a weighted array over spatial pixels and a sampler for inital angles.
`T` is taken from `weights`.
"""
BeamSampler(weights::Array{T, N}, distances::NTuple{N, T}, ρ_d, transform) where {T, N} =
    BeamSampler{T}(sampler_from_weight_array(weights, distances), ρ_d, transform)

################################################################################
# Iteration method
################################################################################
"""
    next(beam::BeamSampler{T}, N::Integer)
Draw N samples from `beam`.
"""
function next(beam::BeamSampler{T}, N::Integer) where T
    _p = T.(beam.ρ_p(N))
    d = T.(beam.ρ_d(_p))
    p = T.(beam.location_transform(_p))
    W = ones(T, N)
    return p, d, W
end
