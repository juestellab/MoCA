"""
Pencil and isotropic point sources for the tests and the validation harness.

"""

using Moca: BeamSampler

const_pos(v) = N -> repeat(Float32.(reshape(collect(v), 3, 1)), 1, N)
const_dir(v) = p -> repeat(Float32.(reshape(collect(v), 3, 1)), 1, size(p, 2))

"""Directions uniform on the unit sphere (`cos θ = 2u - 1`)."""
function isotropic_dir()
    p -> begin
        N = size(p, 2)
        cosθ = 2 .* rand(Float32, N) .- 1
        sinθ = sqrt.(max.(0f0, 1 .- cosθ .^ 2))
        ϕ = 2f0 * Float32(π) .* rand(Float32, N)
        d = Array{Float32}(undef, 3, N)
        @. d[1, :] = sinθ * cos(ϕ)
        @. d[2, :] = sinθ * sin(ϕ)
        @. d[3, :] = cosθ
        d
    end
end

pencil_beam(pos, dir) = BeamSampler{Float32}(const_pos(pos), const_dir(dir), identity)
isotropic_beam(pos) = BeamSampler{Float32}(const_pos(pos), isotropic_dir(), identity)

"""Run `light_transport_gpu` without its unconditional printing."""
quiet_transport(f) = redirect_stdout(() -> redirect_stderr(f, devnull), devnull)