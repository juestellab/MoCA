"""
CPU tests for the Henyey-Greenstein sampler `Moca.scatter`, via its closed-form
moments <cos θ> = g and <cos² θ> = (1 + 2g²)/3, where cos θ = d · d'.
"""

using Test
using Moca
using Statistics
using LinearAlgebra

const NSAMP = 400_000

"""Sample `n` deflection cosines about incoming direction `d` with anisotropy `g`."""
function deflections(d, g; n=NSAMP)
    d = normalize(collect(Float32, d))
    cosθ = Vector{Float32}(undef, n)
    norms = Vector{Float32}(undef, n)
    outz = Vector{Float32}(undef, n)
    for i in 1:n
        d1, d2, d3 = Moca.scatter(d[1], d[2], d[3], Float32(g))
        cosθ[i] = d1 * d[1] + d2 * d[2] + d3 * d[3]
        norms[i] = sqrt(d1^2 + d2^2 + d3^2)
        outz[i] = d3
    end
    (; cosθ, norms, outz)
end

# 4 sigma on the mean of a bounded [-1,1] variate over NSAMP draws.
tol(n=NSAMP) = 4 / sqrt(n)

@testset "scatter" begin

    @testset "output is a unit vector" begin
        for d in ([0, 0, 1], [0, 0, -1], [1, 0, 0], normalize([1.0, 2.0, 3.0])),
            g in (0.0, 0.5, 0.9, -0.5)
            r = deflections(d, g; n=20_000)
            @test maximum(abs.(r.norms .- 1)) < 1e-3
        end
    end

    @testset "HG first moment <cos θ> == g" begin
        d = normalize([0.3, -0.5, 0.6])
        for g in (0.0, 0.3, 0.6, 0.9, -0.3, -0.6)
            r = deflections(d, g)
            @test isapprox(mean(r.cosθ), g; atol=tol())
        end
    end

    @testset "HG second moment <cos² θ> == (1+2g²)/3" begin
        d = normalize([0.3, -0.5, 0.6])
        for g in (0.0, 0.5, 0.9)
            r = deflections(d, g)
            @test isapprox(mean(r.cosθ .^ 2), (1 + 2g^2) / 3; atol=tol())
        end
    end

    # The |d₃| ≈ 1 branch must keep the sign of d₃: a ray going -z that
    # forward-scatters stays near -z.
    @testset "polar branch |d₃| ≈ 1 carries the sign" begin
        for g in (0.0, 0.5, 0.9)
            up = deflections([0, 0, 1], g)
            dn = deflections([0, 0, -1], g)
            @test isapprox(mean(up.cosθ), g; atol=tol())
            @test isapprox(mean(dn.cosθ), g; atol=tol())
            @test isapprox(mean(dn.outz), -g; atol=tol())
            @test isapprox(mean(up.outz), g; atol=tol())
        end
    end

    # g -> -1 is not special-cased; the limit is benign but cancels in floating point.
    @testset "strong back-scattering" begin
        d = normalize([0.3, -0.5, 0.6])
        r = deflections(d, -0.99)
        @test all(isfinite, r.cosθ)
        @test maximum(abs.(r.norms .- 1)) < 1e-3
        @test isapprox(mean(r.cosθ), -0.99; atol=tol())

        rm1 = deflections(d, -1.0; n=10_000)
        @test all(isfinite, rm1.cosθ)
        @test isapprox(mean(rm1.cosθ), -1.0; atol=1e-2)
    end
end
