# Moca validation

Cross-validation of Moca against [MCX](https://mcx.space) (Fang & Boas 2009),
against diffusion theory, and of the directional derivative against finite
differences. The fast unit tests live in `test/` (`Pkg.test()`).

## Running

```bash
# Python side (pmcx); pyproject.toml pins the dependencies
cd test/validation && uv sync --python 3.12 && cd -

# Moca vs MCX  (10 replicates x 4e8 rays / 1e8 photons per case; ~1 h Moca, ~10 min MCX)
julia --project=test/validation test/validation/run_moca.jl --all
(cd test/validation && uv run python run_mcx.py --all)
julia --project=test/validation test/validation/compare.jl       # -> results/report.md

# Diffusion limit and similarity relation  -> results/analytic.md
julia --project=test/validation test/validation/analytic.jl

# Derivative vs finite differences          -> results/derivatives.md
julia --project=test/validation test/validation/derivatives.jl
```

`--quick` on `run_moca.jl`, `run_mcx.py` and `derivatives.jl` runs a smoke test
with too few rays to mean anything.

## Matching the physics

Both codes are unbiased for absorbed energy when the physics is matched:

| | Moca | MCX setting |
|---|---|---|
| Refractive index | not modelled | `n = 1` everywhere |
| Fresnel / specular | none | `isreflect = 0`, `isspecular = 0` |
| Phase function | Henyey–Greenstein | Henyey–Greenstein |
| Time domain | steady state | one CW gate |
| Source position | 0-based, in L | `issrcfrom0 = 1` (pmcx default is 1-based) |

Moca returns the absorbed fraction per voxel per launched ray. MCX's fluence is
converted with `E = μ_a · φ · ΔV`, and `run_mcx.py` checks each run against
MCX's own `energyabs` tally. MCX tallies fluence in float32, which loses
deposits in the brightest voxels at 1e9 photons per run (0.15% at the C1
source), so it runs as 10 replicates of 1e8.

## Phantoms

`phantoms.json` is read by both `run_moca.jl` and `run_mcx.py`. Layers are
0-based half-open voxel ranges in z.

- **C1_infinite_g09**: a homogeneous 100 mm cube (g = 0.9) with an isotropic
  source at the centre. Both codes kill photons at the grid face
  (`pad_mm = 0`); continuing the medium in Moca alone shifts the interior by
  about 1e-4. The comparison is restricted to a 25 mm ball around the source
  (`compare.mode = "radius"`).
- **C3_semiinf, C4_twolayer**: a pencil beam into a slab, wrapped in a
  `shell_voxels`-thick vacuum shell. MCX's label 0 and Moca's `μ_a = μ_s = 0`
  give both codes the same escaping boundary.

`compare.jl` refuses to compare unless both codes' label volumes match exactly.
That guards against array-order mistakes, since numpy is C-ordered.

## Results

Moca vs MCX (`results/report.md`; PASS: |mean z| < 0.5 and |rel. bias| < 1%):

| case | mean z | rel. bias | verdict |
|---|---:|---:|---|
| C1_infinite_g09 | +0.009 | −0.000% | PASS |
| C3_semiinf | −0.045 | −0.017% | PASS |
| C4_twolayer | +0.009 | −0.001% | PASS |

Diffusion theory (`results/analytic.md`): g = 0.9 and g = 0 at equal μ_s' agree
to within 1%, and both converge to the diffusion fluence (within 1.5% for r ≥ 12 mm).

Derivative (`results/derivatives.md`): at the best step in the sweep,
central differences agree with `dE` within MC noise on every functional: rms z
is 1.07 (depth), 1.15 (x-profile), 0.79 (octants) and 0.23 (total). Larger
steps fail on the profiles because of O(eps²) truncation.