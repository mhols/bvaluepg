# Authors: Sofiane Taki-Eddine Rahmani, Gert Zöller, Sebastian Hainzl, Behnam Maleki Asayesh
# Italy adaptation of the original Main_script.jl.
#
# Intentional Italy-only differences:
# - Italy data path and rectangular analysis domain
# - reproducible seed and separate output directory
# - corrected post-burn-in index (BURNIN + 1)
# - mesh cells and run metadata for the PG-grid comparison


# ============================================================
# ETAS-SPDE MCMC
# ============================================================

# -----------------------------
# 1. Packages
# -----------------------------

using DelimitedFiles
using LinearAlgebra
using SparseArrays
using Statistics
using StatsBase
using Clustering
using Dates: DateTime
using Distributions: MersenneTwister, Exponential, Poisson, Normal
using TriangleMesh
using Random
using Random: AbstractRNG

const PROJECT_ROOT = @__DIR__

include(joinpath(PROJECT_ROOT, "src", "catalog.jl"))
include(joinpath(PROJECT_ROOT, "src", "branching_process.jl"))
include(joinpath(PROJECT_ROOT, "src", "spatialSPDE.jl"))
include(joinpath(PROJECT_ROOT, "src", "sampling_utilities.jl"))
include(joinpath(PROJECT_ROOT, "src", "SPDE-ETAS_sampler.jl"))
include(joinpath(PROJECT_ROOT, "src", "etas.jl"))

# ============================================================
# 2. Italy configuration
# ============================================================

# Defaults: small technical test. Optional arguments are data file, iterations,
# output directory, burn-in, and random seed, in that order.
const DATA_FILE = length(ARGS) >= 1 ? abspath(ARGS[1]) :
    joinpath(PROJECT_ROOT, "data", "italy_mc3_sofiane_first_200.txt")
const NITER = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
const OUTDIR = length(ARGS) >= 3 ? abspath(ARGS[3]) :
    joinpath(PROJECT_ROOT, "mcmc_results", "italy")

const BURNIN = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 0
const RANDOM_SEED = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 20260908

const K0 = 0.03
const α0 = 1.80
const c0 = 0.003
const p0 = 1.10
const q0 = 1.60
const D0 = 0.010
const γ0 = 0.40

const ρ0 = 1.5
const σ0 = 0.5
const μ0 = 0.01

const SPDE_ν = 0.5
const MAX_TRIANGLE_AREA = 0.5

# The adapter uses one model-space unit per 100 km and M - Mc as magnitude.
const SPACE_SCALE_KM = 100.0
const CATALOG_M0 = 0.0
const X_MIN_ROT_KM = 265.7071505562919
const X_MAX_ROT_KM = 1269.7572914814305
const Y_MIN_ROT_KM = -1414.1965686237359
const Y_MAX_ROT_KM = -34.860497554831255

# ============================================================
# 3. Load catalog
# ============================================================

Random.seed!(RANDOM_SEED)
data = readdlm(DATA_FILE)

time = data[:, 1]
mag  = data[:, 2]
lon  = data[:, 3]
lat  = data[:, 4]

Tmax = maximum(time)
pts = hcat(lon, lat)

# ============================================================
# 4. Mesh and SPDE objects
# ============================================================

width = (X_MAX_ROT_KM - X_MIN_ROT_KM) / SPACE_SCALE_KM
height = (Y_MAX_ROT_KM - Y_MIN_ROT_KM) / SPACE_SCALE_KM

all((0 .<= lon) .& (lon .<= width)) || error("x coordinates outside Italy domain")
all((0 .<= lat) .& (lat .<= height)) || error("y coordinates outside Italy domain")

corners = [
    0.0 0.0
    width 0.0
    width height
    0.0 height
]
domain = [
    (0.0, 0.0),
    (width, 0.0),
    (width, height),
    (0.0, height),
    (0.0, 0.0),
]

mesh = create_mesh(
    corners;
    point_marker = zeros(Int, size(corners, 1), 0),
    point_attribute = zeros(Float64, size(corners, 1), 0),
    info_str = "Triangular mesh of the Italy analysis domain.",
    verbose = false,
    check_triangulation = false,
    voronoi = true,
    delaunay = true,
    output_edges = true,
    output_cell_neighbors = true,
    quality_meshing = true,
    prevent_steiner_points_boundary = false,
    prevent_steiner_points = false,
    set_max_steiner_points = false,
    set_area_max = false,
    set_angle_min = false,
    add_switches = "a$(MAX_TRIANGLE_AREA)",
)

println("Italy catalogue: $(length(time)) events, $(round(Tmax, digits=2)) days")
println("Italy domain: $(round(width, digits=3)) x $(round(height, digits=3)) model units")
println("Mesh: $(mesh.n_point) points, $(mesh.n_cell) triangles")

C, C_tilde, G = component_matrices(mesh)
C_inv = spdiagm(0 => 1 ./ diag(C_tilde))
Sobs = observation_matrix(mesh, pts')
w = intersected_point_area(mesh, domain)

imat = Diagonal(ones(mesh.n_point))
M = spatialSPDE(2, mesh.n_point, G, C_inv, C)
di = Gridmesh(mesh, w .* Tmax, Sobs, imat)

catalog = Catalog(
    time,
    mag,
    CATALOG_M0,
    lon,
    lat,
    missing,
    Tmax,
    collect(mesh.point[1, :]),
    collect(mesh.point[2, :]),
)

# ============================================================
# 7. Run MCMC
# ============================================================

chain_K,
chain_α,
chain_c,
chain_p,
chain_q,
chain_D,
chain_γ,
chain_μ,
chain_ρ,
chain_σ,
chain_μspde,
chain_intensity,
chain_nbg = etas_spde_mcmc_full(
    catalog,
    M,
    di,
    SPDE_ν,
    Sobs,
    C,
    pts,
    NITER;
    K0 = K0,
    α0 = α0,
    c0 = c0,
    p0 = p0,
    q0 = q0,
    D0 = D0,
    γ0 = γ0,
    ρ0 = ρ0,
    σ0 = σ0,
    μ0 = μ0,
)

# ============================================================
# 8. Output directory
# ============================================================

mkpath(OUTDIR)
keep_from = BURNIN + 1
keep_from <= NITER || error("BURNIN must be smaller than NITER")

writedlm(joinpath(OUTDIR, "mesh_points.tsv"), hcat(mesh.point[1, :], mesh.point[2, :]), '\t')
writedlm(joinpath(OUTDIR, "mesh_cells.tsv"), Matrix(mesh.cell)', '\t')

# ============================================================
# 9. Save parameter chains
# ============================================================

params_matrix = hcat(
    chain_K[keep_from:end],
    chain_α[keep_from:end],
    chain_c[keep_from:end],
    chain_p[keep_from:end],
    chain_q[keep_from:end],
    chain_D[keep_from:end],
    chain_γ[keep_from:end],
    chain_ρ[keep_from:end],
    chain_σ[keep_from:end],
    chain_μspde[keep_from:end],
)

open(joinpath(OUTDIR, "chains_parameters.csv"), "w") do io
    println(io, "K,α,c,p,q,D,γ,ρ,σ,μspde")
    for i in axes(params_matrix, 1)
        println(io, join(params_matrix[i, :], ","))
    end
end

# ============================================================
# 10. Save raw intensity chains
# ============================================================

open(joinpath(OUTDIR, "chains_intensity.csv"), "w") do io
    for i in keep_from:length(chain_intensity)
        println(io, join(chain_intensity[i], ","))
    end
end

# ============================================================
# 11. Save intensity quantiles
# ============================================================

intensity_mat = reduce(hcat, chain_intensity[keep_from:end])'

q025 = mapslices(x -> quantile(x, 0.025), intensity_mat; dims=1)[:]
q250 = mapslices(x -> quantile(x, 0.250), intensity_mat; dims=1)[:]
q500 = mapslices(x -> quantile(x, 0.500), intensity_mat; dims=1)[:]
q750 = mapslices(x -> quantile(x, 0.750), intensity_mat; dims=1)[:]
q975 = mapslices(x -> quantile(x, 0.975), intensity_mat; dims=1)[:]

quantiles_mat = hcat(q975, q750, q500, q250, q025)

open(joinpath(OUTDIR, "chains_intensity_quantiles.csv"), "w") do io
    println(io, "q975,q750,q500,q250,q025")
    for i in axes(quantiles_mat, 1)
        println(io, join(quantiles_mat[i, :], ","))
    end
end

# ============================================================
# 12. Save number of background events
# ============================================================

open(joinpath(OUTDIR, "chains_nbg.csv"), "w") do io
    println(io, "nbg")
    for value in chain_nbg[keep_from:end]
        println(io, value)
    end
end

open(joinpath(OUTDIR, "run_metadata.txt"), "w") do io
    println(io, "purpose=technical_smoke_test_not_scientific_fit")
    println(io, "data_file=$DATA_FILE")
    println(io, "events=$(length(time))")
    println(io, "niter=$NITER")
    println(io, "burnin=$BURNIN")
    println(io, "random_seed=$RANDOM_SEED")
    println(io, "time_unit=days")
    println(io, "km_per_model_unit=$SPACE_SCALE_KM")
    println(io, "magnitude_input=M_minus_Mc")
    println(io, "catalog_M0=$CATALOG_M0")
    println(io, "mesh_points=$(mesh.n_point)")
    println(io, "mesh_triangles=$(mesh.n_cell)")
end

println("All Italy results saved in: $OUTDIR")
