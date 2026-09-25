# Technical smoke test of the Italy catalogue with the SPDE-ETAS sampler.
#
# This is deliberately not a scientific production fit. It checks the complete
# path from the existing BvaluePG event catalogue through coordinate/time
# conversion, meshing, MCMC, and separate result files.

using Dates
using DelimitedFiles
using LinearAlgebra
using SparseArrays
using Statistics
using StatsBase
using Clustering
using Distributions: Exponential, Poisson, Normal, LogNormal, Categorical, Product
using TriangleMesh
using Random

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(PROJECT_ROOT, "src", "catalog.jl"))
include(joinpath(PROJECT_ROOT, "src", "branching_process.jl"))
include(joinpath(PROJECT_ROOT, "src", "spatialSPDE.jl"))
include(joinpath(PROJECT_ROOT, "src", "sampling_utilities.jl"))
include(joinpath(PROJECT_ROOT, "src", "SPDE-ETAS_sampler.jl"))
include(joinpath(PROJECT_ROOT, "src", "etas.jl"))

# ============================================================
# Smoke-test configuration
# ============================================================

const INPUT_FILE = normpath(joinpath(
    PROJECT_ROOT,
    "..",
    "data",
    "preprocess_nnd_rot_cut_bin_Mc_2.5_eta_-4.60_dkm_20_events.csv",
))
const OUTDIR = joinpath(PROJECT_ROOT, "mcmc_results", "italy_smoke")

const MIN_MAGNITUDE = 3.0
const MAX_EVENTS = 200
const NITER = 2
const RANDOM_SEED = 20260908

# Fixed bounds from the existing Italy preprocessing metadata, in rotated km.
const X_MIN_ROT_KM = 265.7071505562919
const X_MAX_ROT_KM = 1269.7572914814305
const Y_MIN_ROT_KM = -1414.1965686237359
const Y_MAX_ROT_KM = -34.860497554831255

# Preserve distance ratios: one model-space unit equals 100 km in both axes.
const SPACE_SCALE_KM = 100.0
const MAX_TRIANGLE_AREA = 0.5
const SPDE_ν = 0.5

# These remain technical starting values, not calibrated Italy priors.
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

const INPUT_DATE_FORMAT = dateformat"yyyy-mm-dd HH:MM:SS.s"

struct ItalyEvent
    event_id::String
    datetime::DateTime
    magnitude::Float64
    x_rot_km::Float64
    y_rot_km::Float64
end

function parse_datetime_milliseconds(value::AbstractString)
    # The source contains microseconds, while Julia DateTime stores milliseconds.
    length(value) >= 23 || error("Unexpected datetime value: $value")
    return DateTime(value[1:23], INPUT_DATE_FORMAT)
end

function load_italy_events(path::AbstractString)
    isfile(path) || error("Italy input file not found: $path")
    lines = readlines(path)
    isempty(lines) && error("Italy input file is empty: $path")

    header = split(first(lines), '|'; keepempty = true)
    column = Dict(name => i for (i, name) in enumerate(header))
    required = ["event_id", "datetime", "mag", "x_rot_km", "y_rot_km", "inside_final_cut"]
    missing_columns = filter(name -> !haskey(column, name), required)
    isempty(missing_columns) || error("Missing columns: $(join(missing_columns, ", "))")

    selected = ItalyEvent[]
    for (row_index, line) in enumerate(Iterators.drop(lines, 1))
        line_number = row_index + 1
        values = split(line, '|'; keepempty = true)
        length(values) == length(header) || error(
            "Line $line_number has $(length(values)) fields; expected $(length(header)).",
        )

        values[column["inside_final_cut"]] == "True" || continue
        magnitude = parse(Float64, values[column["mag"]])
        magnitude >= MIN_MAGNITUDE || continue

        push!(selected, ItalyEvent(
            values[column["event_id"]],
            parse_datetime_milliseconds(values[column["datetime"]]),
            magnitude,
            parse(Float64, values[column["x_rot_km"]]),
            parse(Float64, values[column["y_rot_km"]]),
        ))
    end

    sort!(selected; by = event -> event.datetime)
    isempty(selected) && error("No events remain after the Italy selection.")
    return selected
end

function write_smoke_outputs(
    events,
    eligible_event_count,
    time_days,
    x_model,
    y_model,
    mesh,
    chains,
)
    mkpath(OUTDIR)

    open(joinpath(OUTDIR, "selected_catalog.tsv"), "w") do io
        println(io, "event_id\ttime_days\tmagnitude\tx_model\ty_model")
        for i in eachindex(events)
            println(
                io,
                join((events[i].event_id, time_days[i], events[i].magnitude, x_model[i], y_model[i]), '\t'),
            )
        end
    end
    open(joinpath(OUTDIR, "mesh_points.tsv"), "w") do io
        println(io, "node_id\tx_model\ty_model")
        for i in 1:mesh.n_point
            println(io, join((i, mesh.point[1, i], mesh.point[2, i]), '\t'))
        end
    end

    chain_K, chain_α, chain_c, chain_p, chain_q, chain_D, chain_γ,
    _, chain_ρ, chain_σ, chain_μspde, chain_intensity, chain_nbg = chains

    open(joinpath(OUTDIR, "parameter_chain.csv"), "w") do io
        println(io, "K,alpha,c,p,q,D,gamma,rho,sigma,mu_spde")
        writedlm(
            io,
            hcat(chain_K, chain_α, chain_c, chain_p, chain_q, chain_D, chain_γ, chain_ρ, chain_σ, chain_μspde),
            ',',
        )
    end
    open(joinpath(OUTDIR, "intensity_chain.csv"), "w") do io
        println(io, join(("node_$i" for i in 1:mesh.n_point), ','))
        writedlm(io, reduce(hcat, chain_intensity)', ',')
    end
    open(joinpath(OUTDIR, "background_count_chain.txt"), "w") do io
        println(io, "n_background")
        writedlm(io, chain_nbg)
    end

    open(joinpath(OUTDIR, "run_metadata.txt"), "w") do io
        println(io, "purpose=technical_smoke_test_not_scientific_fit")
        println(io, "input_file=$INPUT_FILE")
        println(io, "min_magnitude=$MIN_MAGNITUDE")
        println(io, "eligible_events=$eligible_event_count")
        println(io, "selected_events=$(length(events))")
        println(io, "selection_strategy=first_events_chronologically")
        println(io, "start_datetime=$(first(events).datetime)")
        println(io, "end_datetime=$(last(events).datetime)")
        println(io, "niter=$NITER")
        println(io, "random_seed=$RANDOM_SEED")
        println(io, "time_unit=days")
        println(io, "space_scale_km=$SPACE_SCALE_KM")
        println(io, "mesh_points=$(mesh.n_point)")
        println(io, "mesh_triangles=$(mesh.n_cell)")
        println(io, "max_triangle_area_model_units=$MAX_TRIANGLE_AREA")
    end
end

function main()
    Random.seed!(RANDOM_SEED)
    eligible_events = load_italy_events(INPUT_FILE)
    eligible_event_count = length(eligible_events)
    events = eligible_events[1:min(MAX_EVENTS, eligible_event_count)]

    datetimes = [event.datetime for event in events]
    all(diff(datetimes) .> Millisecond(0)) || error("Selected event times must be strictly increasing.")

    start_time = first(datetimes)
    time_days = Float64[
        Dates.value(event.datetime - start_time) / (1000 * 60 * 60 * 24)
        for event in events
    ]
    magnitude = [event.magnitude for event in events]
    x_model = [(event.x_rot_km - X_MIN_ROT_KM) / SPACE_SCALE_KM for event in events]
    y_model = [(event.y_rot_km - Y_MIN_ROT_KM) / SPACE_SCALE_KM for event in events]
    points = hcat(x_model, y_model)

    width = (X_MAX_ROT_KM - X_MIN_ROT_KM) / SPACE_SCALE_KM
    height = (Y_MAX_ROT_KM - Y_MIN_ROT_KM) / SPACE_SCALE_KM
    corners = [0.0 0.0; width 0.0; width height; 0.0 height]
    domain = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height), (0.0, 0.0)]

    all(0 .<= x_model .<= width) || error("Selected x coordinates lie outside the fixed domain.")
    all(0 .<= y_model .<= height) || error("Selected y coordinates lie outside the fixed domain.")

    mesh = create_mesh(
        corners;
        point_marker = zeros(Int, size(corners, 1), 0),
        point_attribute = zeros(Float64, size(corners, 1), 0),
        info_str = "Italy smoke-test mesh.",
        verbose = false,
        voronoi = true,
        delaunay = true,
        output_edges = true,
        output_cell_neighbors = true,
        quality_meshing = true,
        set_area_max = false,
        add_switches = "a$(MAX_TRIANGLE_AREA)",
    )

    C, C_tilde, G = component_matrices(mesh)
    C_inv = spdiagm(0 => 1 ./ diag(C_tilde))
    Sobs = observation_matrix(mesh, points')
    integration_area = intersected_point_area(mesh, domain)
    model = spatialSPDE(2, mesh.n_point, G, C_inv, C)
    gridmesh = Gridmesh(
        mesh,
        integration_area .* maximum(time_days),
        Sobs,
        Diagonal(ones(mesh.n_point)),
    )
    catalog = Catalog(
        time_days,
        magnitude,
        MIN_MAGNITUDE,
        x_model,
        y_model,
        start_time,
        maximum(time_days),
        collect(mesh.point[1, :]),
        collect(mesh.point[2, :]),
    )

    println("Italy smoke test: $(length(events)) events, Mc=$MIN_MAGNITUDE")
    println("Period: $(first(datetimes)) to $(last(datetimes)) ($(round(maximum(time_days), digits=2)) days)")
    println("Domain: $(round(width, digits=3)) x $(round(height, digits=3)) model units")
    println("Mesh: $(mesh.n_point) points, $(mesh.n_cell) triangles")

    chains = etas_spde_mcmc_full(
        catalog,
        model,
        gridmesh,
        SPDE_ν,
        Sobs,
        C,
        points,
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

    write_smoke_outputs(events, eligible_event_count, time_days, x_model, y_model, mesh, chains)
    println("Italy smoke-test outputs saved in: $OUTDIR")
end

main()
