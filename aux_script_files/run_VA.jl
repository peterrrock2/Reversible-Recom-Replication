import Pkg

push!(LOAD_PATH, "..")

using RandomNumbers
using MultiScaleMapSampler

pctGraphPath = joinpath("./JSON", "VA_precincts.json")
precinct_name = "loc_prec"
population_col = "TOTPOP"
num_dists = 11
pop_dev = 0.01
rng_seed = 132987987
steps = 100000
edge_weights = "connections"

println("Hello")
nodeData = Set([precinct_name, population_col])
base_graph = BaseGraph(
    pctGraphPath,
    population_col,
    inc_node_data = nodeData,
    edge_weights = edge_weights
)

println("Hello")
graph = MultiLevelGraph(base_graph, [precinct_name])

constraints = initialize_constraints()
add_constraint!(constraints, PopulationConstraint(graph, num_dists, pop_dev))

rng = PCG.PCGStateOneseq(UInt64, rng_seed)
partition = MultiLevelPartition(graph, constraints, num_dists; rng = rng)

proposal = build_forest_recom2(constraints)

measure = Measure(0.0, 1.0) # Measure(gamma, alpha)

writer = Writer(measure, constraints, partition, stdout)

run_metropolis_hastings!(
    partition,
    proposal,
    measure,
    steps,
    rng,
    writer = writer,
    output_freq = 1,
)


