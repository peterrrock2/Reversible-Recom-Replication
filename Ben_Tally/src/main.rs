use ben::decode::BenDecoder;
use clap::{Parser, ValueEnum};
// use indicatif::{ProgressBar, ProgressStyle};
use pbr::ProgressBar;
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Result, Value};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::time::Instant;

#[derive(Parser, Debug, Clone, ValueEnum, PartialEq)]
enum Mode {
    TallyKeys,
    CutEdges,
}

#[derive(Parser, Debug)]
#[command(
    name = "BEN Parquet Tally Tool",
    about = "A tool for tallying and saving data from BEN files to Parquet files.",
    version = "0.1.0"
)]
struct Args {
    #[arg(short, long, default_value = "cut-edges")]
    mode: Mode,
    #[arg(short, long)]
    graph_file: String,
    #[arg(short, long)]
    ben_file: String,
    #[arg(short, long, num_args(1..))]
    keys: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct JsonGraphData {
    directed: bool,
    multigraph: bool,
    graph: Vec<Value>,
    nodes: Vec<Value>,
    adjacency: Vec<Value>,
}

#[derive(Debug)]
struct Graph {
    nodes: Vec<Value>,
    edges: HashSet<(u64, u64)>,
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Args = Args::parse();

    match args.mode {
        Mode::TallyKeys => {
            let graph = make_graph_from_json(&args.graph_file)?;
            let output_file = &args.ben_file.replace(".jsonl.ben", "_tallies.parquet");

            tally_and_save_from_key_list(graph, &args.ben_file, &output_file, args.keys)?;
        }
        Mode::CutEdges => {
            let graph = make_graph_from_json(&args.graph_file)?;
            let output_file = &args.ben_file.replace(".jsonl.ben", "_cut_edges.parquet");

            tally_and_save_cut_edges(graph, &args.ben_file, &output_file)?;
        }
    }
    Ok(())
}

fn make_graph_from_json(file_path: &str) -> Result<Graph> {
    // Read the JSON file
    let mut file = File::open(file_path).expect("File not found");
    let mut data = String::new();
    file.read_to_string(&mut data).expect("Unable to read file");

    // Parse the JSON data
    let graph_data: JsonGraphData = serde_json::from_str(&data)?;

    let mut graph = Graph {
        nodes: graph_data.nodes.clone(),
        edges: HashSet::new(),
    };

    for (source_idx, target_array) in graph_data
        .adjacency
        .iter()
        .enumerate()
        .map(|(x, y)| (x as u64, y.as_array().unwrap().to_vec()))
    {
        for target_data in target_array {
            let target_idx: u64 = target_data["id"].as_u64().unwrap();
            graph
                .edges
                .insert((source_idx.min(target_idx), source_idx.max(target_idx)));
        }
    }

    Ok(graph)
}

fn tally_keys(
    graph: &Graph,
    assignment: &Vec<u16>,
    keys: &Vec<String>,
) -> HashMap<String, HashMap<u16, f64>> {
    let partition_values: HashSet<u16> = HashSet::from_iter(assignment.iter().cloned());

    let mut tallies: HashMap<String, HashMap<u16, f64>> = keys
        .iter()
        .map(|x| {
            (
                x.clone(),
                partition_values.iter().map(|&y| (y, 0.0)).collect(),
            )
        })
        .collect();

    for (idx, node) in graph.nodes.iter().enumerate() {
        let partition_key = assignment[idx];
        for key in keys {
            let json_val = &node[key];
            let value = match json_val {
                Value::Number(n) => n.as_f64().unwrap(),
                Value::String(s) => s.parse::<f64>().unwrap(),
                _ => panic!(
                    "Invalid value type in JSON file. Failed to parse {:?} as f64 for key {:?}",
                    json_val, key
                ),
            };

            *tallies
                .get_mut(key)
                .unwrap()
                .get_mut(&partition_key)
                .unwrap() += value;
        }
    }

    tallies
}

fn save_tallies_to_parquet(
    file_path: &str,
    tallies: &Vec<(u64, u32, u32, HashMap<String, HashMap<u16, f64>>)>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut sample_numbers = Vec::new();
    let mut n_reps_numbers = Vec::new();
    let mut accepted_numbers = Vec::new();

    let mut keys = Vec::new();
    let mut partition_data: HashMap<u16, Vec<Option<f64>>> = HashMap::new();

    // Initialize partition_data with empty vectors for each unique partition key
    for (_, _, _, tally) in tallies {
        for (_, sub_map) in tally {
            for (&partition_key, _) in sub_map {
                partition_data.entry(partition_key).or_insert_with(Vec::new);
            }
        }
    }

    // Fill in the data
    for (sample_num, n_reps, accepted_num, tally) in tallies {
        for (key, sub_map) in tally {
            sample_numbers.push(*sample_num);
            n_reps_numbers.push(*n_reps);
            accepted_numbers.push(*accepted_num);
            keys.push(key.clone());
            for (&partition_key, value) in partition_data.iter_mut() {
                value.push(sub_map.get(&partition_key).copied());
            }
        }
    }

    let mut df = DataFrame::new(vec![
        Series::new("step", sample_numbers),
        Series::new("n_reps", n_reps_numbers),
        Series::new("accepted_count", accepted_numbers),
        Series::new("sum_columns", keys),
    ])?;

    // Add columns for each partition key
    for (partition_key, values) in partition_data {
        df.with_column(Series::new(&format!("district_{}", partition_key), values))?;
    }

    let mut file = File::create(file_path)?;
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Brotli(Some(
            BrotliLevel::try_new(6).unwrap(),
        )))
        .finish(&mut df)?;

    Ok(())
}

fn tally_and_save_from_key_list(
    graph: Graph,
    in_file_name: &str,
    out_file_name: &str,
    key_list: Vec<String>,
) -> io::Result<()> {
    let n_pb_tics = 1000;

    let mut pb = ProgressBar::new(n_pb_tics);

    // pb.set_draw_target(indicatif::ProgressDrawTarget::stderr());
    let mut pb_tics = 0; // For manual printing since my

    let mut ben_file = File::open(in_file_name).expect("BEN file not found");

    let line_checker = BenDecoder::new(&ben_file).unwrap();

    let mut line_count: usize = 0;
    for _ in line_checker.enumerate() {
        line_count += 1;
    }
    print!("Found {:?} unique plans in BEN file\r", line_count);
    println!();

    let pb_step_size = (line_count / n_pb_tics as usize) as u32;
    // pb.set_style(
    //     ProgressStyle::with_template(
    //         "[{elapsed_precise}] {bar:50.cyan/blue} {pos:>7}/{len:7} [ETA: {eta_precise}] {msg}",
    //     )
    //     .unwrap()
    //     .progress_chars("#>-"),
    // );
    let mut previous_step = 0;

    ben_file.seek(SeekFrom::Start(0))?;

    let ben_reader = BufReader::new(ben_file);

    let decoder = BenDecoder::new(ben_reader).unwrap();

    let mut all_tallies: Vec<(u64, u32, u32, HashMap<String, HashMap<u16, f64>>)> =
        Vec::with_capacity(line_count);

    let mut sample_count = 1;
    let mut accepted_count = 1;

    const BATCH_SIZE: usize = 100;
    let mut batch = Vec::with_capacity(BATCH_SIZE);

    let start_time = Instant::now();
    for (_idx, record) in decoder.enumerate() {
        match record {
            Ok((assignment, n_reps)) => {
                batch.push((assignment, n_reps));
                if batch.len() == BATCH_SIZE {
                    let results: Vec<_> = batch
                        .par_iter()
                        .map(|(assignment, n_reps)| {
                            let tallies = tally_keys(&graph, assignment, &key_list);
                            (*n_reps, tallies)
                        })
                        .collect();

                    for (n_reps, tallies) in results {
                        all_tallies.push((sample_count, n_reps as u32, accepted_count, tallies));
                        sample_count += n_reps as u64;
                        accepted_count += 1;
                    }

                    batch.clear();
                }
            }
            Err(e) => {
                panic!("Error: {:?}", e);
            }
        }
        if accepted_count - previous_step >= pb_step_size {
            pb.inc();

            let elapsed = start_time.elapsed();
            let elapsed_secs = elapsed.as_secs_f64();
            let rate = (pb_tics + 1) as f64 / elapsed_secs; // Current rate (iterations per second)
            let remaining_secs = (n_pb_tics - pb_tics - 1) as f64 / rate; // Remaining time in seconds

            let elapsed_mins = (elapsed_secs / 60.0).floor() as u64;
            let elapsed_remain_secs = (elapsed_secs % 60.0) as u64;
            let remaining_mins = (remaining_secs / 60.0).floor() as u64;
            let remaining_remain_secs = (remaining_secs % 60.0) as u64;

            // Update the progress bar message to display the formatted elapsed and remaining times
            pb.message(&format!(
                "Elapsed: {}m {}s, ETA: {}m {}s ",
                elapsed_mins, elapsed_remain_secs, remaining_mins, remaining_remain_secs
            ));
            pb_tics += 1;

            // pb.inc(1);
            // println!("Processed {:?} records stdout", accepted_count);
            // eprintln!("Processed {:?} records stderr", accepted_count);
            io::stderr().flush().unwrap();
            io::stdout().flush().unwrap();
            // pb.println(format!("Processed {:?} records", accepted_count));
            previous_step = accepted_count;
        }
    }

    // Process any remaining records in the batch
    if !batch.is_empty() {
        let results: Vec<_> = batch
            .par_iter()
            .map(|(assignment, n_reps)| {
                let tallies = tally_keys(&graph, assignment, &key_list);
                (*n_reps, tallies)
            })
            .collect();

        for (n_reps, tallies) in results {
            all_tallies.push((sample_count, n_reps as u32, accepted_count, tallies));
            sample_count += n_reps as u64;
            accepted_count += 1;
        }
    }

    pb.finish();

    println!();

    save_tallies_to_parquet(out_file_name, &all_tallies).expect("Unable to save tallies");

    Ok(())
}

fn cut_edges(graph: &Graph, assignment: &Vec<u16>) -> u32 {
    let mut cut_edges = 0;

    for edge in &graph.edges {
        let (source, target) = edge;
        if assignment[*source as usize] != assignment[*target as usize] {
            cut_edges += 1;
        }
    }

    cut_edges
}

fn tally_and_save_cut_edges(
    graph: Graph,
    in_file_name: &str,
    out_file_name: &str,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let n_pb_tics = 100;

    let mut pb = ProgressBar::new(n_pb_tics);

    let mut ben_file = File::open(in_file_name).expect("BEN file not found");

    let line_checker = BenDecoder::new(&ben_file).unwrap();

    let mut line_count: usize = 0;
    for _ in line_checker.enumerate() {
        line_count += 1;
    }
    print!("Found {:?} unique plans in BEN file\r", line_count);
    println!();

    let pb_step_size = (line_count / n_pb_tics as usize) as u32;
    // pb.set_style(
    //     ProgressStyle::with_template(
    //         "[{elapsed_precise}] {bar:50.cyan/blue} {pos:>7}/{len:7} [ETA: {eta_precise}] {msg}",
    //     )
    //     .unwrap()
    //     .progress_chars("#>-"),
    // );
    let mut previous_step = 0;

    ben_file.seek(SeekFrom::Start(0))?;

    let ben_reader = std::io::BufReader::new(ben_file);

    let decoder = BenDecoder::new(ben_reader).unwrap();

    let mut sample_nums = Vec::with_capacity(line_count);
    let mut n_reps_nums = Vec::with_capacity(line_count);
    let mut accepted_nums = Vec::with_capacity(line_count);
    let mut cut_edge_counts = Vec::with_capacity(line_count);

    let mut sample_count = 1;
    let mut accepted_count = 1;

    const BATCH_SIZE: usize = 100;
    let mut batch = Vec::with_capacity(BATCH_SIZE);

    for (_idx, record) in decoder.enumerate() {
        match record {
            Ok((assignment, count)) => {
                batch.push((assignment, count));
                if batch.len() == BATCH_SIZE {
                    let results: Vec<_> = batch
                        .par_iter()
                        .map(|(assignment, count)| {
                            let cut_edges = cut_edges(&graph, assignment);
                            (*count, cut_edges)
                        })
                        .collect();

                    for (n_reps, counts) in results {
                        sample_nums.push(sample_count);
                        n_reps_nums.push(n_reps as u32);
                        accepted_nums.push(accepted_count);
                        cut_edge_counts.push(counts);
                        sample_count += n_reps as u64;
                        accepted_count += 1;
                    }

                    batch.clear();
                }
                if accepted_count - previous_step >= pb_step_size {
                    // pb.inc(1);
                    pb.inc();
                    previous_step = accepted_count;
                }
            }
            Err(e) => {
                panic!("Error: {:?}", e);
            }
        }
    }

    if !batch.is_empty() {
        let results: Vec<_> = batch
            .par_iter()
            .map(|(assignment, count)| {
                let cut_edges = cut_edges(&graph, assignment);
                (*count, cut_edges)
            })
            .collect();

        for (n_reps, counts) in results {
            sample_nums.push(sample_count);
            n_reps_nums.push(n_reps as u32);
            accepted_nums.push(accepted_count);
            cut_edge_counts.push(counts);
            sample_count += n_reps as u64;
            accepted_count += 1;
        }
    }

    pb.finish();

    println!();

    let mut df = DataFrame::new(vec![
        Series::new("step", sample_nums),
        Series::new("n_reps", n_reps_nums),
        Series::new("accepted_count", accepted_nums),
        Series::new("cut_edges", cut_edge_counts),
    ])?;

    let mut file = File::create(out_file_name)?;

    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Brotli(Some(
            BrotliLevel::try_new(6).unwrap(),
        )))
        .finish(&mut df)?;

    Ok(())
}
