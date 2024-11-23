# Reversible Recom Replication

This is a repository containing some of the code needed to replicate the
work for our reversible recom paper. This repository is not complete at this
time.


## Simple Setup

You should be able to get up and running by simply invoking the `setup_all.sh`
script included here from your command line. You will need to have both Julia
and Cargo installed in order for this to run without errors. 


To install Cargo, visit the 
[Rustup Download Page](https://doc.rust-lang.org/cargo/getting-started/installation.html)
and follow the instructions for your operating system. Likewise, to install Julia,
visit the [Julia Installation Page](https://julialang.org/downloads/).
All of the work for this project was done using Julia version 1.9.3, but it should
still work the same in newer Julia versions.


## Replicating the Work

We would like to make sure that we are targeting the Spanning Tree Distribution using Forest Recom. The easiest way to
do this is to check against known distributions. Here are known ground-truth distributions for the 4x4 -> 4 and 7x7 -> 7
distributions.

**4x4**
```
 cuts  tree_count  n_plans  probability
    8         256        1    39.143731
   10         224       14    34.250765
   11          96       24    14.678899
   12          78       78    11.926606
```


**7x7**
```
 cuts   tree_count  n_plans  probability
   28  73446750000      420     2.080769
   29 250665840000     5408     7.101440
   30 528671139300    43468    14.977414
   31 698394196800   219704    19.785720
   32 732191421608   884620    20.743206
   33 577889926624  2686928    16.371797
   34 366428925818  6578950    10.381043
   35 186993438208 12985744     5.297581
   36  78541020648 21167576     2.225091
   37  26977180928 28289752     0.764272
   38   7588753206 31084950     0.214992
   39   1684381728 27036848     0.047719
   40    282316256 17848860     0.007998
   41     31884256  7971064     0.000903
   42      1949522  1949522     0.000055
```


There are three scripts in this repo

- test_VA.sh
- test_4.sh
- test_7.sh

each of these will run the Julia code with 100k steps and then compute the cut-edge counts, and then 
the 4x4 and 7x7 files will print the resulting distribution which you can compare against the above
known distributions for the 4x4 and the 7x7 grids.


## More Detailed Setup instructions in case the script fails


To run the 'test' script files contained in here, you will need to download the
the correct version of the Julia code from github using the included setup file:

```
julia aux_script_files/setup.jl
```

Alternatively, you can activate the Julia interactive terminal by calling `julia`
and then make the following commands:

```
julia> import Pkg
julia> Pkg.activate(".")
julia> Pkg.add(RandomNumbers)
julia> Pkg.add(url="https://github.com/peterrrock2/Multi-Scale-Map-Sampler")
```

### Other dependencies

In order to convert the "atlas" format that is output by the MSMS code to an assignment-vector 
format, it is necessary to install the `msms_parser` and `ben` cli tools. This can be downloaded using the
Cargo package manager. 
The `ben` cli tool may then be installed using the command

```
cargo install binary-ensemble
```

and `msms_parser` can be installed using

```
cargo install --git https://github.com/peterrrock2/msms_parser.git
```

In addition to these CLI tools, you will need the `ben-tally` cli tool that allows for tallying across an output BEN file.
This is included in the 'Ben_Tally' folder and may be installed using

```
cargo install --path ./Ben_Tally
```

There are a couple of different modes for the `ben-tally` CLI. The default mode tallies the cut edges of the partition
and may be invoked using

```
ben-tally -g <path-to-JSON-file> -b <path-to-BEN-file-to-tally>
```

This will output a file with the suffix "cut-edges.parquet" added to it which can then be read in python.
In the event that you would like to tally particular attributes of graph, you will need to use the command


```
ben-tally -m tally-keys -g <path-to-JSON-file> -b <path-to-BEN-file-to-tally> --keys <list-of-keys-to-tally>
```

so an example usage would be

```
ben-tally -m tally-keys -g ./JSON/VA_precincts.json -b VA_Forest_steps_10000000_rng_seed_278986_gamma_0.0_alpha_1.0_ndists_11_20241112_124346.jsonl.ben --keys G16DPRS G16RPRS
```


