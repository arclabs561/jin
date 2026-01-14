# Jin Rigorous Benchmark Task Runner

default:
    @just --list

# Generate all datasets (S, M, L)
gen-all:
    uvx --with numpy python scripts/generate_multiscale_data.py --scale all

# Generate specific scale (e.g. just gen S)
gen scale:
    uvx --with numpy python scripts/generate_multiscale_data.py --scale {{scale}}

# Run benchmark for a scale (e.g. just bench S)
bench scale:
    cargo run --example 04_rigorous_benchmark --release -- --scale {{scale}}

# Run full suite (S, M, L)
all:
    just gen S
    just bench S
    just gen M
    just bench M
    just gen L
    just bench L
    just plot

# Generate plots and report
plot:
    uvx --with numpy --with matplotlib python scripts/plot_pareto.py
    open data/multiscale/plots/benchmark_report.html
