use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, error};
use tracing_subscriber::{filter::EnvFilter, fmt, prelude::*};

use rust_core::{
    config::ProcessingConfig,
    document_processor::DocumentProcessor,
};

#[derive(Parser)]
#[command(name = "rust-core")]
#[command(about = "High-performance Rust core for hybrid ML pipeline")]
#[command(version = "1.0.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Process documents in a directory
    Process {
        /// Input directory containing documents
        #[arg(short, long)]
        input: PathBuf,
        
        /// Output directory for results
        #[arg(short, long)]
        output: PathBuf,
        
        /// Configuration file path
        #[arg(short, long)]
        config: Option<PathBuf>,
        
        /// Maximum concurrent documents
        #[arg(long, default_value = "16")]
        max_concurrent: usize,
        
        /// Memory limit in GB
        #[arg(long, default_value = "60")]
        memory_limit: usize,
    },
    /// Show system information
    SystemInfo,
    /// Run performance benchmark
    Benchmark {
        /// Number of test documents to process
        #[arg(short, long, default_value = "10")]
        count: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    init_tracing()?;
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Process { 
            input, 
            output, 
            config, 
            max_concurrent, 
            memory_limit 
        } => {
            process_documents(input, output, config, max_concurrent, memory_limit).await?;
        }
        Commands::SystemInfo => {
            show_system_info().await?;
        }
        Commands::Benchmark { count } => {
            run_benchmark(count).await?;
        }
    }
    
    Ok(())
}

/// Initialize structured logging
fn init_tracing() -> Result<()> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("rust_core=info,warn,error"));
    
    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_target(true)
                .with_thread_ids(true)
                .with_level(true)
                .compact()
        )
        .with(env_filter)
        .init();
    
    Ok(())
}

/// Process documents using the high-performance pipeline
async fn process_documents(
    input_dir: PathBuf,
    output_dir: PathBuf,
    config_path: Option<PathBuf>,
    max_concurrent: usize,
    memory_limit: usize,
) -> Result<()> {
    info!("=== Rust High-Performance Core v1.0.0 ===");
    info!("Input directory: {:?}", input_dir);
    info!("Output directory: {:?}", output_dir);
    info!("Max concurrent: {}", max_concurrent);
    info!("Memory limit: {}GB", memory_limit);
    
    // Validate input directory
    if !input_dir.exists() {
        anyhow::bail!("Input directory does not exist: {:?}", input_dir);
    }
    
    // Load or create configuration
    let config = if let Some(config_path) = config_path {
        info!("Loading configuration from: {:?}", config_path);
        ProcessingConfig::from_file(&config_path)?
    } else {
        info!("Using default M3 Max optimized configuration");
        ProcessingConfig::default_m3_max(max_concurrent, memory_limit)
    };
    
    // Validate configuration
    config.validate()?;
    
    // Create document processor
    info!("Initializing M3 Max optimized document processor");
    let processor = DocumentProcessor::new(config).await?;
    
    // Process documents
    info!("Starting document processing pipeline");
    let start_time = std::time::Instant::now();
    
    let stats = processor.process_directory(&input_dir, &output_dir).await?;
    
    let total_time = start_time.elapsed();
    
    // Display results
    info!("=== Processing Complete ===");
    info!("Documents processed: {}", stats.documents_processed);
    info!("QA pairs generated: {}", stats.total_qa_pairs_generated);
    info!("Average quality: {:.3}", stats.average_quality);
    info!("Total time: {:?}", total_time);
    info!("Throughput: {:.2} docs/hour", stats.throughput_per_hour());
    info!("Peak memory: {:.2}GB", stats.memory_peak_gb);
    info!("Errors: {}", stats.errors_encountered);
    
    // Performance analysis
    let target_throughput = 25.0;
    if stats.throughput_per_hour() >= target_throughput {
        info!("✅ Performance target achieved!");
    } else {
        error!("⚠️ Below performance target of {} docs/hour", target_throughput);
    }
    
    info!("Output written to: {:?}", output_dir);
    
    Ok(())
}

/// Show system information for optimization tuning
async fn show_system_info() -> Result<()> {
    use sysinfo::{System, SystemExt, CpuExt};
    
    let mut sys = System::new_all();
    sys.refresh_all();
    
    println!("=== System Information ===");
    println!("OS: {}", sys.long_os_version().unwrap_or_else(|| "Unknown".to_string()));
    println!("Kernel: {}", sys.kernel_version().unwrap_or_else(|| "Unknown".to_string()));
    println!("Host: {}", sys.host_name().unwrap_or_else(|| "Unknown".to_string()));
    
    println!("\n=== CPU Information ===");
    if let Some(cpu) = sys.cpus().first() {
        println!("Brand: {}", cpu.brand());
        println!("Frequency: {} MHz", cpu.frequency());
    }
    println!("CPU cores: {}", sys.cpus().len());
    
    println!("\n=== Memory Information ===");
    println!("Total RAM: {:.2} GB", sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("Available RAM: {:.2} GB", sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("Used RAM: {:.2} GB", sys.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("Total Swap: {:.2} GB", sys.total_swap() as f64 / 1024.0 / 1024.0 / 1024.0);
    
    println!("\n=== Optimization Recommendations ===");
    let total_gb = sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    let recommended_limit = (total_gb * 0.7) as usize;
    let recommended_concurrent = sys.cpus().len();
    
    println!("Recommended memory limit: {}GB (70% of total)", recommended_limit);
    println!("Recommended max concurrent: {}", recommended_concurrent);
    
    if sys.cpus().len() >= 16 {
        println!("✅ System has sufficient CPU cores for M3 Max optimization");
    } else {
        println!("⚠️ System may not be M3 Max - consider adjusting concurrency");
    }
    
    if total_gb >= 64.0 {
        println!("✅ System has sufficient RAM for large-scale processing");
    } else {
        println!("⚠️ Consider reducing memory limit for this system");
    }
    
    Ok(())
}

/// Run performance benchmark
async fn run_benchmark(document_count: usize) -> Result<()> {
    use std::io::Write;
    use tempfile::TempDir;
    
    info!("Running performance benchmark with {} test documents", document_count);
    
    // Create temporary directories
    let temp_dir = TempDir::new()?;
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");
    
    std::fs::create_dir_all(&input_dir)?;
    std::fs::create_dir_all(&output_dir)?;
    
    // Generate test documents
    info!("Generating {} test documents", document_count);
    for i in 0..document_count {
        let content = generate_test_document_content(i);
        let file_path = input_dir.join(format!("test_document_{:03}.md", i + 1));
        let mut file = std::fs::File::create(file_path)?;
        file.write_all(content.as_bytes())?;
    }
    
    // Run benchmark
    let config = ProcessingConfig::default_m3_max(16, 60);
    let processor = DocumentProcessor::new(config).await?;
    
    let benchmark_start = std::time::Instant::now();
    let stats = processor.process_directory(&input_dir, &output_dir).await?;
    let benchmark_time = benchmark_start.elapsed();
    
    // Display benchmark results
    println!("=== Benchmark Results ===");
    println!("Test documents: {}", document_count);
    println!("Documents processed: {}", stats.documents_processed);
    println!("Total time: {:?}", benchmark_time);
    println!("Throughput: {:.2} docs/hour", stats.throughput_per_hour());
    println!("Average quality: {:.3}", stats.average_quality);
    println!("Peak memory: {:.2}GB", stats.memory_peak_gb);
    
    // Performance rating
    let throughput = stats.throughput_per_hour();
    let rating = if throughput >= 30.0 {
        "Excellent"
    } else if throughput >= 25.0 {
        "Good"  
    } else if throughput >= 20.0 {
        "Acceptable"
    } else {
        "Below target"
    };
    
    println!("Performance rating: {} ({:.2} docs/hour)", rating, throughput);
    
    Ok(())
}

/// Generate test document content for benchmarking
fn generate_test_document_content(index: usize) -> String {
    format!(r#"
DOCTITLE: Test Feature {}

Product: CXC4012011, CXC4012019
Feature State: Available

## Description

This is a test feature document for benchmarking the high-performance Rust core.
The feature demonstrates advanced capabilities and parameter configuration options
that are typical of Ericsson RAN feature documentation.

## Parameters

- **testParameter{}**: Boolean parameter controlling test functionality
  - MO Class: TestClass{}  
  - Valid Values: false (disabled), true (enabled)
  - Default: false
  - Description: Controls the behavior of test feature {}

- **thresholdValue{}**: Integer parameter for threshold configuration
  - MO Class: TestClass{}
  - Valid Values: 0-100
  - Default: 50
  - Description: Sets the threshold value for feature activation

## Counters

- **testCounter{}**: Count of test events
  - Description: Tracks the number of test events processed
  - MO Class: TestClass{}
  - Counter Type: Incremental

## Configuration

The test feature can be configured using the parameters above.
Proper configuration ensures optimal performance and functionality.
This feature integrates with LTE, 5G, NR, eNodeB, gNodeB, UE, and QoS systems.

## Examples

Configure the feature:
```
testParameter{} = true
thresholdValue{} = 75
```

Monitor performance:
```
testCounter{} >= 1000
```

Technical terms: MIMO, CA, VoLTE, IMS, EPC, RRC, PDCP, RLC, MAC, PHY, TTI
"#, 
    index + 1,
    index + 1, index + 1, index + 1,
    index + 1, index + 1,
    index + 1, index + 1,
    index + 1, index + 1, index + 1,
    index + 1
    )
}