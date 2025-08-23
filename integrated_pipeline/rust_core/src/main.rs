use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, error};
use tracing_subscriber::{filter::EnvFilter, fmt, prelude::*};

use rust_core::{
    simple_document_processor::{SimpleDocumentProcessor, ProcessorConfig, SimpleDocument},
};
use uuid::Uuid;

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
    /// Start MCP server
    McpServer {
        /// Configuration file path
        #[arg(short, long)]
        config: Option<PathBuf>,
        
        /// WebSocket port
        #[arg(long, default_value = "8000")]
        websocket_port: u16,
        
        /// HTTP port
        #[arg(long, default_value = "8001")]
        http_port: u16,
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
            process_documents_simple(input, output, config, max_concurrent, memory_limit).await?;
        }
        Commands::SystemInfo => {
            show_system_info().await?;
        }
        Commands::Benchmark { count } => {
            run_benchmark_simple(count).await?;
        }
        Commands::McpServer { config: _, websocket_port: _, http_port: _ } => {
            println!("MCP server functionality not available in simple build");
            std::process::exit(1);
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
async fn process_documents_simple(
    input_dir: PathBuf,
    output_dir: PathBuf,
    _config_path: Option<PathBuf>,
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
    
    // Create simplified configuration
    let config = ProcessorConfig {
        max_concurrent_docs: max_concurrent,
        memory_limit_gb: memory_limit,
        output_directory: output_dir.to_string_lossy().to_string(),
        enable_quality_validation: true,
    };
    
    // Create document processor
    info!("Initializing simplified document processor");
    let processor = SimpleDocumentProcessor::new(config);
    
    // Process documents
    info!("Starting document processing pipeline");
    let start_time = std::time::Instant::now();
    
    // For demo, process a sample document
    let sample_doc = SimpleDocument {
        id: Uuid::new_v4(),
        title: "Sample Document".to_string(),
        content: "This is a sample document for processing with the Rust core.".to_string(),
        document_type: "text/plain".to_string(),
    };
    
    let result = processor.process_document(sample_doc).await?;
    let total_time = start_time.elapsed();
    
    // Display results
    info!("=== Processing Complete ===");
    info!("Document ID: {}", result.document_id);
    info!("QA pairs generated: {}", result.qa_pairs.len());
    info!("Quality score: {:.3}", result.quality_score);
    info!("Processing time: {}ms", result.processing_time_ms);
    info!("Total time: {:?}", total_time);
    info!("Success: {}", result.success);
    
    for (i, qa) in result.qa_pairs.iter().enumerate() {
        info!("  QA {}: Q: {} | A: {} | Confidence: {:.3}", 
              i + 1, qa.question, qa.answer, qa.confidence);
    }
    
    info!("Processing completed successfully");
    
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
async fn run_benchmark_simple(document_count: usize) -> Result<()> {
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
    let config = ProcessorConfig::default();
    let processor = SimpleDocumentProcessor::new(config);
    
    let benchmark_start = std::time::Instant::now();
    
    // Create test documents
    let mut test_documents = Vec::new();
    for i in 0..document_count {
        let content = generate_test_document_content(i);
        test_documents.push(SimpleDocument {
            id: Uuid::new_v4(),
            title: format!("Test Document {}", i + 1),
            content,
            document_type: "text/markdown".to_string(),
        });
    }
    
    // Process all documents
    let results = processor.process_batch(test_documents).await?;
    let benchmark_time = benchmark_start.elapsed();
    
    let successful = results.iter().filter(|r| r.success).count();
    let total_qa_pairs: usize = results.iter().map(|r| r.qa_pairs.len()).sum();
    let avg_quality: f64 = results.iter().map(|r| r.quality_score).sum::<f64>() / results.len() as f64;
    let throughput = successful as f64 / benchmark_time.as_secs_f64() * 3600.0; // docs/hour
    
    // Display benchmark results
    println!("=== Benchmark Results ===");
    println!("Test documents: {}", document_count);
    println!("Documents processed: {}", successful);
    println!("Total time: {:?}", benchmark_time);
    println!("Throughput: {:.2} docs/hour", throughput);
    println!("Average quality: {:.3}", avg_quality);
    println!("Total QA pairs: {}", total_qa_pairs);
    
    // Performance rating
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
    let feature_num = index + 1;
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
    feature_num,
    feature_num, feature_num, feature_num,
    feature_num, feature_num,
    feature_num, feature_num,
    feature_num, feature_num, 
    feature_num
    )
}

/// Start MCP server for hybrid pipeline integration
async fn start_mcp_server(
    _config_path: Option<PathBuf>, 
    websocket_port: u16,
    http_port: u16
) -> Result<()> {
    info!("=== MCP Server Not Available in Simplified Mode ===");
    info!("WebSocket port: {}", websocket_port);
    info!("HTTP port: {}", http_port);
    
    error!("MCP Server requires the 'full' feature to be enabled.");
    error!("To use MCP server functionality, rebuild with: cargo build --features full");
    
    anyhow::bail!("MCP Server not available in simplified mode");
}