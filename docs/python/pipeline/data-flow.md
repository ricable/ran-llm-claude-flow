# Complete Data Pipeline Flow Diagram

```
RAW INPUT DATA LAYER
├── data/elex/*.zip              [3GPP ZIP files]
├── data/html/                   [HTML documents] 
├── data/pdf/                    [PDF manuals]
├── data/csv/                    [Parameter CSV files]
└── data/txt/                    [Configuration files]

                    ↓ [ZIP EXTRACTION & ORGANIZATION]

PREPROCESSING LAYER (Stage 1: Python)
├── scripts/complete_preprocessing.sh    [Pipeline orchestration]
├── packages/processors/.../docling_converter.py [Multi-format conversion]
└── Output: markdown/[format]/           [Organized markdown by source type]

                    ↓ [DOCUMENT CONVERSION & ENHANCEMENT]

PROCESSING LAYER (Stage 2: LangExtract)
├── packages/processors/.../langextract/
│   ├── core/document_chunker.py        [Smart chunking with table preservation]
│   ├── core/model_selector.py         [Intelligent model routing]
│   └── ollama_langextract.py          [6-category structured extraction]
└── Output: Structured JSON chunks      [Features/parameters/counters/events/procedures/examples]

                    ↓ [STRUCTURED DATA EXTRACTION]

GENERATION LAYER (Stage 3: CMEDIT Integration)
├── packages/processors/.../cmedit/
│   ├── feature_grouped_generator.py    [Feature-aware workflow generation]
│   ├── enhanced_feature_grouped_generator.py [Advanced CMEDIT workflows]
│   └── ultra_diverse_question_patterns.py [Question diversity enforcement]
└── Output: Q&A pairs with workflows    [Conversational format with CMEDIT commands]

                    ↓ [DATASET GENERATION]

FINAL OUTPUT LAYER
├── training_data/
│   ├── *.jsonl                         [Primary LLM training format]
│   ├── *.parquet                       [Analytics and large dataset format]
│   └── *.csv                           [Human-readable analysis format]
├── tables/                             [Extracted tables from documents]
└── multimodal/                         [Images and visual content from PDFs]

                    ↓ [QUALITY ASSURANCE]

QUALITY & OPTIMIZATION LAYER
├── Deduplication                       [Remove duplicate Q&A pairs]
├── Quality scoring                     [Content richness & complexity metrics]
├── Diversity enforcement               [Question pattern variety]
└── Training/validation splits          [Dataset partitioning for ML]
```

## Data Transformation Stages

### Stage 1: Raw Data Ingestion & Organization
**Input:** ZIP files, HTML, PDF, CSV, TXT files
**Process:** File extraction, format detection, directory organization
**Output:** Organized file structure by format type
**Key Component:** `docling_converter.py` with multi-format support

### Stage 2: Document Conversion & Enhancement  
**Input:** Raw document files
**Process:** Docling conversion to markdown, metadata extraction, table/image processing
**Output:** Enhanced markdown with frontmatter metadata
**Key Components:** 
- `EricssonDocConverter` class
- Multimodal processing for PDFs
- Technical metadata extraction

### Stage 3: Content Processing & Chunking
**Input:** Markdown documents
**Process:** Intelligent chunking, complexity analysis, model routing
**Output:** Optimized content chunks for LLM processing
**Key Components:**
- `DocumentChunker` - preserves tables and sections
- `IntelligentModelSelector` - routes by complexity
- Adaptive chunk sizing based on model capabilities

### Stage 4: Structured Data Extraction
**Input:** Document chunks
**Process:** 6-category extraction via Ollama models
**Output:** Structured JSON with features, parameters, counters, events, procedures, examples
**Key Components:**
- Ollama integration with timeout management
- Model selection (gemma3:4b vs qwen3:1.7b)
- Circuit breaker patterns for reliability

### Stage 5: Conversational Dataset Generation
**Input:** Structured data
**Process:** Q&A pair generation with CMEDIT workflow integration
**Output:** Conversation format datasets with diverse question patterns
**Key Components:**
- Feature-grouped generation
- CMEDIT command integration
- Ultra-diverse question pattern enforcement

### Stage 6: Dataset Optimization & Output
**Input:** Raw Q&A pairs
**Process:** Quality scoring, deduplication, format conversion
**Output:** Final training datasets (JSONL/Parquet/CSV)
**Key Features:**
- Multi-format output support
- Quality thresholds and filtering
- Training/validation/test splits

## Processing Pipeline Configuration

### Model Routing Strategy
```
Document Complexity → Model Selection
- Simple (< 10K chars, low technical density) → Fast models (llama3.2:1b, gemma2:2b)
- Moderate (< 50K chars, medium density) → Balanced models (gemma3:4b) 
- Complex (< 100K chars, high density) → Capable models (qwen3:1.7b)
- Very Complex (> 100K chars, very high density) → Premium models (qwen3:7b)
```

### Chunking Strategy
```
Content Type → Chunk Configuration
- Standard documents: 8000 chars max, 800 char overlap
- Table-heavy documents: Preserve table boundaries, larger chunks
- Complex technical content: 12000 chars max for qwen3:1.7b
- Simple content: 10000 chars max for faster processing
```

### Quality Assurance Pipeline
```
Generated Content → Quality Filters
1. Minimum confidence threshold (0.4-0.8)
2. Technical density validation
3. Question diversity enforcement (< 12% same pattern)
4. Answer length validation (180+ characters)
5. Deduplication by semantic similarity
6. Final quality scoring (richness + complexity)
```

## Performance Characteristics

**Throughput Optimization:**
- M3 Max: 8-16 parallel workers
- Adaptive concurrency based on system load
- Memory usage monitoring (114GB limit, 90% of 128GB)
- Circuit breaker patterns for failure resilience

**Quality vs Speed Trade-offs:**
- Fast processing: Basic models, minimal quality checks
- Premium processing: Advanced models, comprehensive quality assurance
- Balanced processing: Optimal model routing with standard quality controls