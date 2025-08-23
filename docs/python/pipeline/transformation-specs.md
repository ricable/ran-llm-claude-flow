# Data Transformation Specifications

## Overview
This document specifies the exact data transformations that occur at each stage of the pipeline, detailing input formats, processing logic, and output schemas.

## Stage 1: Raw Data → Organized Files

### ZIP File Extraction Transformation
```yaml
Input:
  format: application/zip
  structure: 
    - "*.html" files (3GPP documentation)
    - "*.pdf" files (technical manuals) 
    - nested directory structures
    - metadata files

Processing:
  extraction_logic:
    - preserve_directory_structure: true
    - handle_nested_zips: recursive
    - encoding_detection: automatic (chardet)
    - file_filtering: 
        limit_per_zip: 20
        skip_patterns: ["*.tmp", "*.log", "__MACOSX/*"]

Output:
  structure:
    data/html/[original_structure]/file.html
    data/pdf/[original_structure]/file.pdf  
    data/other/[original_structure]/file.*
  metadata:
    extraction_timestamp: ISO8601
    source_zip: filename
    files_extracted: count
    directory_structure: preserved_paths[]
```

### Multi-format File Organization
```yaml
Format_Detection:
  rules:
    - extension_based: primary method
    - mime_type_validation: fallback for ambiguous files
    - content_sniffing: for headerless files
  
  routing_table:
    ".html/.htm": data/html/
    ".pdf": data/pdf/
    ".csv": data/csv/ 
    ".xlsx/.xls": data/excel/
    ".docx/.doc": data/docx/
    ".pptx/.ppt": data/pptx/
    ".json": data/json/
    ".xml": data/xml/
    ".txt": data/txt/
    ".md": data/md/
    "image_formats": data/images/
    "unknown": data/other/
```

## Stage 2: Multi-format → Enhanced Markdown

### Docling Conversion Transformation
```python
# Input Document Processing
class DocumentProcessor:
    def process_document(self, file_path: Path) -> ConversionResult:
        # 1. Format-specific processing
        if file_path.suffix == '.pdf':
            return self.process_pdf(file_path)
        elif file_path.suffix in ['.html', '.htm']:
            return self.process_html(file_path) 
        elif file_path.suffix == '.csv':
            return self.process_csv(file_path)
        # ... additional formats
    
    def process_pdf(self, file_path: Path) -> ConversionResult:
        # OCR and structure extraction
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            generate_page_images=True,
            images_scale=2.0
        )
        return self.converter.convert(file_path, pipeline_options)
```

### Metadata Extraction Schema  
```python
MetadataSchema = {
    "source_file": str,              # Original filename
    "converted_at": datetime,        # Conversion timestamp  
    "format": str,                   # Source format (.pdf, .html, etc.)
    "document_id": Optional[str],    # Extracted Ericsson doc ID
    "version": Optional[str],        # Document version
    "feature_name": Optional[str],   # Extracted feature name
    "cxc_codes": List[str],         # CXC product codes
    
    # Technical content analysis
    "parameters": List[str],         # MO.attribute references
    "counters": List[str],          # Performance counters  
    "enhanced_parameters": List[str], # Extended parameter detection
    "enhanced_counters": List[str],  # Extended counter detection
    
    # Structure analysis
    "has_tables": bool,
    "table_count": int,
    "table_files": List[str],       # Generated table file references
    "has_multimodal": bool,
    "multimodal_pages": int,
    "multimodal_file": Optional[str],
    
    # Content quality metrics
    "word_count": int,
    "technical_term_count": int,
    "unique_technical_terms": int,
    "technical_density": float,     # Technical elements per 1000 words
    "content_richness_score": float, # 0-10 quality score
    "content_complexity_score": float,
    "estimated_reading_time_minutes": int,
    
    # Enhancement metadata
    "has_code_blocks": bool,
    "code_block_count": int,
    "has_feature_relations": bool,
    
    # Organization metadata (if organized output enabled)
    "organization": {
        "category": str,             # lte_features/nr_features/procedures/parameters/counters
        "organized_path": str,
        "organization_timestamp": datetime
    }
}
```

### Content Enhancement Transformation
```python
def enhance_content_structure(content: str) -> str:
    """Multi-stage content enhancement pipeline"""
    
    # Stage 1: HTML artifact removal
    content = remove_html_comments(content)
    content = remove_empty_html_tags(content)  
    content = clean_escaped_characters(content)
    
    # Stage 2: Structure fixes
    content = fix_malformed_headings(content)      # Fix "\\n\\n#\\n\\n"
    content = fix_heading_spacing(content)         # Ensure blank lines around headers
    content = fix_list_formatting(content)         # Proper list spacing
    content = improve_table_formatting(content)    # Clean table structure
    
    # Stage 3: Whitespace normalization
    content = clean_excessive_whitespace(content)  # Limit consecutive newlines
    content = remove_trailing_whitespace(content)  # Clean line endings
    content = normalize_line_endings(content)      # Consistent line breaks
    
    return content
```

### Frontmatter Generation
```yaml
# YAML frontmatter structure
---
source_file: "feature_guide.html"
converted_at: "2024-01-15T10:30:00Z"
format: "html"
document_id: "22104-LZA1234567_1Uen"
version: "BF"
feature_name: "Inter-frequency Load Balancing"
cxc_codes: ["CXC1234567", "CXC2345678"]

technical_metadata:
  parameters: 
    - "EUtranCell.cellId"
    - "ENodeBFunction.tac"
  counters:
    - "EUtranCell.pmActiveUeDlSum"
    - "ENodeB.pmTotalNbrOfUsers"
  word_count: 3847
  technical_density: 12.5
  
quality_scores:
  content_richness_score: 7.2
  content_complexity_score: 8.1
  estimated_reading_time_minutes: 19

structure_analysis:
  has_tables: true
  table_count: 6
  has_code_blocks: true
  section_depth: 4

organization:
  category: "lte_features"
  organized_path: "markdown/organized/features/lte_features/inter_freq_lb.md"
---

# Enhanced markdown content follows...
```

## Stage 3: Markdown → Intelligent Chunks

### Document Complexity Analysis
```python
def analyze_document_complexity(content: str) -> DocumentAnalysis:
    """Comprehensive document analysis for optimal processing"""
    
    # Basic metrics
    length = len(content)
    words = content.lower().split()
    
    # Technical density calculation
    technical_terms = {
        'parameter', 'counter', 'pm', 'feature', 'activate', 'configure',
        'eutran', 'cell', 'node', 'enb', 'lte', '5g', 'nr', 'function',
        'management', 'object', 'mo', 'attribute', 'dn', 'fdn', 'cmedit'
    }
    technical_count = sum(1 for word in words if any(term in word for term in technical_terms))
    technical_density = technical_count / max(len(words), 1)
    
    # Structural complexity  
    table_count = content.count('|') // 5
    parameter_count = len([w for w in words if '.' in w and any(c.isupper() for c in w)])
    section_depth = len([line for line in content.split('\n') if line.strip().startswith('#')])
    
    # Advanced indicators
    has_activation_procedures = 'activate' in content.lower() or 'featurestate' in content.lower()
    has_complex_tables = table_count > 10 or content.count('|') > 100
    has_code_blocks = content.count('```') > 0
    
    # Complexity classification
    if length < 10000 and technical_density < 0.1:
        complexity = DocumentComplexity.SIMPLE
        recommended_tier = ModelTier.FAST
    elif length < 50000 and technical_density < 0.2:
        complexity = DocumentComplexity.MODERATE  
        recommended_tier = ModelTier.BALANCED
    elif length < 100000 and technical_density < 0.4:
        complexity = DocumentComplexity.COMPLEX
        recommended_tier = ModelTier.CAPABLE
    else:
        complexity = DocumentComplexity.VERY_COMPLEX
        recommended_tier = ModelTier.PREMIUM
        
    return DocumentAnalysis(
        length=length,
        complexity=complexity,
        technical_density=technical_density,
        table_count=table_count,
        parameter_count=parameter_count,
        recommended_tier=recommended_tier,
        has_activation_procedures=has_activation_procedures,
        has_complex_tables=has_complex_tables,
        has_code_blocks=has_code_blocks,
        section_depth=section_depth
    )
```

### Chunking Algorithm
```python
def chunk_document(text: str, max_chars: int, overlap: int) -> List[str]:
    """Intelligent document chunking with table preservation"""
    
    # 1. Remove YAML frontmatter
    text = remove_yaml_frontmatter(text)
    
    # 2. Split by headers (preserve structure)
    sections = split_by_headers(text)  # Split on \n#+\s pattern
    
    # 3. Build chunks respecting size limits
    chunks = []
    current_chunk = ""
    
    for section in sections:
        # Check for significant tables
        has_significant_table = ('|' in section and section.count('|') > 10)
        
        if len(current_chunk) + len(section) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Handle oversized sections
            if len(section) > max_chars and not has_significant_table:
                # Split by paragraphs, then sentences if needed
                paragraph_chunks = split_section_by_paragraphs(section, max_chars)
                chunks.extend(paragraph_chunks[:-1])
                current_chunk = paragraph_chunks[-1] if paragraph_chunks else ""
            else:
                # Keep tables intact even if oversized
                current_chunk = section
        else:
            current_chunk += "\n" + section if current_chunk else section
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # 4. Add overlap between chunks
    if len(chunks) > 1 and overlap > 0:
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                prev_chunk = chunks[i-1]
                prev_overlap = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
                overlapped_chunks.append(prev_overlap + "\n\n" + chunk)
        chunks = overlapped_chunks
    
    return chunks
```

### Model Selection Logic
```python
def select_optimal_model(doc_analysis: DocumentAnalysis) -> str:
    """Dynamic model selection based on document characteristics"""
    
    model_profiles = {
        "llama3.2:1b": {
            "tier": ModelTier.FAST,
            "max_context": 4096,
            "avg_speed": 350.0,  # chars/second
            "accuracy": 0.85,
            "memory_mb": 1200
        },
        "gemma3:4b": {
            "tier": ModelTier.BALANCED, 
            "max_context": 8192,
            "avg_speed": 200.0,
            "accuracy": 0.92,
            "memory_mb": 4200
        },
        "qwen3:1.7b": {
            "tier": ModelTier.CAPABLE,
            "max_context": 8192, 
            "avg_speed": 120.0,
            "accuracy": 0.95,
            "memory_mb": 2800
        }
    }
    
    # Selection logic based on complexity
    if doc_analysis.complexity == DocumentComplexity.SIMPLE:
        return "llama3.2:1b"  # Fast processing for simple content
    elif doc_analysis.complexity == DocumentComplexity.MODERATE:
        return "gemma3:4b"    # Balanced for standard content
    elif doc_analysis.complexity == DocumentComplexity.COMPLEX:
        return "qwen3:1.7b"   # Capable model for complex content
    else:  # VERY_COMPLEX
        return "qwen3:7b" if available else "qwen3:1.7b"  # Premium model
```

### Chunk Metadata Schema
```python
ChunkMetadata = {
    "chunk_id": int,                    # Sequential chunk identifier
    "total_chunks": int,                # Total chunks for document
    "chunk_size_chars": int,            # Character count
    "source_document": str,             # Original document filename
    "document_complexity": str,         # SIMPLE/MODERATE/COMPLEX/VERY_COMPLEX
    "recommended_model": str,           # Selected model for processing
    "has_tables": bool,                 # Contains significant tables
    "header_count": int,                # Number of headers in chunk
    "paragraph_count": int,             # Number of paragraphs
    "technical_density": float,         # Technical terms per 1000 words
    "overlap_start": int,               # Characters of overlap from previous chunk
    "overlap_end": int,                 # Characters of overlap to next chunk
    "processing_hints": {
        "preserve_table_structure": bool,
        "complex_technical_content": bool,
        "requires_context_awareness": bool
    }
}
```

## Stage 4: Chunks → Structured Data (LangExtract)

### Six-Category Extraction Schema
```python
ExtractionResult = {
    "features": [
        {
            "name": str,                    # Feature name
            "description": str,             # Feature description
            "cxc_codes": List[str],        # Associated CXC codes
            "activation_procedure": str,    # How to activate
            "dependencies": List[str],      # Required features
            "parameters": List[str],        # Associated parameters
            "counters": List[str],          # Associated counters
            "confidence": float            # Extraction confidence 0-1
        }
    ],
    
    "parameters": [
        {
            "name": str,                    # Parameter name (MO.attribute format)
            "mo_class": str,               # Managed Object class
            "attribute": str,              # Attribute name
            "description": str,            # Parameter description
            "data_type": str,              # int/string/boolean/enum
            "range_or_values": str,        # Valid range or enum values
            "default_value": str,          # Default value if specified
            "unit": Optional[str],         # Measurement unit
            "related_counters": List[str], # Associated PM counters
            "configuration_impact": str,   # Impact description
            "confidence": float
        }
    ],
    
    "counters": [
        {
            "name": str,                    # Counter name (pmXxxYyy format)
            "description": str,            # Counter description
            "measurement_type": str,        # gauge/counter/distribution
            "unit": str,                   # Measurement unit
            "collection_method": str,      # How it's collected
            "related_parameters": List[str], # Associated parameters
            "threshold_guidance": str,     # Threshold recommendations
            "troubleshooting_use": str,    # How to use for troubleshooting
            "confidence": float
        }
    ],
    
    "events": [
        {
            "name": str,                    # Event name
            "description": str,            # Event description
            "severity": str,               # Critical/Major/Minor/Warning
            "trigger_conditions": str,     # What causes this event
            "impact": str,                 # System impact
            "resolution_steps": str,       # How to resolve
            "related_counters": List[str], # Counters to monitor
            "related_parameters": List[str], # Parameters to check
            "confidence": float
        }
    ],
    
    "procedures": [
        {
            "name": str,                    # Procedure name
            "description": str,            # Procedure description
            "type": str,                   # activation/configuration/troubleshooting/optimization
            "prerequisites": List[str],    # Required conditions
            "steps": List[str],            # Step-by-step instructions
            "verification": List[str],     # How to verify completion
            "rollback": Optional[str],     # How to undo if needed
            "related_features": List[str], # Associated features
            "cmedit_commands": List[str],  # Required CMEDIT commands
            "confidence": float
        }
    ],
    
    "examples": [
        {
            "title": str,                   # Example title
            "description": str,            # Example description
            "type": str,                   # cmedit/xml/cli/configuration
            "code": str,                   # Example code/configuration
            "context": str,                # When to use this example
            "expected_output": Optional[str], # Expected results
            "notes": Optional[str],        # Additional notes
            "related_procedures": List[str], # Associated procedures
            "confidence": float
        }
    ],
    
    # Extraction metadata
    "extraction_metadata": {
        "model_used": str,              # Which model performed extraction
        "processing_time_seconds": float,
        "chunk_id": int,
        "source_document": str,
        "extraction_timestamp": datetime,
        "total_items_extracted": int,
        "confidence_score": float,      # Overall extraction confidence
        "validation_passed": bool,      # JSON structure validation
        "retry_count": int             # Number of retries needed
    }
}
```

### Ollama Request/Response Transformation
```python
def generate_extraction_prompt(chunk_content: str, category: str) -> str:
    """Generate category-specific extraction prompt"""
    
    base_prompt = f"""
    Extract {category} information from the following Ericsson RAN technical document chunk.
    
    Document Content:
    {chunk_content}
    
    Please extract all {category} following the JSON schema provided.
    Focus on technical accuracy and completeness.
    """
    
    category_specific_prompts = {
        "features": base_prompt + """
        For each feature, include:
        - Exact feature name as stated in document
        - Clear description of functionality
        - CXC codes if mentioned
        - Activation procedures
        - Dependencies and related features
        """,
        
        "parameters": base_prompt + """
        For each parameter, include:
        - Full MO.attribute name (e.g., EUtranCell.cellId)
        - Data type and valid range
        - Description of purpose and impact
        - Default values if specified
        """,
        
        # ... other category-specific instructions
    }
    
    return category_specific_prompts[category]

def validate_extraction_response(response: str, category: str) -> Tuple[bool, Dict]:
    """Validate and parse extraction response"""
    try:
        # Parse JSON response
        data = json.loads(response)
        
        # Category-specific validation
        if category not in data:
            return False, {"error": f"Missing {category} key in response"}
        
        # Validate structure for each item
        for item in data[category]:
            if not validate_item_schema(item, category):
                return False, {"error": f"Invalid schema for {category} item"}
        
        return True, data
        
    except json.JSONDecodeError as e:
        return False, {"error": f"JSON parse error: {str(e)}"}
```

### Error Handling & Retry Logic
```python
async def extract_with_retry(chunk: str, category: str, model: str) -> Dict:
    """Extract structured data with comprehensive retry logic"""
    
    for attempt in range(MAX_RETRIES):
        try:
            # Generate prompt
            prompt = generate_extraction_prompt(chunk, category)
            
            # Make Ollama request with timeout
            response = await ollama_client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0.7,
                    "num_predict": 8192,
                    "timeout": TIMEOUT_SECONDS
                }
            )
            
            # Validate response
            is_valid, result = validate_extraction_response(response.response, category)
            
            if is_valid:
                # Record successful extraction
                record_extraction_success(model, category, time.time() - start_time)
                return result
            else:
                logger.warning(f"Invalid response on attempt {attempt + 1}: {result['error']}")
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout on attempt {attempt + 1} for {category}")
            record_timeout(model, category)
            
        except Exception as e:
            logger.error(f"Extraction error on attempt {attempt + 1}: {str(e)}")
            record_error(model, category, str(e))
    
    # All retries failed
    logger.error(f"Failed to extract {category} after {MAX_RETRIES} attempts")
    return {"error": "Max retries exceeded", "category": category}
```

## Stage 5: Structured Data → Conversational Dataset

### Q&A Generation Transformation
```python
def generate_qa_pairs(structured_data: Dict, source_context: str) -> List[Dict]:
    """Generate diverse Q&A pairs from structured data"""
    
    qa_pairs = []
    
    # Process each category
    for category, items in structured_data.items():
        if category == "extraction_metadata":
            continue
            
        for item in items:
            # Generate multiple Q&A pairs per item
            pairs = generate_category_specific_qa(category, item, source_context)
            qa_pairs.extend(pairs)
    
    # Apply diversity enforcement
    qa_pairs = enforce_question_diversity(qa_pairs)
    
    # Apply quality filtering
    qa_pairs = filter_by_quality(qa_pairs)
    
    return qa_pairs

def generate_category_specific_qa(category: str, item: Dict, context: str) -> List[Dict]:
    """Generate category-specific question-answer pairs"""
    
    qa_generators = {
        "features": generate_feature_qa,
        "parameters": generate_parameter_qa,
        "counters": generate_counter_qa,
        "events": generate_event_qa,
        "procedures": generate_procedure_qa,
        "examples": generate_example_qa
    }
    
    return qa_generators[category](item, context)
```

### Question Pattern Templates
```python
QUESTION_PATTERNS = {
    "features": {
        "factual": [
            "What is the {feature_name} feature?",
            "What does {feature_name} do?",
            "What is the purpose of {feature_name}?"
        ],
        "technical": [
            "What are the technical specifications of {feature_name}?",
            "How does {feature_name} work technically?",
            "What is the CXC code for {feature_name}?"
        ],
        "procedural": [
            "How do you activate {feature_name}?",
            "What are the steps to configure {feature_name}?",
            "How do you implement {feature_name}?"
        ],
        "integration": [
            "What features does {feature_name} depend on?",
            "How does {feature_name} interact with other features?",
            "What are the prerequisites for {feature_name}?"
        ]
    },
    
    "parameters": {
        "factual": [
            "What is the {parameter_name} parameter?",
            "What does {parameter_name} control?",
            "What is the purpose of {parameter_name}?"
        ],
        "technical": [
            "What is the data type of {parameter_name}?",
            "What is the valid range for {parameter_name}?",
            "What is the default value of {parameter_name}?"
        ],
        "configuration": [
            "How do you set {parameter_name}?",
            "How do you configure {parameter_name}?",
            "What is the CMEDIT command to modify {parameter_name}?"
        ],
        "impact": [
            "What happens when you change {parameter_name}?",
            "How does {parameter_name} affect system performance?",
            "What is the impact of modifying {parameter_name}?"
        ]
    }
    
    # ... patterns for other categories
}
```

### CMEDIT Workflow Integration
```python
def integrate_cmedit_workflow(item: Dict, category: str) -> str:
    """Integrate CMEDIT commands into answer based on category"""
    
    cmedit_templates = {
        "features": """
        To work with this feature:
        
        1. **Check current status:**
           ```
           describe FeatureState
           get FeatureState.featureStateId {feature_name}
           ```
        
        2. **Activate the feature:**
           ```
           set FeatureState.featureStateId {feature_name} ACTIVATED
           ```
        
        3. **Verify activation:**
           ```
           get FeatureState.featureStateId {feature_name}
           ```
        """,
        
        "parameters": """
        To configure this parameter:
        
        1. **Check current value:**
           ```
           get {mo_class}.{attribute}
           ```
        
        2. **Set new value:**
           ```
           set {mo_class}.{attribute} {new_value}
           ```
        
        3. **Verify the change:**
           ```
           get {mo_class}.{attribute}
           ```
        """,
        
        "procedures": """
        Complete CMEDIT workflow:
        
        {step_by_step_commands}
        
        **Verification commands:**
        {verification_commands}
        """
    }
    
    template = cmedit_templates.get(category, "")
    return template.format(**item) if template else ""
```

### Quality Assessment Algorithm
```python
def assess_qa_quality(question: str, answer: str, source_item: Dict) -> float:
    """Comprehensive Q&A pair quality assessment"""
    
    quality_score = BASE_CONFIDENCE  # Start with base score
    
    # Length bonuses
    if len(answer) >= 200:
        quality_score += LENGTH_BONUS_2
    elif len(answer) >= 50:
        quality_score += LENGTH_BONUS_1
    
    # Technical content analysis
    technical_terms = count_technical_terms(answer)
    if technical_terms > 0:
        quality_score += min(technical_terms * TECHNICAL_TERM_MULTIPLIER, MAX_TECHNICAL_BONUS)
    
    # Question-answer coherence
    word_overlap = calculate_word_overlap(question, answer)
    quality_score += min(word_overlap * WORD_OVERLAP_MULTIPLIER, MAX_WORD_OVERLAP_BONUS)
    
    # CMEDIT integration bonus
    if contains_cmedit_commands(answer):
        quality_score += CMEDIT_INTEGRATION_BONUS
    
    # Apply penalties
    if len(answer) < MIN_ANSWER_LENGTH:
        quality_score -= ANSWER_LENGTH_PENALTY
        
    if contains_placeholders(answer):
        quality_score -= PLACEHOLDER_PENALTY
        
    if not has_sufficient_technical_density(answer):
        quality_score -= LOW_TECHNICAL_PENALTY
    
    return max(0.0, min(1.0, quality_score))
```

### Diversity Enforcement Algorithm
```python
def enforce_question_diversity(qa_pairs: List[Dict]) -> List[Dict]:
    """Enforce diversity across question patterns"""
    
    # Track pattern usage
    first_word_counts = defaultdict(int)
    pattern_counts = defaultdict(int)
    total_questions = len(qa_pairs)
    
    diverse_pairs = []
    regeneration_queue = []
    
    for pair in qa_pairs:
        question = pair['question']
        first_word = question.split()[0].lower()
        
        # Check first word percentage
        first_word_percentage = (first_word_counts[first_word] / total_questions) * 100
        
        if first_word_percentage < MAX_FIRST_WORD_PERCENTAGE:
            # Acceptable diversity
            first_word_counts[first_word] += 1
            diverse_pairs.append(pair)
        else:
            # Queue for regeneration
            regeneration_queue.append(pair)
    
    # Regenerate questions that violate diversity
    for pair in regeneration_queue:
        for attempt in range(MAX_REGENERATION_ATTEMPTS):
            regenerated = regenerate_question(pair, avoid_patterns=get_overused_patterns(first_word_counts))
            if passes_diversity_check(regenerated, first_word_counts):
                diverse_pairs.append(regenerated)
                break
    
    return diverse_pairs
```

## Stage 6: Conversational Data → Final Datasets

### Output Format Transformations

#### JSONL Format (Primary Training Format)
```python
def generate_jsonl_output(qa_pairs: List[Dict], metadata: Dict) -> str:
    """Generate JSON Lines format for LLM training"""
    
    jsonl_lines = []
    
    for pair in qa_pairs:
        # Conversation format for training
        conversation_entry = {
            "conversations": [
                {
                    "from": "human",
                    "value": pair["question"]
                },
                {
                    "from": "gpt", 
                    "value": pair["answer"]
                }
            ],
            
            # Training metadata
            "id": pair["id"],
            "source": pair["source_document"],
            "category": pair["category"],
            "quality_score": pair["quality_score"],
            "confidence": pair["confidence"],
            
            # Technical metadata
            "technical_density": pair.get("technical_density", 0.0),
            "contains_cmedit": pair.get("contains_cmedit", False),
            "extraction_category": pair.get("extraction_category"),
            
            # Generation metadata
            "model_used": pair.get("model_used"),
            "generation_timestamp": pair.get("timestamp"),
            "diversity_score": pair.get("diversity_score", 0.0)
        }
        
        jsonl_lines.append(json.dumps(conversation_entry))
    
    return "\n".join(jsonl_lines)
```

#### Parquet Format (Analytics)
```python
def generate_parquet_output(qa_pairs: List[Dict]) -> pd.DataFrame:
    """Generate Parquet format for analytics and large datasets"""
    
    records = []
    
    for pair in qa_pairs:
        record = {
            # Core content
            "question": pair["question"],
            "answer": pair["answer"],
            "id": pair["id"],
            
            # Quality metrics
            "quality_score": pair["quality_score"],
            "confidence": pair["confidence"],
            "diversity_score": pair.get("diversity_score", 0.0),
            
            # Content analysis
            "question_length": len(pair["question"]),
            "answer_length": len(pair["answer"]),
            "technical_term_count": count_technical_terms(pair["answer"]),
            "contains_cmedit": pair.get("contains_cmedit", False),
            "contains_code_blocks": "```" in pair["answer"],
            
            # Source tracking
            "source_document": pair["source_document"],
            "source_category": pair["category"],
            "extraction_category": pair.get("extraction_category"),
            
            # Processing metadata
            "model_used": pair.get("model_used"),
            "processing_timestamp": pair.get("timestamp"),
            "retry_count": pair.get("retry_count", 0)
        }
        
        records.append(record)
    
    return pd.DataFrame(records)
```

#### CSV Format (Human-Readable Analysis)
```python
def generate_csv_output(qa_pairs: List[Dict]) -> pd.DataFrame:
    """Generate CSV format for human analysis"""
    
    simplified_records = []
    
    for pair in qa_pairs:
        record = {
            "question": pair["question"],
            "answer": pair["answer"][:500] + "..." if len(pair["answer"]) > 500 else pair["answer"],
            "quality_score": round(pair["quality_score"], 2),
            "category": pair["category"],
            "source": os.path.basename(pair["source_document"]),
            "technical_terms": count_technical_terms(pair["answer"]),
            "contains_commands": "Yes" if pair.get("contains_cmedit", False) else "No",
            "confidence": round(pair["confidence"], 2)
        }
        
        simplified_records.append(record)
    
    return pd.DataFrame(simplified_records)
```

### Dataset Statistics & Validation
```python
def generate_dataset_statistics(qa_pairs: List[Dict]) -> Dict:
    """Generate comprehensive dataset statistics"""
    
    stats = {
        "overview": {
            "total_qa_pairs": len(qa_pairs),
            "generation_timestamp": datetime.now().isoformat(),
            "average_quality_score": statistics.mean([p["quality_score"] for p in qa_pairs]),
            "quality_score_distribution": get_score_distribution(qa_pairs, "quality_score")
        },
        
        "content_analysis": {
            "average_question_length": statistics.mean([len(p["question"]) for p in qa_pairs]),
            "average_answer_length": statistics.mean([len(p["answer"]) for p in qa_pairs]),
            "technical_density_avg": statistics.mean([p.get("technical_density", 0) for p in qa_pairs]),
            "cmedit_integration_rate": sum([1 for p in qa_pairs if p.get("contains_cmedit", False)]) / len(qa_pairs)
        },
        
        "diversity_metrics": {
            "unique_first_words": len(set([p["question"].split()[0].lower() for p in qa_pairs])),
            "question_pattern_diversity": calculate_pattern_diversity(qa_pairs),
            "category_distribution": get_category_distribution(qa_pairs)
        },
        
        "quality_assurance": {
            "high_quality_pairs": sum([1 for p in qa_pairs if p["quality_score"] >= 0.8]),
            "low_quality_pairs": sum([1 for p in qa_pairs if p["quality_score"] < 0.5]),
            "deduplication_removed": len(qa_pairs) - len(remove_duplicates(qa_pairs)),
            "validation_passed": all([validate_qa_pair(p) for p in qa_pairs])
        }
    }
    
    return stats
```

### Final Validation Pipeline
```python
def validate_final_dataset(dataset_path: Path) -> Dict[str, Any]:
    """Comprehensive validation of final dataset"""
    
    validation_results = {
        "file_validation": {
            "file_exists": dataset_path.exists(),
            "file_size_mb": dataset_path.stat().st_size / (1024 * 1024) if dataset_path.exists() else 0,
            "format_valid": validate_file_format(dataset_path)
        },
        
        "content_validation": {
            "json_parseable": True,
            "schema_compliant": True,
            "no_empty_content": True,
            "encoding_valid": True
        },
        
        "quality_validation": {
            "min_quality_threshold_met": True,
            "diversity_requirements_met": True,
            "technical_content_present": True,
            "cmedit_integration_adequate": True
        }
    }
    
    try:
        # Load and validate dataset
        if dataset_path.suffix == '.jsonl':
            validate_jsonl_content(dataset_path, validation_results)
        elif dataset_path.suffix == '.parquet':
            validate_parquet_content(dataset_path, validation_results)
        elif dataset_path.suffix == '.csv':
            validate_csv_content(dataset_path, validation_results)
            
    except Exception as e:
        validation_results["validation_error"] = str(e)
        validation_results["overall_valid"] = False
    
    return validation_results
```

This comprehensive transformation specification provides the exact details of how data flows through each stage of the pipeline, with precise schemas, algorithms, and validation criteria for each transformation step.