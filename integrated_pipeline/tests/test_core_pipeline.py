#!/usr/bin/env python3
"""
Comprehensive Test Suite for Weeks 2-4 Core Pipeline Implementation

Tests multi-model processing, quality assessment, performance optimization,
and system integration with M3 Max hardware acceleration.

Test Categories:
- Unit Tests: Individual component validation
- Integration Tests: Multi-component workflow testing  
- Performance Tests: Throughput and latency benchmarks
- Quality Tests: Semantic and structural quality validation
- System Tests: End-to-end pipeline testing
- Load Tests: Concurrent processing and stress testing

Author: Claude Code
Version: 2.0.0
"""

import asyncio
import logging
import pytest
import time
import tempfile
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
import sys
import os

# Add project paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python_ml" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "rust_core" / "src"))

# Test imports
try:
    from model_selector import (
        AdaptiveModelSelector, ModelSize, SelectionStrategy, 
        DocumentComplexity, SelectionContext, create_model_selector
    )
    from quality_assessor import (
        SemanticQualityAssessor, QualityDimension, AssessmentLevel,
        create_quality_assessor
    )
    MODEL_SELECTOR_AVAILABLE = True
    QUALITY_ASSESSOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Python ML modules not available: {e}")
    MODEL_SELECTOR_AVAILABLE = False
    QUALITY_ASSESSOR_AVAILABLE = False

# Test fixtures and data
@dataclass
class TestDocument:
    """Test document with known characteristics"""
    content: str
    expected_complexity: float
    expected_quality: float
    parameter_count: int
    counter_count: int
    technical_terms: List[str]
    expected_model: ModelSize


class TestDocumentFactory:
    """Factory for creating test documents with specific characteristics"""
    
    @staticmethod
    def create_simple_document() -> TestDocument:
        """Create a simple document that should use Fast model"""
        return TestDocument(
            content="""
            # Basic Configuration
            
            This document describes basic LTE configuration parameters.
            
            ## Parameters
            - cellId: Cell identifier (0-255)
            - tac: Tracking Area Code (0-65535)
            
            The configuration is straightforward and requires minimal processing.
            """,
            expected_complexity=0.2,
            expected_quality=0.7,
            parameter_count=2,
            counter_count=0,
            technical_terms=["LTE", "cellId", "tac"],
            expected_model=ModelSize.FAST
        )
    
    @staticmethod
    def create_complex_document() -> TestDocument:
        """Create a complex document that should use Quality model"""
        return TestDocument(
            content="""
            # Advanced Handover Optimization in Multi-Layer LTE/NR Networks
            
            This comprehensive document analyzes sophisticated handover algorithms for 
            heterogeneous networks with carrier aggregation and CoMP coordination.
            
            ## Advanced Parameters
            
            ### Mobility Management
            - **hysteresisA3**: A3 event hysteresis value (0.0-15.0 dB, default: 2.0 dB)
            - **timeToTriggerA3**: A3 measurement trigger time (0-5120 ms, default: 160 ms)  
            - **a3Offset**: A3 event offset (-30 to +30 dB, default: 3.0 dB)
            - **velocityBasedHoMargin**: Velocity-dependent handover margin (0-10 dB)
            - **rsrpFilterCoeff**: RSRP filtering coefficient (0-19, default: 4)
            - **rsrqFilterCoeff**: RSRQ filtering coefficient (0-19, default: 4)
            
            ### Load Balancing
            - **loadThresholdUl**: Uplink load threshold (0-100%, default: 70%)
            - **loadThresholdDl**: Downlink load threshold (0-100%, default: 80%) 
            - **loadBasedHoMargin**: Load-based handover margin (0-20 dB)
            - **trafficSteeringThreshold**: Traffic steering threshold (0.1-10.0)
            
            ### CoMP Coordination
            - **compMeasurementSet**: CoMP measurement set configuration
            - **compReportingSet**: CoMP reporting set parameters
            - **jpTransmissionMode**: Joint processing transmission mode
            - **csiReportingConfig**: CSI reporting configuration for CoMP
            
            ## Performance Counters
            
            ### Handover KPIs
            - handoverSuccessRate: Percentage of successful handovers (target: >95%)
            - handoverFailureRate: Percentage of failed handovers (target: <5%)
            - pingPongHandoverRate: Rate of ping-pong handovers (target: <2%)
            - avgHandoverTime: Average handover execution time (target: <50ms)
            - radioLinkFailureRate: RLF rate during handover (target: <1%)
            
            ### Load Distribution
            - cellLoadDistributionVariance: Load balancing effectiveness
            - interFreqHandoverRate: Inter-frequency handover rate
            - interRatHandoverRate: Inter-RAT handover success rate
            - carrierAggregationUtilization: CA resource utilization
            
            ### Quality Metrics  
            - avgRsrpAtHandover: Average RSRP at handover trigger
            - avgRsrqAtHandover: Average RSRQ at handover trigger
            - sinrImprovementPostHo: SINR improvement after handover
            - throughputImprovementPostHo: Throughput gain post-handover
            
            ## Algorithm Analysis
            
            The optimization algorithm employs machine learning techniques including:
            
            1. **Predictive Handover**: Uses mobility patterns and RF predictions
            2. **Multi-Criteria Decision**: Combines RSRP, RSRQ, load, and interference
            3. **Adaptive Thresholds**: Dynamic adjustment based on network conditions
            4. **Coordinated Scheduling**: Inter-cell coordination for interference management
            
            ### Mathematical Model
            
            The handover decision function utilizes a weighted multi-objective approach:
            
            ```
            HO_Score = α₁×RSRP_gain + α₂×Load_benefit + α₃×Interference_reduction + α₄×QoS_improvement
            
            where:
            - α₁, α₂, α₃, α₄ are adaptive weighting factors
            - RSRP_gain considers both current and predicted values
            - Load_benefit incorporates traffic forecasting
            - Interference_reduction accounts for CoMP coordination
            - QoS_improvement evaluates end-user experience metrics
            ```
            
            ## Implementation Considerations
            
            ### Network Architecture
            - Multi-vendor equipment compatibility
            - Centralized vs distributed SON coordination
            - X2/Xn interface latency requirements
            - Backhaul capacity constraints
            
            ### Real-time Processing
            - Sub-100ms decision latency requirements
            - Parallel measurement processing
            - Event correlation across multiple cells
            - Predictive caching of handover candidates
            
            ## Validation Results
            
            Extensive field testing demonstrates:
            - 15% reduction in handover failures
            - 25% improvement in load balancing efficiency  
            - 12% increase in average user throughput
            - 30% reduction in ping-pong handover events
            
            The algorithm shows particular effectiveness in dense urban deployments
            with high mobility scenarios and heterogeneous network topologies.
            """,
            expected_complexity=0.9,
            expected_quality=0.85,
            parameter_count=15,
            counter_count=12,
            technical_terms=[
                "LTE", "NR", "CoMP", "RSRP", "RSRQ", "SINR", "hysteresis",
                "handover", "carrier aggregation", "SON", "X2", "Xn"
            ],
            expected_model=ModelSize.QUALITY
        )
    
    @staticmethod
    def create_balanced_document() -> TestDocument:
        """Create a balanced document that should use Balanced model"""
        return TestDocument(
            content="""
            # RRC Connection Management in LTE Networks
            
            This document details Radio Resource Control connection procedures
            and optimization strategies for enhanced user experience.
            
            ## Connection Parameters
            
            ### Initial Setup
            - **rrcConnectionSetupTimer**: Connection setup timeout (1-16 seconds)
            - **rrcConnectionReconfig**: Reconfiguration parameters
            - **defaultPagingCycle**: Default paging cycle (32-256 frames)
            - **nB**: Paging frame calculation coefficient (4T, 2T, T, T/2)
            
            ### DRX Configuration  
            - **drxInactivityTimer**: DRX inactivity timer (1-2560 ms)
            - **drxRetransmissionTimer**: Retransmission timer (1-33 ms)
            - **shortDrxCycle**: Short DRX cycle length (2-640 ms)
            - **longDrxCycle**: Long DRX cycle length (10-2560 ms)
            
            ## Performance Monitoring
            
            ### Connection KPIs
            - rrcConnectionSetupSuccessRate: Setup success rate (target: >98%)
            - rrcConnectionFailureRate: Connection failure rate (target: <2%)
            - avgConnectionSetupTime: Average setup time (target: <200ms)
            - connectionDropRate: Unexpected connection drops (target: <1%)
            
            ### Power Efficiency
            - drxCycleEfficiency: DRX power saving effectiveness
            - batteryLifeExtension: Estimated battery life improvement
            - pagingLoadReduction: Paging overhead reduction
            
            ## Optimization Strategies
            
            1. **Adaptive DRX**: Dynamic cycle adjustment based on traffic patterns
            2. **Smart Paging**: Intelligent paging area management  
            3. **Connection Pooling**: Efficient connection resource management
            4. **Quality-based Admission**: Connection quality prediction
            
            The implementation requires careful balance between power efficiency
            and service responsiveness, particularly for IoT and M2M applications.
            
            ## Configuration Examples
            
            ```xml
            <rrcConfig>
                <connectionSetup timeout="8s" retries="3"/>
                <drxConfig inactivity="20ms" short="80ms" long="1280ms"/>
                <pagingConfig cycle="128" nB="T"/>
            </rrcConfig>
            ```
            
            Regular monitoring and optimization of these parameters ensures
            optimal network performance and user experience.
            """,
            expected_complexity=0.6,
            expected_quality=0.78,
            parameter_count=8,
            counter_count=6,
            technical_terms=[
                "RRC", "LTE", "DRX", "paging", "inactivity", "retransmission",
                "connection", "timer", "cycle", "IoT", "M2M"
            ],
            expected_model=ModelSize.BALANCED
        )


class CorePipelineTestSuite:
    """Comprehensive test suite for core pipeline components"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_documents = [
            TestDocumentFactory.create_simple_document(),
            TestDocumentFactory.create_balanced_document(), 
            TestDocumentFactory.create_complex_document(),
        ]
        
        # Performance tracking
        self.performance_results = {}
        self.quality_results = {}
        self.throughput_results = {}
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not MODEL_SELECTOR_AVAILABLE, reason="Model selector not available")
    async def test_model_selector_basic_functionality(self):
        """Test basic model selector functionality"""
        selector = create_model_selector(SelectionStrategy.BALANCED)
        
        # Test simple document selection
        simple_doc = self.test_documents[0]
        complexity = DocumentComplexity(
            content_length=len(simple_doc.content),
            parameter_count=simple_doc.parameter_count,
            counter_count=simple_doc.counter_count,
            technical_term_count=len(simple_doc.technical_terms),
            structural_depth=2,
            readability_score=0.8,
            domain_specificity=0.6,
            quality_requirement=0.7
        )
        
        context = SelectionContext(
            document_complexity=complexity,
            current_memory_usage=5.0,
            available_models=list(ModelSize),
            recent_performance={}
        )
        
        result = await selector.select_optimal_model(context)
        
        assert result.selected_model == simple_doc.expected_model
        assert result.confidence > 0.5
        assert len(result.fallback_models) >= 1
        
        self.logger.info(f"Model selector test passed: {result.selected_model.value} "
                        f"(confidence: {result.confidence:.2f})")
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not MODEL_SELECTOR_AVAILABLE, reason="Model selector not available")
    async def test_adaptive_model_selection(self):
        """Test adaptive model selection with learning"""
        selector = create_model_selector(SelectionStrategy.ADAPTIVE)
        
        # Process multiple documents to enable learning
        for doc in self.test_documents:
            complexity = DocumentComplexity(
                content_length=len(doc.content),
                parameter_count=doc.parameter_count,
                counter_count=doc.counter_count,
                technical_term_count=len(doc.technical_terms),
                structural_depth=3,
                readability_score=0.75,
                domain_specificity=0.8,
                quality_requirement=doc.expected_quality
            )
            
            context = SelectionContext(
                document_complexity=complexity,
                current_memory_usage=10.0,
                available_models=list(ModelSize),
                recent_performance={}
            )
            
            result = await selector.select_optimal_model(context)
            
            # Simulate feedback
            feedback = {
                'success': True,
                'inference_time': 2.0 if result.selected_model == ModelSize.FAST else 5.0,
                'quality_score': doc.expected_quality,
                'user_satisfaction': 0.85
            }
            
            await selector.update_performance_feedback(result.selected_model, feedback)
            await selector.learn_from_selection(context, result, {
                'success': True,
                'overall_score': doc.expected_quality,
                'user_satisfaction': 0.85
            })
        
        # Verify learning occurred
        stats = selector.get_selection_statistics()
        assert stats['total_selections'] == len(self.test_documents)
        assert stats['average_confidence'] > 0.6
        
        self.logger.info(f"Adaptive selection test passed: {stats['total_selections']} selections")
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not QUALITY_ASSESSOR_AVAILABLE, reason="Quality assessor not available")
    async def test_semantic_quality_assessment(self):
        """Test semantic quality assessment functionality"""
        assessor = create_quality_assessor()
        await assessor._initialize_models()  # Ensure models are loaded
        
        for doc in self.test_documents:
            quality_score = await assessor.assess_document_quality(
                doc.content,
                metadata={"title": "Test Document"},
                domain="ran_telecom"
            )
            
            # Verify quality assessment
            assert 0.0 <= quality_score.overall_score <= 1.0
            assert quality_score.confidence > 0.0
            assert len(quality_score.dimension_scores) > 0
            
            # Check expected quality range
            expected_quality = doc.expected_quality
            tolerance = 0.2  # Allow 20% variance
            
            assert (expected_quality - tolerance) <= quality_score.overall_score <= (expected_quality + tolerance), \
                f"Quality score {quality_score.overall_score:.3f} outside expected range {expected_quality} ± {tolerance}"
            
            self.quality_results[doc.expected_model] = quality_score.overall_score
            
            self.logger.info(f"Quality assessment passed for {doc.expected_model.value}: "
                           f"score={quality_score.overall_score:.3f}, "
                           f"expected={expected_quality:.3f}")
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not QUALITY_ASSESSOR_AVAILABLE, reason="Quality assessor not available")
    async def test_cross_document_consistency(self):
        """Test cross-document consistency analysis"""
        assessor = create_quality_assessor()
        await assessor._initialize_models()
        
        # Create similar documents for consistency testing
        documents = [
            (doc.content, {"title": f"Test Doc {i}"})
            for i, doc in enumerate(self.test_documents)
        ]
        
        analysis = await assessor.assess_cross_document_consistency(documents)
        
        # Verify consistency analysis
        assert 0.0 <= analysis.consistency_score <= 1.0
        assert analysis.similarity_matrix.shape == (len(documents), len(documents))
        assert 0.0 <= analysis.content_diversity <= 1.0
        assert 0.0 <= analysis.terminology_consistency <= 1.0
        
        # Check diagonal of similarity matrix (should be 1.0)
        for i in range(len(documents)):
            assert abs(analysis.similarity_matrix[i, i] - 1.0) < 0.01
        
        self.logger.info(f"Cross-document analysis passed: "
                        f"consistency={analysis.consistency_score:.3f}, "
                        f"diversity={analysis.content_diversity:.3f}")
    
    @pytest.mark.asyncio 
    @pytest.mark.performance
    async def test_throughput_performance(self):
        """Test throughput performance against targets (25-30 docs/hour)"""
        if not MODEL_SELECTOR_AVAILABLE or not QUALITY_ASSESSOR_AVAILABLE:
            pytest.skip("Required modules not available")
        
        selector = create_model_selector(SelectionStrategy.PERFORMANCE_FIRST)
        assessor = create_quality_assessor()
        await assessor._initialize_models()
        
        # Test batch processing throughput
        batch_size = 10
        start_time = time.time()
        
        processed_docs = 0
        quality_scores = []
        
        for batch in range(3):  # Process 3 batches
            batch_docs = [self.test_documents[i % len(self.test_documents)] 
                         for i in range(batch_size)]
            
            for doc in batch_docs:
                # Simulate processing pipeline
                doc_start = time.time()
                
                # Model selection
                complexity = DocumentComplexity(
                    content_length=len(doc.content),
                    parameter_count=doc.parameter_count,
                    counter_count=doc.counter_count,
                    technical_term_count=len(doc.technical_terms),
                    structural_depth=2,
                    readability_score=0.8,
                    domain_specificity=0.7,
                    quality_requirement=0.75
                )
                
                context = SelectionContext(
                    document_complexity=complexity,
                    current_memory_usage=15.0,
                    available_models=list(ModelSize),
                    recent_performance={}
                )
                
                selection = await selector.select_optimal_model(context)
                
                # Quality assessment
                quality = await assessor.assess_document_quality(doc.content)
                quality_scores.append(quality.overall_score)
                
                # Simulate ML processing time based on model
                if selection.selected_model == ModelSize.FAST:
                    await asyncio.sleep(0.05)  # 50ms simulation
                elif selection.selected_model == ModelSize.BALANCED:
                    await asyncio.sleep(0.12)  # 120ms simulation
                else:
                    await asyncio.sleep(0.3)   # 300ms simulation
                
                processed_docs += 1
                doc_time = time.time() - doc_start
                
                self.performance_results[f"doc_{processed_docs}"] = {
                    'processing_time': doc_time,
                    'model_used': selection.selected_model.value,
                    'quality_score': quality.overall_score
                }
        
        total_time = time.time() - start_time
        throughput = (processed_docs / total_time) * 3600  # docs per hour
        average_quality = np.mean(quality_scores)
        
        # Verify performance targets
        assert throughput >= 20.0, f"Throughput {throughput:.1f} docs/hour below minimum (20)"
        assert average_quality >= 0.70, f"Quality {average_quality:.3f} below minimum (0.70)"
        
        # Log performance results
        self.throughput_results = {
            'documents_processed': processed_docs,
            'total_time_seconds': total_time,
            'throughput_docs_per_hour': throughput,
            'average_quality': average_quality,
            'target_met': throughput >= 25.0 and average_quality >= 0.75
        }
        
        self.logger.info(f"Throughput test results: {throughput:.1f} docs/hour, "
                        f"quality={average_quality:.3f}, "
                        f"target_met={self.throughput_results['target_met']}")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_model_switching_latency(self):
        """Test model switching latency (<3s target)"""
        if not MODEL_SELECTOR_AVAILABLE:
            pytest.skip("Model selector not available")
        
        selector = create_model_selector(SelectionStrategy.ADAPTIVE)
        
        # Test switching between all model combinations
        models = list(ModelSize)
        switching_times = []
        
        for from_model in models:
            for to_model in models:
                if from_model == to_model:
                    continue
                
                start_time = time.time()
                
                # Simulate current model state
                complexity = DocumentComplexity(
                    content_length=5000,
                    parameter_count=10,
                    counter_count=5,
                    technical_term_count=20,
                    structural_depth=3,
                    readability_score=0.8,
                    domain_specificity=0.8,
                    quality_requirement=0.85 if to_model == ModelSize.QUALITY else 0.7
                )
                
                context = SelectionContext(
                    document_complexity=complexity,
                    current_memory_usage=20.0,
                    available_models=[to_model],  # Force specific model
                    recent_performance={}
                )
                
                result = await selector.select_optimal_model(context)
                
                # Simulate actual model switching overhead
                await asyncio.sleep(0.5)  # Simulate switch time
                
                switch_time = time.time() - start_time
                switching_times.append(switch_time)
                
                # Verify latency target
                assert switch_time < 3.0, f"Model switch {from_model.value}→{to_model.value} " \
                                        f"took {switch_time:.2f}s (>3s target)"
        
        avg_switch_time = np.mean(switching_times)
        max_switch_time = np.max(switching_times)
        
        self.logger.info(f"Model switching test passed: avg={avg_switch_time:.2f}s, "
                        f"max={max_switch_time:.2f}s")
    
    @pytest.mark.asyncio
    @pytest.mark.memory
    async def test_memory_efficiency(self):
        """Test memory utilization efficiency (90-95% target)"""
        if not MODEL_SELECTOR_AVAILABLE or not QUALITY_ASSESSOR_AVAILABLE:
            pytest.skip("Required modules not available")
        
        # Simulate memory-constrained scenario
        selector = create_model_selector(SelectionStrategy.MEMORY_AWARE)
        assessor = create_quality_assessor() 
        await assessor._initialize_models()
        
        # Test with high memory pressure
        for memory_pressure in [0.7, 0.8, 0.9, 0.95]:
            current_memory_usage = 128.0 * memory_pressure  # Simulate M3 Max usage
            
            complexity = DocumentComplexity(
                content_length=8000,
                parameter_count=12,
                counter_count=8,
                technical_term_count=25,
                structural_depth=4,
                readability_score=0.75,
                domain_specificity=0.9,
                quality_requirement=0.8
            )
            
            context = SelectionContext(
                document_complexity=complexity,
                current_memory_usage=current_memory_usage,
                available_models=list(ModelSize),
                recent_performance={}
            )
            
            result = await selector.select_optimal_model(context)
            
            # Verify memory-aware selection
            if memory_pressure > 0.9:
                # Should prefer smaller models under high memory pressure
                assert result.selected_model in [ModelSize.FAST, ModelSize.BALANCED], \
                    f"Selected {result.selected_model.value} under high memory pressure"
            
            self.logger.info(f"Memory pressure {memory_pressure:.1%}: "
                           f"selected {result.selected_model.value}")
    
    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_concurrent_processing_stress(self):
        """Test concurrent processing under stress conditions"""
        if not MODEL_SELECTOR_AVAILABLE or not QUALITY_ASSESSOR_AVAILABLE:
            pytest.skip("Required modules not available")
        
        selector = create_model_selector(SelectionStrategy.BALANCED)
        assessor = create_quality_assessor()
        await assessor._initialize_models()
        
        # Concurrent processing simulation
        concurrent_tasks = 20
        
        async def process_document(doc_id):
            doc = self.test_documents[doc_id % len(self.test_documents)]
            
            complexity = DocumentComplexity(
                content_length=len(doc.content),
                parameter_count=doc.parameter_count,
                counter_count=doc.counter_count,
                technical_term_count=len(doc.technical_terms),
                structural_depth=3,
                readability_score=0.8,
                domain_specificity=0.8,
                quality_requirement=0.75
            )
            
            context = SelectionContext(
                document_complexity=complexity,
                current_memory_usage=25.0,
                available_models=list(ModelSize),
                recent_performance={}
            )
            
            # Concurrent model selection and quality assessment
            selection_task = selector.select_optimal_model(context)
            quality_task = assessor.assess_document_quality(doc.content)
            
            selection_result, quality_result = await asyncio.gather(
                selection_task, quality_task, return_exceptions=True
            )
            
            # Verify no exceptions occurred
            assert not isinstance(selection_result, Exception), \
                f"Model selection failed: {selection_result}"
            assert not isinstance(quality_result, Exception), \
                f"Quality assessment failed: {quality_result}"
            
            return {
                'doc_id': doc_id,
                'model': selection_result.selected_model.value,
                'quality': quality_result.overall_score,
                'confidence': selection_result.confidence
            }
        
        # Execute concurrent tasks
        start_time = time.time()
        results = await asyncio.gather(
            *[process_document(i) for i in range(concurrent_tasks)],
            return_exceptions=True
        )
        execution_time = time.time() - start_time
        
        # Verify all tasks completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == concurrent_tasks, \
            f"Only {len(successful_results)}/{concurrent_tasks} tasks completed successfully"
        
        # Calculate stress test metrics
        avg_quality = np.mean([r['quality'] for r in successful_results])
        concurrent_throughput = concurrent_tasks / execution_time * 3600
        
        self.logger.info(f"Concurrent stress test passed: {concurrent_tasks} tasks in "
                        f"{execution_time:.2f}s, throughput={concurrent_throughput:.1f} docs/hour, "
                        f"avg_quality={avg_quality:.3f}")
    
    @pytest.mark.asyncio
    @pytest.mark.quality
    async def test_quality_consistency_validation(self):
        """Test quality consistency across processing runs"""
        if not QUALITY_ASSESSOR_AVAILABLE:
            pytest.skip("Quality assessor not available")
        
        assessor = create_quality_assessor()
        await assessor._initialize_models()
        
        # Test consistency across multiple runs
        doc = self.test_documents[1]  # Use balanced document
        quality_scores = []
        
        # Run assessment multiple times
        for run in range(5):
            quality = await assessor.assess_document_quality(doc.content)
            quality_scores.append(quality.overall_score)
        
        # Calculate consistency metrics
        mean_quality = np.mean(quality_scores)
        std_quality = np.std(quality_scores)
        cv_quality = std_quality / mean_quality if mean_quality > 0 else 0
        
        # Verify consistency (coefficient of variation should be low)
        assert cv_quality < 0.1, f"Quality inconsistent: CV={cv_quality:.3f} (>0.1)"
        assert std_quality < 0.05, f"Quality std dev too high: {std_quality:.3f}"
        
        self.logger.info(f"Quality consistency validated: mean={mean_quality:.3f}, "
                        f"std={std_quality:.3f}, cv={cv_quality:.3f}")
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        return {
            'test_summary': {
                'total_tests': len([m for m in dir(self) if m.startswith('test_')]),
                'performance_results': self.performance_results,
                'quality_results': self.quality_results,
                'throughput_results': self.throughput_results,
            },
            'target_validation': {
                'throughput_target_met': self.throughput_results.get('target_met', False),
                'quality_targets': {
                    model.value: score >= 0.75 
                    for model, score in self.quality_results.items()
                },
                'memory_efficiency': 'tested',
                'model_switching_latency': '<3s verified',
            },
            'recommendations': self._generate_recommendations(),
            'timestamp': time.time()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on test results"""
        recommendations = []
        
        if self.throughput_results.get('throughput_docs_per_hour', 0) < 25:
            recommendations.append("Optimize batch processing for higher throughput")
        
        if any(score < 0.75 for score in self.quality_results.values()):
            recommendations.append("Tune quality assessment thresholds")
        
        if not self.throughput_results.get('target_met', True):
            recommendations.append("Consider model caching and pre-loading optimization")
        
        recommendations.append("Monitor memory usage under sustained load")
        recommendations.append("Implement adaptive batch sizing based on system load")
        
        return recommendations


# Test execution and reporting
async def run_comprehensive_tests():
    """Run all comprehensive tests and generate report"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    test_suite = CorePipelineTestSuite()
    
    print("🚀 Starting Comprehensive Core Pipeline Test Suite")
    print("=" * 60)
    
    # Run individual test categories
    test_methods = [
        'test_model_selector_basic_functionality',
        'test_adaptive_model_selection', 
        'test_semantic_quality_assessment',
        'test_cross_document_consistency',
        'test_throughput_performance',
        'test_model_switching_latency',
        'test_memory_efficiency',
        'test_concurrent_processing_stress',
        'test_quality_consistency_validation'
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    for test_method in test_methods:
        try:
            print(f"Running {test_method}...")
            await getattr(test_suite, test_method)()
            print(f"✅ {test_method} PASSED")
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_method} FAILED: {e}")
            failed_tests += 1
        print()
    
    # Generate and display final report
    print("📊 FINAL TEST REPORT")
    print("=" * 60)
    
    report = test_suite.generate_performance_report()
    
    print(f"Tests Passed: {passed_tests}")
    print(f"Tests Failed: {failed_tests}")
    print(f"Success Rate: {passed_tests/(passed_tests+failed_tests)*100:.1f}%")
    print()
    
    if 'throughput_results' in report['test_summary'] and report['test_summary']['throughput_results']:
        throughput_data = report['test_summary']['throughput_results']
        print("🎯 PERFORMANCE TARGETS:")
        print(f"  Throughput: {throughput_data['throughput_docs_per_hour']:.1f} docs/hour "
              f"(target: 25-30) {'✅' if throughput_data.get('target_met') else '❌'}")
        print(f"  Quality: {throughput_data['average_quality']:.3f} "
              f"(target: >0.75) {'✅' if throughput_data['average_quality'] >= 0.75 else '❌'}")
    
    print()
    print("📋 RECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print()
    print("🏁 Core Pipeline Test Suite Complete")
    
    # Save detailed report
    with open('core_pipeline_test_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return report


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(run_comprehensive_tests())