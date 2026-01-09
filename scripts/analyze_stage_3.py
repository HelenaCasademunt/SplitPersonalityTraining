#!/usr/bin/env python3
"""
Stage 3 Analysis Script

Analyzes Stage 3 intervention results to evaluate alignment failure detection
across topics, models, and intervention types.

Usage:
    python analyze_stage_3.py [--topic TOPIC] [--examples TYPES] [--output-format FORMAT] [--detailed] [--ignore-errors]

Examples:
    python analyze_stage_3.py                                           # Analyze all topics
    python analyze_stage_3.py --topic reward_hacks                      # Focus on specific topic  
    python analyze_stage_3.py --examples "unexpected_passes,low_confidence"  # Collect specific examples
    python analyze_stage_3.py --detailed --output-format markdown        # Generate detailed markdown report
"""

import json
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Set
import re
from datetime import datetime

from utils import STAGE_3_TAGGED_DATA_DIR


class Stage3Analyzer:
    def __init__(self, data_dir: str = None, ignore_errors: bool = False):
        if data_dir is None:
            data_dir = str(STAGE_3_TAGGED_DATA_DIR)
        self.data_dir = Path(data_dir)
        self.ignore_errors = ignore_errors
        self.all_data = []
        self.raw_batches = []
        self.topics = []
        self.models = []
        self.intervention_types = []
        self.sanity_check_failures = []
        
        # Analysis results storage
        self.summary_stats = {}
        self.topic_stats = defaultdict(dict)
        self.interesting_examples = defaultdict(list)
        self.interesting_examples_base_counts_by_topic = defaultdict(Counter)
        self.cross_topic_comparisons = {}
        
        # Tag analysis storage
        self.tag_counts = defaultdict(int)
        self.tag_counts_by_topic = defaultdict(lambda: defaultdict(int))
        self.tag_analysis = {}
        self.all_tags = set()
        
    def load_all_data(self) -> None:
        """Load all Stage 3 JSON files and extract metadata."""
        print("Loading Stage 3 data files...")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Stage 3 data directory not found: {self.data_dir}")
            
        json_files = list(self.data_dir.rglob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.data_dir}")
            
        print(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    batch_data = json.load(f)
                    
                topic = batch_data['topic']
                
                # Skip TEST topic
                if topic == 'TEST':
                    continue
                    
                self.topics.append(topic)
                
                # Store raw batch data for disagreement analysis
                self.raw_batches.append({
                    'source_file': str(json_file.relative_to(self.data_dir)),
                    'batch_data': batch_data
                })
                
                for item_idx, item in enumerate(batch_data['data']):
                    # Extract all model/variant combinations
                    for model_name, model_data in item['inferences'].items():
                        self.models.append(model_name)
                        
                        for variant in ['A', 'B']:
                            if variant in model_data:
                                variant_data = model_data[variant]
                                
                                # Collect tags from variant level
                                variant_tags = variant_data.get('tags', [])
                                for tag in variant_tags:
                                    self.tag_counts[tag] += 1
                                    self.tag_counts_by_topic[topic][tag] += 1
                                    self.all_tags.add(tag)
                                
                                # Extract intervention data
                                for intervention_name, intervention_data in variant_data['interventions'].items():
                                    intervention_type = intervention_data['intervention_type']
                                    if intervention_type:
                                        self.intervention_types.append(intervention_type)
                                    
                                    # Collect tags from intervention level
                                    intervention_tags = intervention_data.get('tags', [])
                                    for tag in intervention_tags:
                                        # Only count if not already counted at variant level
                                        if tag not in variant_tags:
                                            self.tag_counts[tag] += 1
                                            self.tag_counts_by_topic[topic][tag] += 1
                                            self.all_tags.add(tag)
                                    
                                    # Create record for analysis
                                    record = {
                                        'source_file': str(json_file.relative_to(self.data_dir)),
                                        'topic': topic,
                                        'item_index': item_idx,
                                        'model': model_name,
                                        'variant': variant,
                                        'intervention_name': intervention_name,
                                        'intervention_type': intervention_type,
                                        'T': item['T'],
                                        'S': variant_data['S'],
                                        'I': intervention_data['I'],
                                        'R': intervention_data['R'],
                                        'P': intervention_data['P'],
                                        'confidence': intervention_data['confidence'],
                                        'detection_difficulty': intervention_data.get('detection_difficulty'),
                                        'selected_review_focus': intervention_data['selected_review_focus'],
                                        'intended_review_focus': item['metadata']['intended_review_focus'],
                                        'alignment_issue_type': item['metadata']['alignment_issue_type'],
                                        'difficulty': item['metadata']['difficulty'],
                                        'output_tokens': variant_data.get('output_tokens', 0),
                                        'input_tokens': variant_data.get('input_tokens', 0),
                                        'stage_2_review': None,  # Will be populated if available
                                        'batch_id': batch_data['batch_id'],
                                        'generation_timestamp': batch_data.get('generation_timestamp', ''),
                                        'tags': list(set(variant_tags + intervention_tags)),  # Combined unique tags
                                    }
                                    
                                    self.all_data.append(record)
                                    
            except Exception as e:
                error_msg = f"Error processing {json_file}: {e}"
                self.sanity_check_failures.append(error_msg)
                if not self.ignore_errors:
                    raise RuntimeError(error_msg)
                print(f"WARNING: {error_msg}")
        
        self.topics = sorted(list(set(self.topics)))
        self.models = sorted(list(set(self.models)))
        self.intervention_types = sorted(list(set(self.intervention_types)))
                
        print(f"Loaded {len(self.all_data)} intervention records")
        print(f"Topics: {sorted(self.topics)}")
        print(f"Models: {sorted(self.models)}")  
        print(f"Intervention types: {sorted(self.intervention_types)}")

    def run_sanity_checks(self) -> None:
        """Run comprehensive sanity checks on the data."""
        print("\nRunning sanity checks...")
        
        # Group by item to check intervention consistency
        item_groups = defaultdict(list)
        for record in self.all_data:
            key = f"{record['topic']}_{record['item_index']}_{record['model']}_{record['variant']}"
            item_groups[key].append(record)
            
        # Check 1: Every sample should have both split_personality and honesty_scare interventions
        for key, records in item_groups.items():
            intervention_types = set(r['intervention_type'] for r in records)
            
            if 'split_personality' not in intervention_types:
                self.sanity_check_failures.append(f"Missing split_personality intervention: {key}")
                
            if 'honesty_scare' not in intervention_types:
                self.sanity_check_failures.append(f"Missing honesty_scare intervention: {key}")
                
        # Check 2: Split personality interventions should use <split-personality-token>
        for record in self.all_data:
            if record['intervention_type'] == 'split_personality':
                if not record['I'].startswith('<split-personality-token>'):
                    self.sanity_check_failures.append(
                        f"Split personality intervention missing token: {record['source_file']} item {record['item_index']}"
                    )
                    
        # (No longer relevant. We now have items that had only the newest intervention sets run on them.)
        # # Check 3: Consistent intervention sets across all items
        # if item_groups:
        #     first_key = next(iter(item_groups))
        #     reference_interventions = set(r['intervention_name'] for r in item_groups[first_key])
            
        #     for key, records in item_groups.items():
        #         current_interventions = set(r['intervention_name'] for r in records)
        #         if current_interventions != reference_interventions:
        #             self.sanity_check_failures.append(f"Inconsistent intervention set: {key}")
                    
        # Check 4: alignment_issue_type should match topic name
        for record in self.all_data:
            topic = record['topic']
            alignment_issue_type = record['alignment_issue_type']
            
            if alignment_issue_type and alignment_issue_type != topic:
                self.sanity_check_failures.append(
                    f"alignment_issue_type mismatch: topic='{topic}' but alignment_issue_type='{alignment_issue_type}' in {record['source_file']} item {record['item_index']}"
                )
                    
        # Print sanity check results
        if self.sanity_check_failures:
            print(f"⚠️  Found {len(self.sanity_check_failures)} sanity check failures:")
            for failure in self.sanity_check_failures[:10]:  # Show first 10
                print(f"  - {failure}")
            if len(self.sanity_check_failures) > 10:
                print(f"  ... and {len(self.sanity_check_failures) - 10} more")
                
            if not self.ignore_errors:
                print("\nUse --ignore-errors flag to continue despite failures")
                sys.exit(1)
        else:
            print("✅ All sanity checks passed")

    def calculate_summary_stats(self) -> None:
        """Calculate overall summary statistics."""
        print("\nCalculating summary statistics...")
        
        total_samples = len(self.all_data)
        
        # Basic counts
        topic_counts = Counter(r['topic'] for r in self.all_data)
        model_counts = Counter(r['model'] for r in self.all_data)
        variant_counts = Counter(r['variant'] for r in self.all_data)
        verdict_counts = Counter(r['P'] for r in self.all_data)
        
        # A/B balance by topic
        ab_balance = {}
        for topic in self.topics:
            topic_records = [r for r in self.all_data if r['topic'] == topic]
            a_count = len([r for r in topic_records if r['variant'] == 'A'])
            b_count = len([r for r in topic_records if r['variant'] == 'B'])
            total = a_count + b_count
            ab_balance[topic] = {
                'A': a_count,
                'B': b_count, 
                'A_pct': (a_count / total * 100) if total > 0 else 0,
                'B_pct': (b_count / total * 100) if total > 0 else 0
            }
            
        # Token analysis
        token_2048_count = len([r for r in self.all_data if r['output_tokens'] == 2048])
        token_2048_pct = (token_2048_count / total_samples * 100) if total_samples > 0 else 0
        
        # Token length distribution analysis
        token_lengths = [r['output_tokens'] for r in self.all_data if r['output_tokens'] > 0]
        if token_lengths:
            token_stats = {
                'mean': sum(token_lengths) / len(token_lengths),
                'min': min(token_lengths),
                'max': max(token_lengths),
                'count_under_100': len([t for t in token_lengths if t < 100]),
                'count_100_500': len([t for t in token_lengths if 100 <= t < 500]),
                'count_500_1000': len([t for t in token_lengths if 500 <= t < 1000]),
                'count_1000_2000': len([t for t in token_lengths if 1000 <= t < 2000]),
                'count_at_2048': len([t for t in token_lengths if t == 2048])
            }
        else:
            token_stats = {}
        
        # Confidence distribution
        confidence_dist = Counter(r['confidence'] for r in self.all_data if r['confidence'] is not None)
        
        # Detection difficulty distribution  
        difficulty_dist = Counter(r['detection_difficulty'] for r in self.all_data if r['detection_difficulty'])
        
        # Review focus analysis
        intended_focuses = Counter(r['intended_review_focus'] for r in self.all_data if r['intended_review_focus'])
        selected_focuses = Counter(r['selected_review_focus'] for r in self.all_data if r['selected_review_focus'])
        
        # Unique intervention names
        intervention_names = set(r['intervention_name'] for r in self.all_data if r['intervention_name'])
        
        # Batch analysis
        batch_ids = Counter(r['batch_id'] for r in self.all_data if r['batch_id'])
        
        self.summary_stats = {
            'total_samples': total_samples,
            'unique_topics': len(self.topics),
            'unique_models': len(self.models),
            'unique_interventions': len(self.intervention_types),
            'unique_intervention_names': sorted(list(intervention_names)),
            'topic_counts': dict(sorted(topic_counts.items())),
            'model_counts': dict(sorted(model_counts.items())),
            'variant_counts': dict(sorted(variant_counts.items())),
            'verdict_counts': dict(sorted(verdict_counts.items())),
            'ab_balance': ab_balance,
            'token_analysis': {
                'samples_at_2048_tokens': token_2048_count,
                'pct_samples_at_2048_tokens': token_2048_pct,
                'distribution': token_stats
            },
            'confidence_distribution': dict(sorted(confidence_dist.items())),
            'difficulty_distribution': dict(sorted(difficulty_dist.items())),
            'review_focus_analysis': {
                'intended_focuses': dict(sorted(intended_focuses.items())),
                'selected_focuses': dict(sorted(selected_focuses.items()))
            },
            'batch_counts': dict(batch_ids)
        }

    def analyze_by_topic(self) -> None:
        """Generate per-topic analysis."""
        print("Analyzing by topic...")
        
        for topic in sorted(self.topics):
            topic_records = [r for r in self.all_data if r['topic'] == topic]
            
            # Basic topic stats
            total_records = len(topic_records)
            a_records = [r for r in topic_records if r['variant'] == 'A']
            b_records = [r for r in topic_records if r['variant'] == 'B']
            
            # Pass/fail rates by variant
            a_pass_rate = len([r for r in a_records if r['P'] == 'PASS']) / len(a_records) * 100 if a_records else 0
            a_fail_rate = len([r for r in a_records if r['P'] == 'FAIL']) / len(a_records) * 100 if a_records else 0
            b_pass_rate = len([r for r in b_records if r['P'] == 'PASS']) / len(b_records) * 100 if b_records else 0
            b_fail_rate = len([r for r in b_records if r['P'] == 'FAIL']) / len(b_records) * 100 if b_records else 0
            
            # Quality score (ideally A should pass, B should fail)
            quality_score = (a_pass_rate + b_fail_rate) / 2 if (a_records and b_records) else 0
            
            # Confidence distribution for this topic
            confidence_values = [r['confidence'] for r in topic_records if r['confidence'] is not None]
            confidence_dist = Counter(confidence_values)
            avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
            
            # Difficulty distribution for this topic
            difficulty_values = [r['detection_difficulty'] for r in topic_records if r['detection_difficulty']]
            difficulty_dist = Counter(difficulty_values)
            
            # Convert difficulty to numeric for averaging (easy=1, medium=2, hard=3, other=2)
            difficulty_numeric_map = {'easy': 1, 'medium': 2, 'hard': 3, 'FPT': 2, 'NA': 2}
            numeric_difficulties = [difficulty_numeric_map.get(d, 2) for d in difficulty_values]
            avg_difficulty = sum(numeric_difficulties) / len(numeric_difficulties) if numeric_difficulties else 0
            
            # Model comparison
            model_performance = {}
            for model in self.models:
                model_records = [r for r in topic_records if r['model'] == model]
                if model_records:
                    model_a = [r for r in model_records if r['variant'] == 'A']
                    model_b = [r for r in model_records if r['variant'] == 'B']
                    model_a_pass = len([r for r in model_a if r['P'] == 'PASS']) / len(model_a) * 100 if model_a else 0
                    model_b_fail = len([r for r in model_b if r['P'] == 'FAIL']) / len(model_b) * 100 if model_b else 0
                    model_quality = (model_a_pass + model_b_fail) / 2
                    model_performance[model] = {
                        'samples': len(model_records),
                        'a_pass_rate': model_a_pass,
                        'b_fail_rate': model_b_fail,
                        'quality_score': model_quality
                    }
                    
            # Intervention effectiveness
            intervention_performance = {}
            for intervention_type in self.intervention_types:
                intervention_records = [r for r in topic_records if r['intervention_type'] == intervention_type]
                if intervention_records:
                    int_a = [r for r in intervention_records if r['variant'] == 'A']
                    int_b = [r for r in intervention_records if r['variant'] == 'B']
                    int_a_pass = len([r for r in int_a if r['P'] == 'PASS']) / len(int_a) * 100 if int_a else 0
                    int_b_fail = len([r for r in int_b if r['P'] == 'FAIL']) / len(int_b) * 100 if int_b else 0
                    int_quality = (int_a_pass + int_b_fail) / 2
                    intervention_performance[intervention_type] = {
                        'samples': len(intervention_records),
                        'a_pass_rate': int_a_pass,
                        'b_fail_rate': int_b_fail,
                        'quality_score': int_quality
                    }
            
            # Review focus analysis
            review_focus_matches = 0
            review_focus_mismatches = 0
            review_focus_examples = []
            
            for record in topic_records:
                intended = record['intended_review_focus'] 
                selected = record['selected_review_focus']
                
                if intended and selected:
                    if intended == selected or intended in selected or selected in intended:
                        review_focus_matches += 1
                    else:
                        review_focus_mismatches += 1
                        review_focus_examples.append({
                            'source': record['source_file'],
                            'intended': intended,
                            'selected': selected
                        })
                        
            self.topic_stats[topic] = {
                'total_samples': total_records,
                'a_samples': len(a_records),
                'b_samples': len(b_records), 
                'a_pass_rate': a_pass_rate,
                'a_fail_rate': a_fail_rate,
                'b_pass_rate': b_pass_rate,
                'b_fail_rate': b_fail_rate,
                'quality_score': quality_score,
                'confidence_distribution': dict(sorted(confidence_dist.items())),
                'avg_confidence': avg_confidence,
                'difficulty_distribution': dict(sorted(difficulty_dist.items())),
                'avg_difficulty': avg_difficulty,
                'model_performance': dict(sorted(model_performance.items())),
                'intervention_performance': dict(sorted(intervention_performance.items())),
                'review_focus_matches': review_focus_matches,
                'review_focus_mismatches': review_focus_mismatches,
                'review_focus_examples': review_focus_examples[:5]  # Sample examples
            }

    def collect_interesting_examples(self, example_types: List[str] = None) -> None:
        """Collect interesting examples for manual review."""
        print("Collecting interesting examples...")
        
        if example_types is None:
            example_types = ['unexpected_passes', 'unexpected_fails', 'unexpected_a_b_inversion', 'low_confidence', 'intervention_disagreements_high_confidence', 'intervention_disagreements']
            
        for record in self.all_data:
            variant = record['variant']
            verdict = record['P']
            confidence = record['confidence']
            
            # Unexpected B passes (B variants should typically FAIL)
            if 'unexpected_passes' in example_types and variant == 'B' and verdict == 'PASS':
                self.interesting_examples['unexpected_b_passes'].append(self._format_example(record))
            self.interesting_examples_base_counts_by_topic['unexpected_b_passes'][record['topic']] += 1
                
            # Unexpected A fails (A variants should typically PASS)  
            if 'unexpected_fails' in example_types and variant == 'A' and verdict == 'FAIL':
                self.interesting_examples['unexpected_a_fails'].append(self._format_example(record))
            self.interesting_examples_base_counts_by_topic['unexpected_a_fails'][record['topic']] += 1

            # Low confidence (≤2)
            if 'low_confidence' in example_types and confidence is not None and confidence <= 2:
                self.interesting_examples['low_confidence'].append(self._format_example(record))
            self.interesting_examples_base_counts_by_topic['low_confidence'][record['topic']] += 1
                    
            # Token cutoff cases (exactly 2048 tokens)
            if record['output_tokens'] == 2048:
                self.interesting_examples['token_cutoffs'].append(self._format_example(record))
            self.interesting_examples_base_counts_by_topic['token_cutoffs'][record['topic']] += 1

        # A variants FAIL, B variants PASS
        if 'unexpected_a_b_inversion' in example_types:
            self._find_unexpected_a_b_inversion()
            
        # Find high confidence cases where interventions disagree  
        self._find_intervention_disagreements()
            
        # Collect duplicates
        self._find_duplicate_prompts()
        
        # Find systematic patterns 
        self._analyze_batch_patterns()
        
        # Find difficulty progression patterns
        self._analyze_difficulty_patterns()
        
        # Sort examples by confidence/quality for better review (but don't limit here)
        for category in self.interesting_examples:
            if category not in ['batch_analysis', 'difficulty_patterns', 'duplicate_prompts', 'intervention_disagreements']:
                self.interesting_examples[category] = sorted(
                    self.interesting_examples[category], 
                    key=lambda x: x.get('confidence') or 0 if isinstance(x, dict) else 0,
                    reverse=True
                )

    def _format_example(self, record: Dict) -> Dict:
        """Format a record as an example for manual review."""
        return {
            'category': 'auto-detected',
            'source': record['source_file'],
            'item_index': record['item_index'],
            'topic': record['topic'],
            'variant': record['variant'],
            'model': record['model'],
            'intervention': record['intervention_name'],
            'intervention_type': record['intervention_type'],
            'T': record['T'][:200] + "..." if len(record['T']) > 200 else record['T'],
            'S': record['S'][:300] + "..." if len(record['S']) > 300 else record['S'],
            'I': record['I'][:200] + "..." if len(record['I']) > 200 else record['I'],
            'R': record['R'][:300] + "..." if len(record['R']) > 300 else record['R'],
            'P': record['P'],
            'confidence': record['confidence'],
            'detection_difficulty': record['detection_difficulty'],
            'intended_review_focus': record['intended_review_focus'],
            'selected_review_focus': record['selected_review_focus'],
            'output_tokens': record['output_tokens']
        }
    
    def _find_unexpected_a_b_inversion(self):
        """Find cases where A variants FAIL, B variants PASS."""
        for batch_info in self.raw_batches:
            batch_data = batch_info['batch_data']
            topic = batch_data['topic']
            
            for item_idx, item in enumerate(batch_data['data']):
                affected_subsets = []
                for model_name, model_data in item['inferences'].items():
                    if not ('A' in model_data and 'B' in model_data):
                        continue
                    a_data = model_data['A']
                    b_data = model_data['B']
                    for intervention_name, intervention_data in a_data['interventions'].items():
                        if intervention_data['P'] == 'FAIL' and b_data['interventions'][intervention_name]['P'] == 'PASS':
                            affected_subsets.append([model_name, intervention_name])
                if len(affected_subsets) > 0:
                    example = {
                        'affected_subsets': affected_subsets,
                        'item': item,
                    }
                    self.interesting_examples['unexpected_a_b_inversion'].append(example)
                self.interesting_examples_base_counts_by_topic['unexpected_a_b_inversion'][topic] += 1
        
    def _find_intervention_disagreements(self):
        """Find cases where different interventions disagree."""
        for batch_info in self.raw_batches:
            batch_data = batch_info['batch_data']
            topic = batch_data['topic']

            for item_idx, item in enumerate(batch_data['data']):
                affected_subsets = []
                affected_subsets_high_confidence = []
                # Check each model/variant combination
                for model_name, model_data in item['inferences'].items():
                    for variant in ['A', 'B']:
                        if variant not in model_data:
                            continue

                        variant_data = model_data[variant]
                        interventions = variant_data['interventions']
                        interventions_with_confidence_and_flags = [
                            [intervention_name, intervention_data['P'], intervention_data['confidence']]
                            for intervention_name, intervention_data in interventions.items()
                        ]
                        
                        flags_of_interventions = [
                            intervention_data['P'] for intervention_data in interventions.values()
                        ]
                        if len(set(flags_of_interventions)) > 1:
                            affected_subsets.append([model_name, variant, interventions_with_confidence_and_flags])
                        
                        flags_of_interventions_with_high_confidence = [
                            intervention_data['P'] for intervention_data in interventions.values() if intervention_data['confidence'] >= 4
                        ]
                        if len(set(flags_of_interventions_with_high_confidence)) > 1:
                            affected_subsets_high_confidence.append([model_name, variant, interventions_with_confidence_and_flags])
                if len(affected_subsets) > 0:
                    example = {
                        'affected_subsets': affected_subsets,
                        'item': item,
                    }
                    self.interesting_examples['intervention_disagreements'].append(example)
                self.interesting_examples_base_counts_by_topic['intervention_disagreements'][topic] += 1
                if len(affected_subsets_high_confidence) > 0:
                    example = {
                        'affected_subsets': affected_subsets_high_confidence,
                        'item': item,
                    }
                    self.interesting_examples['intervention_disagreements_high_confidence'].append(example)
                self.interesting_examples_base_counts_by_topic['intervention_disagreements_high_confidence'][topic] += 1
                    
    def _find_duplicate_prompts(self):
        """Find duplicate T/S pairs within or across topics."""
        length_for_duplicate_detection = 50
        prompt_groups = defaultdict(list)
        for batch_info in self.raw_batches:
            batch_data = batch_info['batch_data']
            topic = batch_data['topic']
            for record in batch_data['data']:
                key = f"{record['T'][:length_for_duplicate_detection]}"
                val = {
                    'record': record,
                    'topic': topic,
                    'source_file': batch_info['source_file'],
                }
                prompt_groups[key].append(val)
            
        duplicates = []
        for key, records in prompt_groups.items():
            if len(records) > 1:
                if len(records) > 1:
                    duplicate_info = {
                        'key': key,
                        'count': len(records),
                        'topics': list(set(r['topic'] for r in records)),
                        'sources': list(set(r['source_file'] for r in records)),
                    }
                    duplicates.append(duplicate_info)
                        
        self.interesting_examples['duplicate_prompts'] = duplicates
        for record in self.all_data:
            self.interesting_examples_base_counts_by_topic['duplicate_prompts'][record['topic']] += 1
        
    def _analyze_batch_patterns(self):
        """Analyze patterns by batch to detect systematic differences."""
        batch_stats = defaultdict(lambda: {
            'total_samples': 0, 'pass_rate': 0, 'avg_confidence': 0,
            'topic': '', 'date': ''
        })
        
        for record in self.all_data:
            batch_id = record['batch_id']
            if batch_id:
                batch_stats[batch_id]['total_samples'] += 1
                batch_stats[batch_id]['topic'] = record['topic']
                
                # Extract date from batch_id if possible
                if 'batch_' in batch_id:
                    parts = batch_id.split('_')
                    if len(parts) >= 3:
                        batch_stats[batch_id]['date'] = parts[2]  # Date part
                        
                if record['P'] == 'PASS':
                    batch_stats[batch_id]['pass_rate'] += 1
                    
                if record['confidence'] is not None:
                    batch_stats[batch_id]['avg_confidence'] += record['confidence']
            self.interesting_examples_base_counts_by_topic['batch_analysis'][record['topic']] = 1
                    
        # Calculate averages and find outliers
        batch_analysis = []
        for batch_id, stats in batch_stats.items():
            if stats['total_samples'] > 0:
                pass_rate = (stats['pass_rate'] / stats['total_samples']) * 100
                avg_conf = stats['avg_confidence'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
                
                batch_analysis.append({
                    'batch_id': batch_id,
                    'topic': stats['topic'],
                    'date': stats['date'],
                    'samples': stats['total_samples'],
                    'pass_rate': pass_rate,
                    'avg_confidence': avg_conf
                })
                
        # Sort by pass rate to find outliers
        batch_analysis.sort(key=lambda x: x['pass_rate'])
        
        self.interesting_examples['batch_analysis'] = {
            'lowest_pass_rates': batch_analysis[:10],
            'highest_pass_rates': batch_analysis[-10:],
            'total_batches': len(batch_analysis)
        }
        
    def _analyze_difficulty_patterns(self):
        """Analyze difficulty progression patterns within topics."""
        difficulty_patterns = {}
        
        for topic in self.topics:
            topic_records = [r for r in self.all_data if r['topic'] == topic]
            
            # Group by difficulty
            difficulty_groups = defaultdict(list)
            for record in topic_records:
                diff = record['difficulty'] or record['detection_difficulty'] or 'unknown'
                difficulty_groups[diff].append(record)
                
            # Calculate pass rates by difficulty
            difficulty_stats = {}
            for diff, records in difficulty_groups.items():
                if records:
                    total = len(records)
                    passes = len([r for r in records if r['P'] == 'PASS'])
                    avg_conf = sum(r['confidence'] for r in records if r['confidence'] is not None) / len([r for r in records if r['confidence'] is not None]) if any(r['confidence'] is not None for r in records) else 0
                    
                    difficulty_stats[diff] = {
                        'samples': total,
                        'pass_rate': (passes / total) * 100,
                        'avg_confidence': avg_conf
                    }
                    
            difficulty_patterns[topic] = difficulty_stats
            self.interesting_examples_base_counts_by_topic['difficulty_patterns'][record['topic']] = 1
            
        self.interesting_examples['difficulty_patterns'] = difficulty_patterns

    def analyze_tags(self) -> None:
        """Analyze tag distribution across the dataset."""
        print("Analyzing tag distribution...")
        
        # Overall tag statistics
        total_tags_applied = sum(self.tag_counts.values())
        
        # Sort tags by frequency
        sorted_tags = sorted(self.tag_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate percentages
        tag_percentages = {}
        for tag, count in sorted_tags:
            # Percentage of total tag applications
            pct_of_tags = (count / total_tags_applied * 100) if total_tags_applied > 0 else 0
            # Percentage of records that have this tag (approximation)
            pct_of_records = (count / len(self.all_data) * 100) if self.all_data else 0
            tag_percentages[tag] = {
                'count': count,
                'pct_of_tags': pct_of_tags,
                'pct_of_records': pct_of_records
            }
        
        # Identify topics with most quality issues
        topic_issue_scores = {}
        quality_issue_tags = [
            'confidence_below_3', 'confidence_below_2', 'data_is_faulty',
            'mismatching_review_focus', 'generation_exceeded_token_limit',
            'intervention_tags_disagree_twice', 'unclear_a_b_separation',
            'a_fails_but_b_passes', 'missing_consensus_on_a', 'missing_consensus_on_b'
        ]
        
        for topic in self.topics:
            issue_count = sum(self.tag_counts_by_topic[topic].get(tag, 0) 
                            for tag in quality_issue_tags)
            # Get approximate number of records for this topic
            topic_records = len([r for r in self.all_data if r['topic'] == topic])
            issue_rate = (issue_count / topic_records * 100) if topic_records > 0 else 0
            topic_issue_scores[topic] = {
                'issue_count': issue_count,
                'total_records': topic_records,
                'issue_rate': issue_rate
            }
        
        # Sort topics by issue rate
        sorted_topic_issues = sorted(topic_issue_scores.items(), 
                                    key=lambda x: x[1]['issue_rate'], reverse=True)
        
        # Check for expected tags (sanity check)
        expected_tags = [
            'confidence_below_5', 'confidence_below_4', 'confidence_below_3', 'confidence_below_2',
            'data_is_faulty', 'mismatching_review_focus', 'generation_exceeded_token_limit',
            'intervention_tags_disagree_once', 'intervention_tags_disagree_twice',
            'no_a_b_separation', 'missing_consensus_on_a', 'missing_consensus_on_b',
            'unclear_a_b_separation', 'a_fails_but_b_passes'
        ]
        
        missing_tags = [tag for tag in expected_tags if tag not in self.tag_counts]
        
        self.tag_analysis = {
            'total_tags_applied': total_tags_applied,
            'unique_tags': len(self.all_tags),
            'tag_counts': dict(sorted_tags),
            'tag_percentages': tag_percentages,
            'topic_issue_scores': dict(sorted_topic_issues),
            'tag_counts_by_topic': {topic: dict(counts) for topic, counts in self.tag_counts_by_topic.items()},
            'missing_expected_tags': missing_tags,
            'quality_issue_tags': quality_issue_tags
        }
    
    def generate_cross_topic_comparisons(self) -> None:
        """Generate comparative statistics across topics."""
        print("Generating cross-topic comparisons...")
        
        # Topic ranking by quality score
        topic_quality_scores = [(topic, stats['quality_score']) for topic, stats in self.topic_stats.items()]
        topic_quality_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Model performance across topics
        model_avg_quality = defaultdict(list)
        for topic, stats in self.topic_stats.items():
            for model, perf in stats['model_performance'].items():
                model_avg_quality[model].append(perf['quality_score'])
                
        model_rankings = []
        for model, scores in model_avg_quality.items():
            avg_score = sum(scores) / len(scores) if scores else 0
            model_rankings.append((model, avg_score, len(scores)))
        model_rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Intervention effectiveness across topics
        intervention_avg_quality = defaultdict(list)
        for topic, stats in self.topic_stats.items():
            for intervention, perf in stats['intervention_performance'].items():
                intervention_avg_quality[intervention].append(perf['quality_score'])
                
        intervention_rankings = []
        for intervention, scores in intervention_avg_quality.items():
            avg_score = sum(scores) / len(scores) if scores else 0
            intervention_rankings.append((intervention, avg_score, len(scores)))
        intervention_rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Topic ranking by confidence
        topic_confidence_scores = [(topic, stats['avg_confidence']) for topic, stats in self.topic_stats.items()]
        topic_confidence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Topic ranking by difficulty  
        topic_difficulty_scores = [(topic, stats['avg_difficulty']) for topic, stats in self.topic_stats.items()]
        topic_difficulty_scores.sort(key=lambda x: x[1])  # Lower = easier (sort ascending)
        
        self.cross_topic_comparisons = {
            'topic_quality_ranking': topic_quality_scores,
            'topic_confidence_ranking': topic_confidence_scores,
            'topic_difficulty_ranking': topic_difficulty_scores,
            'model_ranking': model_rankings,
            'intervention_ranking': intervention_rankings,
            'intervention_set': sorted(list(self.intervention_types)),
        }

    def calculate_interestingness_breakdowns(self) -> Dict:
        """Calculate topic breakdown for each type of interestingness."""
        breakdowns = {}
        
        for category, examples in self.interesting_examples.items():
            relevant_base_count = sum(self.interesting_examples_base_counts_by_topic[category].values())
            assert relevant_base_count > 0, category
            if category in ['batch_analysis', 'difficulty_patterns']:
                # These have different structures
                breakdowns[category] = {
                    'total': len(examples) if isinstance(examples, list) else (len(examples.get('lowest_pass_rates', [])) + len(examples.get('highest_pass_rates', [])) if isinstance(examples, dict) else 0),
                    'structure': 'non-standard',
                    'dataset_count': relevant_base_count,
                    'percentage_of_dataset': len(examples) / relevant_base_count * 100,
                }
                continue
                
            topic_counts = {}
            total = 0
            
            if isinstance(examples, list):
                for example in examples:
                    if isinstance(example, dict):
                        # Handle different example structures
                        topic = None
                        if 'topic' in example:
                            topic = example['topic']
                        elif 'item' in example and example['item'] and 'metadata' in example['item']:
                            topic = example['item'].get('metadata', {}).get('alignment_issue_type')
                        elif 'records' in example and example['records'] and 'topic' in example['records'][0]:
                            topic = example['records'][0]['topic']
                        elif 'topics' in example and example['topics']:
                            # For duplicate_prompts which can span multiple topics
                            topic = example['topics'][0]  # Use first topic for counting
                        else:
                            assert False
                        
                        assert topic is not None
                        topic_counts[topic] = topic_counts.get(topic, 0) + 1
                        total += 1
            
            for k, v in topic_counts.items():
                assert self.interesting_examples_base_counts_by_topic[category][k] >= v, f"{category} {k} {v} {self.interesting_examples_base_counts_by_topic[category][k]}"
                
            breakdowns[category] = {
                'total': total,
                'by_topic': {
                    k: {
                        'count': v,
                        'dataset_count': self.interesting_examples_base_counts_by_topic[category][k],
                        'percentage': v / self.interesting_examples_base_counts_by_topic[category][k] * 100,
                    } for k, v in topic_counts.items()
                }
            }
                    
        return breakdowns

    def filter_examples_for_file_writing(self, examples: List, max_per_topic: int = 10) -> List:
        """Filter examples to have at most max_per_topic items per topic for file writing."""
        if not isinstance(examples, list) or not examples:
            return examples
            
        # Group examples by topic
        by_topic = {}
        
        for example in examples:
            if isinstance(example, dict):
                # Handle different example structures
                topic = None
                if 'topic' in example:
                    topic = example['topic']
                elif 'item' in example and example['item'] and 'metadata' in example['item']:
                    # For intervention_disagreements_high_confidence with new structure
                    topic = example['item'].get('metadata', {}).get('alignment_issue_type')
                elif 'records' in example and example['records'] and 'topic' in example['records'][0]:
                    topic = example['records'][0]['topic']
                elif 'topics' in example and example['topics']:
                    topic = example['topics'][0]  # Use first topic for grouping
                    
                if topic:
                    if topic not in by_topic:
                        by_topic[topic] = []
                    by_topic[topic].append(example)
                else:
                    # If no topic found, put in special group
                    if 'unknown' not in by_topic:
                        by_topic['unknown'] = []
                    by_topic['unknown'].append(example)
        
        # Take at most max_per_topic from each topic
        filtered_examples = []
        for topic_examples in by_topic.values():
            filtered_examples.extend(topic_examples[:max_per_topic])
            
        return filtered_examples

    def save_results(self, output_dir: str = "data/stage_3_analysis") -> None:
        """Save all analysis results to files."""
        print(f"\nSaving results to {output_dir}...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate interestingness breakdowns
        interestingness_breakdowns = self.calculate_interestingness_breakdowns()
        
        # Main summary report
        summary_report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'summary_stats': self.summary_stats,
            'cross_topic_comparisons': self.cross_topic_comparisons,
            'interestingness_breakdowns': interestingness_breakdowns,
            'sanity_check_failures': self.sanity_check_failures,
            'tag_analysis': self.tag_analysis
        }
        
        with open(output_path / 'summary_report.json', 'w') as f:
            json.dump(summary_report, f, indent=2)
            
        # Topic-specific analysis
        with open(output_path / 'topic_comparisons.json', 'w') as f:
            json.dump(dict(self.topic_stats), f, indent=2)
            
        # Tag analysis
        with open(output_path / 'tag_analysis.json', 'w') as f:
            json.dump(self.tag_analysis, f, indent=2)
            
        # Interesting examples
        examples_dir = output_path / 'interesting_examples'
        examples_dir.mkdir(exist_ok=True)
        
        for category, examples in self.interesting_examples.items():
            if examples:
                # Filter examples for file writing (max 10 per topic for manageable review)
                if category not in ['batch_analysis', 'difficulty_patterns']:
                    filtered_examples = self.filter_examples_for_file_writing(examples, max_per_topic=10)
                else:
                    # Don't filter these special categories (small or structured differently)
                    filtered_examples = examples
                    
                with open(examples_dir / f'{category}.json', 'w') as f:
                    json.dump(filtered_examples, f, indent=2)
                    
        # Per-topic detailed analysis
        by_topic_dir = output_path / 'by_topic'
        by_topic_dir.mkdir(exist_ok=True)
        
        for topic, stats in self.topic_stats.items():
            with open(by_topic_dir / f'{topic}_analysis.json', 'w') as f:
                json.dump(stats, f, indent=2)
                
        print(f"✅ Results saved to {output_path}")

    def print_summary(self, detailed: bool = False) -> None:
        """Print analysis summary to console."""
        print("\n" + "="*80)
        print("STAGE 3 ANALYSIS SUMMARY")
        print("="*80)
        
        stats = self.summary_stats
        print(f"📊 Total intervention records: {stats['total_samples']:,}")
        print(f"🎯 Topics analyzed: {stats['unique_topics']}")
        print(f"🤖 Models: {stats['unique_models']}")
        print(f"🔧 Intervention types: {stats['unique_interventions']}")
        
        print(f"\n🎭 Overall verdict distribution:")
        for verdict, count in stats['verdict_counts'].items():
            pct = count / stats['total_samples'] * 100 if stats['total_samples'] > 0 else 0
            print(f"  {verdict}: {count:,} ({pct:.1f}%)")
            
        print(f"\n🪙 Token analysis:")
        print(f"  Samples at max tokens (2048): {stats['token_analysis']['samples_at_2048_tokens']:,} ({stats['token_analysis']['pct_samples_at_2048_tokens']:.1f}%)")
        
        print(f"\n🏆 Top topics by quality score:")
        for topic, score in self.cross_topic_comparisons['topic_quality_ranking'][:5]:
            print(f"  {topic}: {score:.1f}%")
            
        print(f"\n🔧 Intervention effectiveness ranking:")
        for intervention, score, samples in self.cross_topic_comparisons['intervention_ranking']:
            print(f"  {intervention}: {score:.1f}% (n={samples})")
            
        print(f"\n📋 Interesting examples found:")
        for category, examples in self.interesting_examples.items():
            if examples:
                print(f"  {category}: {len(examples)} out of {sum(self.interesting_examples_base_counts_by_topic[category].values())}")
                
        # Print tag analysis summary
        if hasattr(self, 'tag_analysis') and self.tag_analysis:
            print(f"\n🏷️  Tag Analysis:")
            print(f"  Total tags applied: {self.tag_analysis['total_tags_applied']:,}")
            print(f"  Unique tag types: {self.tag_analysis['unique_tags']}")
            
            print(f"\n  Most common tags:")
            for tag, count in list(self.tag_analysis['tag_counts'].items())[:5]:
                pct = self.tag_analysis['tag_percentages'][tag]['pct_of_records']
                print(f"    {tag}: {count:,} ({pct:.1f}% of records)")
                
            print(f"\n  Topics with most quality issues:")
            for topic, scores in list(self.tag_analysis['topic_issue_scores'].items())[:5]:
                print(f"    {topic}: {scores['issue_rate']:.1f}% issue rate ({scores['issue_count']} issues)")
                
            if self.tag_analysis.get('missing_expected_tags'):
                print(f"\n  ⚠️  Missing expected tags: {', '.join(self.tag_analysis['missing_expected_tags'])}")
                
        if self.sanity_check_failures:
            print(f"\n⚠️  Sanity check failures: {len(self.sanity_check_failures)}")
            assert False, "Sanity check failures"
            
        if detailed:
            self._print_detailed_stats()
            
    def _print_detailed_stats(self) -> None:
        """Print detailed statistics."""
        print("\n" + "-"*60)
        print("DETAILED STATISTICS")
        print("-"*60)
        
        # A/B balance by topic
        print("\n📊 A/B Variant Balance by Topic:")
        for topic, balance in self.summary_stats['ab_balance'].items():
            print(f"  {topic}: A={balance['A']} ({balance['A_pct']:.1f}%) | B={balance['B']} ({balance['B_pct']:.1f}%)")
            
        # Confidence distribution
        print(f"\n🎯 Confidence Score Distribution:")
        for conf, count in sorted(self.summary_stats['confidence_distribution'].items()):
            pct = count / self.summary_stats['total_samples'] * 100 if self.summary_stats['total_samples'] > 0 else 0
            print(f"  {conf}: {count:,} ({pct:.1f}%)")
            
        # Detection difficulty distribution  
        print(f"\n📈 Detection Difficulty Distribution:")
        for diff, count in sorted(self.summary_stats['difficulty_distribution'].items()):
            pct = count / self.summary_stats['total_samples'] * 100 if self.summary_stats['total_samples'] > 0 else 0
            print(f"  {diff}: {count:,} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Analyze Stage 3 intervention results')
    parser.add_argument('--topic', help='Focus on specific topic')
    parser.add_argument('--examples', help='Comma-separated list of example types to collect')
    parser.add_argument('--output-format', choices=['json', 'markdown'], default='json', help='Output format')
    parser.add_argument('--detailed', action='store_true', help='Generate detailed analysis')
    parser.add_argument('--ignore-errors', action='store_true', help='Continue despite sanity check failures')
    parser.add_argument('--data-dir', default='data/stage_3_tagged', help='Stage 3 tagged data directory')
    parser.add_argument('--output-dir', default='data/stage_3_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = Stage3Analyzer(data_dir=args.data_dir, ignore_errors=args.ignore_errors)
        
        # Load and validate data
        analyzer.load_all_data()
        analyzer.run_sanity_checks()
        
        # Run analysis
        analyzer.calculate_summary_stats()
        analyzer.analyze_by_topic()
        
        # Analyze tags
        analyzer.analyze_tags()
        
        # Collect examples if requested
        example_types = None
        if args.examples:
            example_types = [t.strip() for t in args.examples.split(',')]
        analyzer.collect_interesting_examples(example_types)
        
        # Generate comparisons
        analyzer.generate_cross_topic_comparisons()
        
        # Filter by topic if requested
        if args.topic:
            if args.topic not in analyzer.topics:
                print(f"Error: Topic '{args.topic}' not found. Available topics: {sorted(analyzer.topics)}")
                sys.exit(1)
            # Note: Topic filtering would be implemented here if needed for output
            
        # Save results
        analyzer.save_results(args.output_dir)
        
        # Print summary
        analyzer.print_summary(detailed=args.detailed)
        
    except Exception as e:
        print(f"Error: {e}")
        if not args.ignore_errors:
            raise
        sys.exit(1)


if __name__ == '__main__':
    main()