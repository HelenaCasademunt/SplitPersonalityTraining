#!/usr/bin/env python3
"""
Script to tag Stage 3 data for quality filtering.

This script:
1. Copies Stage 3 data to stage_3_tagged directory
2. Removes baseline interventions
3. Adds quality tags at multiple hierarchical levels
"""

import json
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Set
import sys
import re
from datetime import datetime

from utils import STAGE_3_DATA_DIR, STAGE_3_TAGGED_DATA_DIR

# Training data leakage patterns
# Split into two groups: patterns that should be checked in both I and R fields,
# and patterns that should only be checked in R field (review meta-commentary)

# Patterns to check in BOTH I and R fields (actual training data leakage)
LEAKAGE_PATTERNS_BOTH_FIELDS = [
    # A/B Prompt references
    (r'\b[AB]\s+(?:variant|prompt)\b', "A/B variant/prompt"),
    (r'\bthe\s+[AB]\s+prompt\b', "the A/B prompt"),
    (r'\bbiased\s+\[?[AB]\]?\s+prompt\b', "biased [A/B] prompt"),
    (r'\bunbiased\s+\[?[AB]\]?\s+prompt\b', "unbiased [A/B] prompt"),
    (r'\bsystem\s+prompt\s+[AB]\b', "system prompt A/B"),

    # Stage references
    (r'\bStage\s+[123]\b', "Stage 1/2/3"),
    (r'\bstage\s+[123]\b', "stage 1/2/3"),

    # Sample/Batch enumeration (when referring to generation process)
    (r'\bSample\s+\d+(?:\s*[:\-])', "Sample N:"),
    (r'\bbatch\s+\d+\b', "batch N"),
    (r'\bitem\s+\d+\s+of\s+\d+\b', "item N of M"),

    # Project jargon
    (r'\bFPT\b', "FPT"),
    (r'\bFalse\s+Positive\s+Test\b', "False Positive Test"),

    # Intervention names (should not appear in training data)
    (r'\bhonesty\s+scare\b', "honesty scare"),
    (r'\bsplit\s+personality\b', "split personality"),
    (r'\bintervention\s+(?:type|name|method)\b', "intervention type/name/method"),

    # Process references
    (r'\bthis\s+sample\s+(?:tests|demonstrates|uses)\b', "this sample tests/demonstrates/uses"),
    (r'\b(?:Using|Applying)\s+the\s+\w+\s+intervention\b', "Using/Applying the X intervention"),
    (r'\bthis\s+response\s+(?:was\s+)?generated\s+using\b', "this response (was) generated using"),
]

# Patterns to check ONLY in R field (review meta-commentary that appears legitimately in interventions)
LEAKAGE_PATTERNS_R_ONLY = [
    (r'\breview\s+focus\s+(?:is|was|should)\b', "review focus is/was/should"),
    (r'\bselected\s+review\s+focus\b', "selected review focus"),
    (r'\bintended\s+review\s+focus\b', "intended review focus"),
    (r'\bformal\s+investigation\b', "formal investigation"),
    (r'\bunder\s+(?:review|investigation|audit)\b', "under review/investigation/audit"),
]

# Compile patterns for efficiency
COMPILED_PATTERNS_BOTH = [(re.compile(pattern, re.IGNORECASE), desc) for pattern, desc in LEAKAGE_PATTERNS_BOTH_FIELDS]
COMPILED_PATTERNS_R_ONLY = [(re.compile(pattern, re.IGNORECASE), desc) for pattern, desc in LEAKAGE_PATTERNS_R_ONLY]


class Stage3Tagger:
    def __init__(self):
        self.source_dir = STAGE_3_DATA_DIR
        self.target_dir = STAGE_3_TAGGED_DATA_DIR
        self.baseline_removed_count = 0
        self.files_processed = 0
        self.items_processed = 0
        self.tags_applied = defaultdict(int)
        self.global_id_counter = 0
        self.unique_ids = set()  # To verify uniqueness
        self.variants_dropped_no_interventions = defaultdict(int)  # Track dropped variants by model
        
    def run(self):
        """Main execution function."""
        print("Stage 3 Data Tagging Script")
        print("=" * 60)
        
        # Step 1: Clean and prepare target directory
        self.prepare_target_directory()
        
        # Step 2: Process all JSON files
        self.process_all_files()
        
        # Step 3: Verify baseline removal occurred
        self.verify_baseline_removal()
        
        # Step 4: Print summary
        self.print_summary()
        
    def prepare_target_directory(self):
        """Delete and recreate the target directory."""
        print(f"Preparing target directory: {self.target_dir}")
        
        if self.target_dir.exists():
            print(f"  Removing existing directory...")
            shutil.rmtree(self.target_dir)
            
        print(f"  Creating fresh directory structure...")
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
    def process_all_files(self):
        """Process all JSON files from source to target."""
        print(f"\nProcessing files from {self.source_dir}")
        
        json_files = list(self.source_dir.rglob("*.json"))
        print(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            self.process_single_file(json_file)
            
    def process_single_file(self, source_file: Path):
        """Process a single JSON file."""
        # Calculate relative path and target location
        relative_path = source_file.relative_to(self.source_dir)
        target_file = self.target_dir / relative_path
        
        # Create parent directories if needed
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Read source data
        with open(source_file, 'r') as f:
            data = json.load(f)
            
        # Process the batch data
        if 'data' in data and isinstance(data['data'], list):
            for item_idx, item in enumerate(data['data']):
                # Pass the file name and item index for ID generation
                self.process_item(item, data, source_file.stem, item_idx)
                self.items_processed += 1
                
        # Write tagged data to target
        with open(target_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        self.files_processed += 1
        if self.files_processed % 10 == 0:
            print(f"  Processed {self.files_processed} files...")
            
    def process_item(self, item: Dict, batch_data: Dict, file_name: str, item_idx: int):
        """Process a single item, adding tags at all levels."""
        # Get intended review focus from metadata (for intervention-level tags)
        intended_review_focus = item.get('metadata', {}).get('intended_review_focus')

        # Process each model's inferences
        for model_name, model_data in item.get('inferences', {}).items():
            # First pass: Check for and drop variants without interventions field
            variants_to_drop = []
            for variant in ['A', 'B']:
                if variant in model_data:
                    variant_data = model_data[variant]
                    if 'interventions' not in variant_data:
                        variants_to_drop.append(variant)
                        self.variants_dropped_no_interventions[model_name] += 1

            # Drop variants without interventions
            for variant in variants_to_drop:
                del model_data[variant]

            # Check if only A variant exists (no B)
            has_a = 'A' in model_data
            has_b = 'B' in model_data
            no_a_b_separation = has_a and not has_b

            # Collect P values for consensus calculation
            a_p_values = []
            b_p_values = []

            # Process A and B variants
            for variant in ['A', 'B']:
                if variant not in model_data:
                    continue

                variant_data = model_data[variant]
                
                # Initialize tags list if not present
                if 'tags' not in variant_data:
                    variant_data['tags'] = []
                
                # Check for token limit exceeded (variant-level)
                if variant_data.get('output_tokens', 0) >= 2048:
                    variant_data['tags'].append('generation_exceeded_token_limit')
                    self.tags_applied['generation_exceeded_token_limit'] += 1
                
                # Process interventions
                interventions_to_remove = []
                intervention_p_values = []
                
                for intervention_name, intervention_data in variant_data.get('interventions', {}).items():
                    # Remove baseline interventions
                    if intervention_name.endswith('_baseline'):
                        interventions_to_remove.append(intervention_name)
                        self.baseline_removed_count += 1
                        continue
                    
                    # Initialize tags list for intervention
                    if 'tags' not in intervention_data:
                        intervention_data['tags'] = []
                    
                    # Add unique IDs to each intervention
                    self.add_ids_to_intervention(intervention_data, file_name, item_idx, 
                                                model_name, variant, intervention_name)
                    
                    # Apply intervention-level tags
                    self.apply_intervention_tags(intervention_data, intended_review_focus)
                    
                    # Collect P values for disagreement analysis
                    if 'P' in intervention_data:
                        intervention_p_values.append(intervention_data['P'])
                        if variant == 'A':
                            a_p_values.append(intervention_data['P'])
                        else:
                            b_p_values.append(intervention_data['P'])
                
                # Remove baseline interventions
                for intervention_name in interventions_to_remove:
                    del variant_data['interventions'][intervention_name]
                
                # Check for intervention disagreements (variant-level tags)
                self.check_intervention_disagreements(variant_data, intervention_p_values)
                
                # Propagate model-level tags to variant
                if no_a_b_separation:
                    variant_data['tags'].append('no_a_b_separation')
                    self.tags_applied['no_a_b_separation'] += 1
            
            # Calculate consensus and apply model-level comparison tags
            if has_a and has_b:
                self.apply_consensus_tags(model_data, a_p_values, b_p_values)
                
    def add_ids_to_intervention(self, intervention_data: Dict, file_name: str, 
                                item_idx: int, model_name: str, variant: str, 
                                intervention_name: str):
        """Add unique IDs to each intervention."""
        # Generate running number ID
        self.global_id_counter += 1
        intervention_data['id_numeric'] = self.global_id_counter
        
        # Generate interpretable string ID
        # Keep the full intervention name to ensure uniqueness
        # Replace any problematic characters for file paths
        clean_model = model_name.replace('/', '_')
        
        # Create the interpretable ID
        interpretable_id = f"{file_name}__idx{item_idx}__{clean_model}_{variant}__{intervention_name}"
        intervention_data['id_string'] = interpretable_id
        
        # Verify uniqueness
        if interpretable_id in self.unique_ids:
            raise ValueError(f"Duplicate ID detected: {interpretable_id}")
        self.unique_ids.add(interpretable_id)
    
    def apply_intervention_tags(self, intervention_data: Dict, intended_review_focus: str):
        """Apply tags at the intervention level."""
        tags = intervention_data['tags']
        
        # Confidence-based tags
        confidence = intervention_data.get('confidence')
        if confidence is not None:
            if confidence < 5:
                tags.append('confidence_below_5')
                self.tags_applied['confidence_below_5'] += 1
            if confidence < 4:
                tags.append('confidence_below_4')
                self.tags_applied['confidence_below_4'] += 1
            if confidence < 3:
                tags.append('confidence_below_3')
                self.tags_applied['confidence_below_3'] += 1
            if confidence < 2:
                tags.append('confidence_below_2')
                self.tags_applied['confidence_below_2'] += 1
        
        # Faulty data tag
        if intervention_data.get('faulty_data', False):
            tags.append('data_is_faulty')
            self.tags_applied['data_is_faulty'] += 1
        
        # Review focus mismatch
        selected_review_focus = intervention_data.get('selected_review_focus')
        if intended_review_focus and selected_review_focus:
            # Check if neither is a substring of the other
            if not (intended_review_focus in selected_review_focus or
                    selected_review_focus in intended_review_focus):
                tags.append('mismatching_review_focus')
                self.tags_applied['mismatching_review_focus'] += 1

        # Detection difficulty tags
        difficulty_internal = intervention_data.get('detection_difficulty_internal')
        difficulty_external = intervention_data.get('detection_difficulty_external')

        # Tag based on external difficulty
        if difficulty_external == 'easy':
            tags.append('easy_detection_difficulty_external')
            self.tags_applied['easy_detection_difficulty_external'] += 1
        elif difficulty_external == 'medium':
            tags.append('medium_detection_difficulty_external')
            self.tags_applied['medium_detection_difficulty_external'] += 1
        elif difficulty_external == 'hard':
            tags.append('hard_detection_difficulty_external')
            self.tags_applied['hard_detection_difficulty_external'] += 1

        # Tag based on internal difficulty
        if difficulty_internal == 'easy':
            tags.append('easy_detection_difficulty_internal')
            self.tags_applied['easy_detection_difficulty_internal'] += 1
        elif difficulty_internal == 'medium':
            tags.append('medium_detection_difficulty_internal')
            self.tags_applied['medium_detection_difficulty_internal'] += 1
        elif difficulty_internal == 'hard':
            tags.append('hard_detection_difficulty_internal')
            self.tags_applied['hard_detection_difficulty_internal'] += 1

        # Paradoxical case: external easier than internal
        # This should never happen - external detection should be harder or equal
        if difficulty_internal and difficulty_external:
            difficulty_order = {'easy': 1, 'medium': 2, 'hard': 3, 'NA': 0}
            internal_level = difficulty_order.get(difficulty_internal, 0)
            external_level = difficulty_order.get(difficulty_external, 0)

            # If external is easier (lower number) than internal, that's paradoxical
            # But only flag if both are actual difficulties (not NA)
            if external_level > 0 and internal_level > 0 and external_level < internal_level:
                tags.append('detection_difficulty_easier_externally_than_internally')
                self.tags_applied['detection_difficulty_easier_externally_than_internally'] += 1

        # Training data leakage detection
        # Check I (intervention) and R (review) fields for leaked terminology
        i_text = intervention_data.get('I', '')
        r_text = intervention_data.get('R', '')

        has_leakage = False

        # Check patterns that apply to both fields
        if self.check_leakage_both_fields(i_text) or self.check_leakage_both_fields(r_text):
            has_leakage = True

        # Check R-only patterns (only in review field)
        if self.check_leakage_r_only(r_text):
            has_leakage = True

        if has_leakage:
            tags.append('training_data_leakage')
            self.tags_applied['training_data_leakage'] += 1

        # Stage 3 review quality control
        stage_3_reviews_passed = intervention_data.get('stage_3_reviews_passed')
        if stage_3_reviews_passed is False:
            tags.append('failed_external_review')
            self.tags_applied['failed_external_review'] += 1

    def check_leakage_both_fields(self, text: str) -> bool:
        """Check if text contains leakage patterns that apply to both I and R fields."""
        if not text:
            return False

        for pattern, _ in COMPILED_PATTERNS_BOTH:
            if pattern.search(text):
                return True
        return False

    def check_leakage_r_only(self, text: str) -> bool:
        """Check if text contains leakage patterns that only apply to R field."""
        if not text:
            return False

        for pattern, _ in COMPILED_PATTERNS_R_ONLY:
            if pattern.search(text):
                return True
        return False
                
    def check_intervention_disagreements(self, variant_data: Dict, p_values: List[str]):
        """Check for disagreements between interventions and add tags."""
        if not p_values:
            return
            
        tags = variant_data['tags']
        
        # Count occurrences of each verdict
        p_counter = Counter(p_values)
        pass_count = p_counter.get('PASS', 0)
        fail_count = p_counter.get('FAIL', 0)
        
        # Check for disagreements (at least once)
        if pass_count >= 1 and fail_count >= 1:
            tags.append('intervention_tags_disagree_once')
            self.tags_applied['intervention_tags_disagree_once'] += 1
            
            # Propagate to all interventions
            for intervention_data in variant_data.get('interventions', {}).values():
                if 'intervention_tags_disagree_once' not in intervention_data['tags']:
                    intervention_data['tags'].append('intervention_tags_disagree_once')
        
        # Check for disagreements (at least twice)
        if pass_count >= 2 and fail_count >= 2:
            tags.append('intervention_tags_disagree_twice')
            self.tags_applied['intervention_tags_disagree_twice'] += 1
            
            # Propagate to all interventions
            for intervention_data in variant_data.get('interventions', {}).values():
                if 'intervention_tags_disagree_twice' not in intervention_data['tags']:
                    intervention_data['tags'].append('intervention_tags_disagree_twice')
                    
    def apply_consensus_tags(self, model_data: Dict, a_p_values: List[str], b_p_values: List[str]):
        """Calculate consensus and apply comparison tags at model level."""
        # Calculate consensus for A and B
        a_consensus = self.calculate_consensus(a_p_values)
        b_consensus = self.calculate_consensus(b_p_values)
        
        # Determine which tags to apply
        tags_to_add = []
        
        if a_consensus == "UNCLEAR":
            tags_to_add.append('missing_consensus_on_a')
            self.tags_applied['missing_consensus_on_a'] += 1
            
        if b_consensus == "UNCLEAR":
            tags_to_add.append('missing_consensus_on_b')
            self.tags_applied['missing_consensus_on_b'] += 1
            
        if a_consensus == b_consensus:
            tags_to_add.append('unclear_a_b_separation')
            self.tags_applied['unclear_a_b_separation'] += 1
            
        if a_consensus == "FAIL" and b_consensus == "PASS":
            tags_to_add.append('a_fails_but_b_passes')
            self.tags_applied['a_fails_but_b_passes'] += 1
        
        # Propagate tags to all variants and interventions
        for variant in ['A', 'B']:
            if variant in model_data:
                variant_data = model_data[variant]
                
                # Add to variant tags
                for tag in tags_to_add:
                    if tag not in variant_data['tags']:
                        variant_data['tags'].append(tag)
                
                # Add to all intervention tags
                for intervention_data in variant_data.get('interventions', {}).values():
                    for tag in tags_to_add:
                        if tag not in intervention_data['tags']:
                            intervention_data['tags'].append(tag)
                            
    def calculate_consensus(self, p_values: List[str]) -> str:
        """Calculate consensus from P values."""
        if not p_values:
            return "UNCLEAR"
            
        p_counter = Counter(p_values)
        pass_count = p_counter.get('PASS', 0)
        fail_count = p_counter.get('FAIL', 0)
        
        if pass_count > fail_count:
            return "PASS"
        elif fail_count > pass_count:
            return "FAIL"
        else:
            return "UNCLEAR"
            
    def verify_baseline_removal(self):
        """Verify that baseline interventions were removed."""
        print(f"\nVerification:")
        print(f"  Baseline interventions removed: {self.baseline_removed_count}")
        
        if self.baseline_removed_count == 0:
            raise Exception("No baseline interventions were found and removed. This is unexpected - the sanity check has failed.")
            
    def print_summary(self):
        """Print summary statistics."""
        print(f"\nSummary:")
        print(f"  Files processed: {self.files_processed}")
        print(f"  Items processed: {self.items_processed}")
        print(f"  Baseline interventions removed: {self.baseline_removed_count}")
        print(f"  Total interventions with IDs: {self.global_id_counter}")
        print(f"  Unique string IDs verified: {len(self.unique_ids)}")

        # Warn about dropped variants
        if self.variants_dropped_no_interventions:
            print(f"\n⚠️  WARNING: Dropped variants missing 'interventions' field:")
            total_dropped = 0
            for model_name in sorted(self.variants_dropped_no_interventions.keys()):
                count = self.variants_dropped_no_interventions[model_name]
                total_dropped += count
                print(f"  {model_name}: {count} variant(s)")
            print(f"  TOTAL: {total_dropped} variant(s) dropped")
            print(f"  These variants had incomplete stage 3 data and were excluded from tagged output.")

        print(f"\nTags Applied:")
        for tag, count in sorted(self.tags_applied.items()):
            print(f"  {tag}: {count}")

        print(f"\n✅ Tagging complete! Tagged data saved to: {self.target_dir}")


def main():
    """Main entry point."""
    tagger = Stage3Tagger()
    tagger.run()


if __name__ == '__main__':
    main()