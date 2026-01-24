#!/usr/bin/env python3
"""
Analyze and visualize samples where models disagree.
Creates an interactive HTML interface to explore disagreements and compare model outputs.
"""

import json
import glob
from html import escape as html_escape
from pathlib import Path
from collections import Counter

def load_model_results(base_dir: Path, eval_condition: str = "baseline"):
    """Load results for all models under a specific eval condition."""
    models_config = {
        'baseline': 'baseline',
        'aug_5pct': 'aug_5pct',
        'aug_15pct': 'aug_15pct',
        '15pct_no_sysprompt': '15pct_no_sysprompt',
        '15pct_mismatch': '15pct_mismatch',
        'full_no_sysprompt': 'full_no_sysprompt',
        'full_mismatch': 'full_mismatch',
    }
    
    all_results = {}
    
    for model_name, model_key in models_config.items():
        result_dir = base_dir / f"results_{model_key}_{eval_condition}"
        
        if not result_dir.exists():
            print(f"⚠️  {result_dir} not found, skipping {model_name}")
            continue
        
        json_files = sorted(result_dir.glob("eval_results_*.json"))
        if not json_files:
            print(f"⚠️  No JSON in {result_dir}")
            continue
        
        with open(json_files[0]) as f:
            data = json.load(f)
        
        # Index by (topic, sample_idx)
        indexed = {}
        for sample in data.get('samples', []):
            key = (sample['topic'], sample['sample_idx'])
            indexed[key] = sample
        
        all_results[model_name] = indexed
        print(f"✓ Loaded {model_name}: {len(indexed)} samples")
    
    return all_results


def find_disagreements(all_results):
    """Find samples where at least 2 models disagree on the verdict."""
    # Get all sample keys
    all_keys = set()
    for model_results in all_results.values():
        all_keys.update(model_results.keys())
    
    disagreements = []
    
    for key in sorted(all_keys):
        verdicts = []
        models_data = {}
        
        for model_name, results in all_results.items():
            if key in results:
                sample = results[key]
                verdict = sample.get('verdict', 'UNKNOWN')
                verdicts.append(verdict)
                models_data[model_name] = sample
        
        # Check if there's disagreement (at least 2 different verdicts)
        unique_verdicts = set(verdicts)
        if len(unique_verdicts) >= 2:
            disagreements.append({
                'key': key,
                'topic': key[0],
                'sample_idx': key[1],
                'models_data': models_data,
                'verdict_counts': Counter(verdicts),
                'num_models': len(verdicts),
            })
    
    return disagreements


def generate_html(disagreements, output_path: Path, eval_condition: str):
    """Generate interactive HTML for exploring disagreements."""
    
    model_display_names = {
        'baseline': 'Baseline',
        'aug_5pct': '5% mix',
        'aug_15pct': '15% mix',
        '15pct_no_sysprompt': '15% No SysPrompt',
        '15pct_mismatch': '15% Mismatch',
        'full_no_sysprompt': 'Full No SysPrompt',
        'full_mismatch': 'Full Mismatch',
    }
    
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <title>Model Disagreement Analysis</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1, h2 { color: #333; }
        .summary {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .filters {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .filter-group {
            margin-bottom: 10px;
        }
        .filter-button {
            padding: 6px 12px;
            margin: 2px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            cursor: pointer;
            font-size: 0.9em;
        }
        .filter-button.active {
            background: #2196f3;
            color: white;
            border-color: #2196f3;
        }
        .sample {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sample-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
        }
        .sample-id { font-weight: bold; color: #666; }
        .topic {
            display: inline-block;
            background: #e3f2fd;
            color: #1976d2;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.9em;
            font-weight: 500;
        }
        .section {
            margin: 15px 0;
        }
        .section-title {
            font-weight: 600;
            color: #555;
            margin-bottom: 8px;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .content {
            background: #fafafa;
            padding: 12px;
            border-radius: 4px;
            border-left: 3px solid #2196f3;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 0.9em;
            line-height: 1.5;
        }
        .models-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .model-card {
            background: #f9f9f9;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
        }
        .model-card.correct {
            border-color: #4caf50;
            background: #f1f8f4;
        }
        .model-card.incorrect {
            border-color: #f44336;
            background: #fef5f5;
        }
        .model-name {
            font-weight: 600;
            font-size: 1.1em;
            margin-bottom: 10px;
            color: #333;
        }
        .model-verdict {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 500;
            margin-bottom: 10px;
        }
        .model-verdict.correct {
            background: #c8e6c9;
            color: #2e7d32;
        }
        .model-verdict.incorrect {
            background: #ffcdd2;
            color: #c62828;
        }
        .model-output {
            background: white;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.85em;
            max-height: 300px;
            overflow-y: auto;
            margin-top: 8px;
            border-left: 3px solid #9c27b0;
            display: none;
        }
        .model-output.expanded {
            display: block;
        }
        .toggle-output {
            background: #2196f3;
            color: white;
            border: none;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8em;
            margin-top: 5px;
        }
        .toggle-output:hover {
            background: #1976d2;
        }
        .model-parsed {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
        .model-selector {
            margin: 15px 0;
        }
        .model-checkbox {
            margin-right: 15px;
            font-size: 0.9em;
        }
        .expected-flag {
            background: #fff3e0;
            border-left-color: #ff9800;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <h1>Model Disagreement Analysis</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Evaluation Condition:</strong> """ + eval_condition + """</p>
        <p><strong>Total Samples with Disagreement:</strong> """ + str(len(disagreements)) + """</p>
        <p>Showing samples where at least 2 models disagree on the verdict.</p>
    </div>

    <div class="filters">
        <div class="filter-group">
            <strong>Filter by Topic:</strong><br>
            <button class="filter-button active" onclick="filterByTopic('all')">All Topics</button>
"""
    
    # Add topic buttons
    topics = sorted(set(d['topic'] for d in disagreements))
    for topic in topics:
        count = sum(1 for d in disagreements if d['topic'] == topic)
        html += f'            <button class="filter-button" onclick="filterByTopic(\'{topic}\')">{topic} ({count})</button>\n'
    
    html += """        </div>
        <div class="filter-group">
            <strong>Model Selection (show/hide columns):</strong><br>
"""
    
    # Add model checkboxes
    all_models = ['baseline', 'aug_5pct', 'aug_15pct', '15pct_no_sysprompt', '15pct_mismatch', 'full_no_sysprompt', 'full_mismatch']
    for model in all_models:
        display_name = model_display_names.get(model, model)
        html += f'            <label class="model-checkbox"><input type="checkbox" value="{model}" checked onchange="toggleModel(\'{model}\')">{display_name}</label>\n'
    
    html += """        </div>
        <div class="filter-group">
            <button class="filter-button" onclick="expandAll()">Expand All Reviews</button>
            <button class="filter-button" onclick="collapseAll()">Collapse All Reviews</button>
        </div>
    </div>

    <div id="samples-container">
"""
    
    # Add each disagreement sample
    for d in disagreements:
        key = d['key']
        topic = d['topic']
        sample_idx = d['sample_idx']
        models_data = d['models_data']
        
        # Get base info from first available model
        first_sample = list(models_data.values())[0]
        
        html += f"""
    <div class="sample" data-topic="{html_escape(topic)}">
        <div class="sample-header">
            <div class="sample-id">Sample #{sample_idx}</div>
            <div class="topic">{html_escape(topic)}</div>
        </div>

        <div class="section">
            <div class="section-title">Task</div>
            <div class="content">{html_escape(first_sample.get('task', ''))}</div>
        </div>

        <div class="section">
            <div class="section-title">System Prompt</div>
            <div class="content">{html_escape(first_sample.get('system_prompt', ''))}</div>
        </div>

        <div class="section">
            <div class="section-title">Expected Flag</div>
            <div class="content expected-flag">{html_escape(first_sample.get('expected_flag', ''))}</div>
        </div>

        <div class="section">
            <div class="section-title">Response Being Evaluated</div>
            <div class="content">{html_escape(first_sample.get('response_evaluated', ''))}</div>
        </div>

        <div class="section">
            <div class="section-title">Model Outputs & Reviews</div>
            <div class="models-grid">
"""
        
        # Add each model's output
        for model_name in all_models:
            if model_name not in models_data:
                continue
            
            sample = models_data[model_name]
            verdict = sample.get('verdict', 'UNKNOWN') or 'UNKNOWN'
            is_correct = verdict == 'CORRECT'
            parsed_flag = sample.get('parsed_flag', '') or ''
            generated_output = sample.get('generated_output', '') or ''
            
            card_class = 'correct' if is_correct else 'incorrect'
            verdict_class = 'correct' if is_correct else 'incorrect'
            display_name = model_display_names.get(model_name, model_name)
            
            html += f"""
                <div class="model-card {card_class}" data-model="{html_escape(model_name)}">
                    <div class="model-name">{html_escape(display_name)}</div>
                    <div class="model-verdict {verdict_class}">{html_escape(verdict)}</div>
                    <div class="model-parsed">Parsed Flag: <strong>{html_escape(parsed_flag)}</strong></div>
                    <button class="toggle-output" onclick="toggleOutput(this)">Show Review</button>
                    <div class="model-output">{html_escape(generated_output)}</div>
                </div>
"""
        
        html += """
            </div>
        </div>
    </div>
"""
    
    html += """
    </div>

    <script>
        let currentTopic = 'all';
        let visibleModels = new Set(['baseline', 'aug_5pct', 'aug_15pct', '15pct_no_sysprompt', '15pct_mismatch', 'full_no_sysprompt', 'full_mismatch']);

        function toggleOutput(button) {
            console.log('toggleOutput called');
            const output = button.nextElementSibling;
            console.log('Next element:', output);
            console.log('Has expanded class:', output.classList.contains('expanded'));
            
            if (output.classList.contains('expanded')) {
                output.classList.remove('expanded');
                button.textContent = 'Show Review';
                console.log('Collapsed');
            } else {
                output.classList.add('expanded');
                button.textContent = 'Hide Review';
                console.log('Expanded');
            }
        }

        function expandAll() {
            document.querySelectorAll('.toggle-output').forEach(btn => {
                const output = btn.nextElementSibling;
                if (!output.classList.contains('expanded')) {
                    output.classList.add('expanded');
                    btn.textContent = 'Hide Review';
                }
            });
        }

        function collapseAll() {
            document.querySelectorAll('.toggle-output').forEach(btn => {
                const output = btn.nextElementSibling;
                if (output.classList.contains('expanded')) {
                    output.classList.remove('expanded');
                    btn.textContent = 'Show Review';
                }
            });
        }

        function filterByTopic(topic) {
            currentTopic = topic;
            updateDisplay();
            
            // Update button states
            document.querySelectorAll('.filter-button').forEach(btn => {
                if ((topic === 'all' && btn.textContent.includes('All Topics')) ||
                    btn.textContent.includes(topic)) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
        }

        function toggleModel(model) {
            if (visibleModels.has(model)) {
                visibleModels.delete(model);
            } else {
                visibleModels.add(model);
            }
            updateDisplay();
        }

        function updateDisplay() {
            const samples = document.querySelectorAll('.sample');
            
            samples.forEach(sample => {
                // Filter by topic
                const topicMatch = currentTopic === 'all' || sample.dataset.topic === currentTopic;
                sample.style.display = topicMatch ? 'block' : 'none';
                
                // Show/hide model cards
                if (topicMatch) {
                    sample.querySelectorAll('.model-card').forEach(card => {
                        const model = card.dataset.model;
                        card.style.display = visibleModels.has(model) ? 'block' : 'none';
                    });
                }
            });
        }
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    # Use current directory results
    base_dir = Path(__file__).parent.parent
    output_dir = Path(__file__).parent
    
    eval_condition = 'mismatch'  # Can be changed to 'no_sysprompt' or 'mismatch'
    
    print("=" * 80)
    print(f"Loading results for eval condition: {eval_condition}")
    print("=" * 80)
    
    all_results = load_model_results(base_dir, eval_condition)
    
    if not all_results:
        print("❌ No results loaded!")
        return
    
    print("\nFinding disagreements...")
    disagreements = find_disagreements(all_results)
    
    print(f"\n✓ Found {len(disagreements)} samples with disagreements")
    
    # Print some stats
    print("\nDisagreement patterns:")
    for d in disagreements[:5]:
        print(f"  {d['topic']}/{d['sample_idx']}: {dict(d['verdict_counts'])}")
    
    # Generate HTML
    html_output = output_dir / 'model_disagreements.html'
    print(f"\nGenerating HTML interface...")
    generate_html(disagreements, html_output, eval_condition)
    print(f"✓ Saved to: {html_output}")
    print(f"\n🌐 Open in browser: file://{html_output.absolute()}")
    print(f"\nClick the link above to open in your browser!")


if __name__ == '__main__':
    main()

