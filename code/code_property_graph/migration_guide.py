#!/usr/bin/env python3
"""
Migration Guide: Transitioning from old CPG generation code to the new package.

This script helps users migrate their existing CPG generation workflows to use
the new @code_property_graph package.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_property_graph import CPGGenerator, CPGProcessor, CPGAnalyzer, CPGConfig


def migrate_genCpgs_py():
    """
    Migrate from genCpgs.py to the new package.
    
    Old code:
    ```python
    # Old genCpgs.py approach
    def process_project(project):
        # ... manual implementation
        subprocess.run(['git', 'checkout', '-f', commit_hash], cwd=project_path, check=True)
        subprocess.run(['joern-parse', project_path, '-J-Xmx25G', '-o', cpg_file], check=True)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=21) as executor:
        executor.map(process_project, projects)
    ```
    
    New approach:
    ```python
    # New package approach
    from code_property_graph import CPGGenerator
    
    generator = CPGGenerator()
    results = generator.generate_cpgs_for_projects(projects, parallel=True)
    ```
    """
    print("=== Migrating from genCpgs.py ===")
    
    # Old approach simulation
    print("Old approach (genCpgs.py):")
    print("- Manual git checkout and joern-parse calls")
    print("- Manual thread pool management")
    print("- Hardcoded paths and settings")
    print("- Limited error handling")
    
    # New approach
    print("\nNew approach (@code_property_graph package):")
    print("- Automated git operations and CPG generation")
    print("- Built-in parallel processing")
    print("- Configurable paths and settings")
    print("- Comprehensive error handling and validation")
    
    # Migration example
    print("\nMigration Example:")
    print("```python")
    print("# OLD: genCpgs.py")
    print("def process_project(project):")
    print("    # ... manual implementation")
    print("    subprocess.run(['git', 'checkout', '-f', commit_hash], cwd=project_path, check=True)")
    print("    subprocess.run(['joern-parse', project_path, '-J-Xmx25G', '-o', cpg_file], check=True)")
    print("")
    print("# NEW: @code_property_graph package")
    print("from code_property_graph import CPGGenerator")
    print("generator = CPGGenerator()")
    print("results = generator.generate_cpgs_for_projects(projects, parallel=True)")
    print("```")
    
    return True


def migrate_gen_graphml_sh():
    """
    Migrate from gen_graphml.sh to the new package.
    
    Old approach:
    ```bash
    # Old shell script approach
    joern-export "$cpg_file_path" -J-Xmx25G --repr all --format graphml -o "$graphml_file"
    zip -r "$cpg_output_path/$graphml_zip_name" "$graphml_file"
    gsutil cp "$cpg_output_path/$graphml_zip_name" "gs://$bucket_name/$project/$graphml_zip_name"
    ```
    
    New approach:
    ```python
    # New package approach
    from code_property_graph import CPGProcessor
    
    processor = CPGProcessor()
    results = processor.process_project_with_cloud_upload(project_name)
    ```
    """
    print("\n=== Migrating from gen_graphml.sh ===")
    
    # Old approach simulation
    print("Old approach (gen_graphml.sh):")
    print("- Shell script with manual joern-export calls")
    print("- Manual ZIP compression")
    print("- Manual cloud storage upload")
    print("- Limited error handling and retry logic")
    
    # New approach
    print("\nNew approach (@code_property_graph package):")
    print("- Python-based processing with error handling")
    print("- Automated GraphML export and compression")
    print("- Integrated cloud storage upload")
    print("- Comprehensive error handling and retry logic")
    
    # Migration example
    print("\nMigration Example:")
    print("```bash")
    print("# OLD: gen_graphml.sh")
    print("joern-export \"$cpg_file_path\" -J-Xmx25G --repr all --format graphml -o \"$graphml_file\"")
    print("zip -r \"$cpg_output_path/$graphml_zip_name\" \"$graphml_file\"")
    print("gsutil cp \"$cpg_output_path/$graphml_zip_name\" \"gs://$bucket_name/$project/$graphml_zip_name\"")
    print("```")
    print("")
    print("```python")
    print("# NEW: @code_property_graph package")
    print("from code_property_graph import CPGProcessor")
    print("processor = CPGProcessor()")
    print("results = processor.process_project_with_cloud_upload(project_name)")
    print("```")
    
    return True


def migrate_cpgs_script_py():
    """
    Migrate from cpgs_script.py to the new package.
    
    Old approach:
    ```python
    # Old cpgs_script.py approach
    def filter_cpg_by_long_names(graph, long_names):
        # ... manual implementation
        ces_names = [clean_java_qualified_name(name) for name in long_names]
        start_nodes = [n for n, d in graph.nodes(data=True) if d.get('name') in ces_names]
        # ... manual graph traversal
    ```
    
    New approach:
    ```python
    # New package approach
    from code_property_graph import CPGAnalyzer
    
    analyzer = CPGAnalyzer()
    filtered_graph = analyzer.filter_cpg_by_long_names(graph, long_names)
    ```
    """
    print("\n=== Migrating from cpgs_script.py ===")
    
    # Old approach simulation
    print("Old approach (cpgs_script.py):")
    print("- Manual CPG filtering implementation")
    print("- Manual graph traversal logic")
    print("- Limited analysis capabilities")
    print("- Hardcoded filtering logic")
    
    # New approach
    print("\nNew approach (@code_property_graph package):")
    print("- Built-in CPG filtering methods")
    print("- Advanced graph analysis capabilities")
    print("- Comprehensive code smell detection")
    print("- Extensible analysis framework")
    
    # Migration example
    print("\nMigration Example:")
    print("```python")
    print("# OLD: cpgs_script.py")
    print("def filter_cpg_by_long_names(graph, long_names):")
    print("    ces_names = [clean_java_qualified_name(name) for name in long_names]")
    print("    start_nodes = [n for n, d in graph.nodes(data=True) if d.get('name') in ces_names]")
    print("    # ... manual implementation")
    print("```")
    print("")
    print("```python")
    print("# NEW: @code_property_graph package")
    print("from code_property_graph import CPGAnalyzer")
    print("analyzer = CPGAnalyzer()")
    print("filtered_graph = analyzer.filter_cpg_by_long_names(graph, long_names)")
    print("")
    print("# Additional analysis capabilities")
    print("report = analyzer.generate_cpg_report(graph, project_name)")
    print("smells = analyzer.find_code_smells(graph)")
    print("complexity = analyzer.analyze_method_complexity(graph)")
    print("```")
    
    return True


def create_migration_examples():
    """Create example migration scripts for common use cases."""
    print("\n=== Creating Migration Examples ===")
    
    examples_dir = Path(__file__).parent / "migration_examples"
    examples_dir.mkdir(exist_ok=True)
    
    # Example 1: Basic CPG generation migration
    example1 = """#!/usr/bin/env python3
\"\"\"
Migration Example 1: Basic CPG Generation
Migrates from genCpgs.py to the new package
\"\"\"

from code_property_graph import CPGGenerator, CPGConfig

def old_approach():
    \"\"\"Old genCpgs.py approach (simplified)\"\"\"
    import subprocess
    import concurrent.futures
    
    def process_project(project):
        # Manual implementation
        project_path = f"/workspace/repos/{project}"
        cpg_output_path = f"/workspace/s2156631-thesis/cpgs/{project}"
        
        # Manual git checkout and joern-parse
        subprocess.run(['git', 'checkout', '-f', commit_hash], cwd=project_path, check=True)
        subprocess.run(['joern-parse', project_path, '-J-Xmx25G', '-o', cpg_file], check=True)
    
    # Manual thread pool management
    with concurrent.futures.ThreadPoolExecutor(max_workers=21) as executor:
        executor.map(process_project, projects)

def new_approach():
    \"\"\"New @code_property_graph package approach\"\"\"
    # Create configuration
    config = CPGConfig(
        repos_path="/workspace/repos",
        cpgs_output_path="/workspace/s2156631-thesis/cpgs",
        max_workers=21
    )
    
    # Initialize generator
    generator = CPGGenerator(config)
    
    # Generate CPGs with built-in parallel processing
    projects = ['commons-beanutils', 'commons-codec', 'commons-collections']
    results = generator.generate_cpgs_for_projects(projects, parallel=True)
    
    # Check results
    for project, cpg_files in results.items():
        print(f"{project}: {len(cpg_files)} CPG files generated")

if __name__ == "__main__":
    print("Running new approach...")
    new_approach()
"""
    
    # Example 2: GraphML processing migration
    example2 = """#!/usr/bin/env python3
\"\"\"
Migration Example 2: GraphML Processing
Migrates from gen_graphml.sh to the new package
\"\"\"

from code_property_graph import CPGProcessor, CPGConfig

def old_approach():
    \"\"\"Old gen_graphml.sh approach (simplified)\"\"\"
    import subprocess
    
    # Manual joern-export and compression
    subprocess.run([
        'joern-export', cpg_file, '-J-Xmx25G', '--repr', 'all', 
        '--format', 'graphml', '-o', graphml_file
    ])
    
    # Manual ZIP compression
    subprocess.run(['zip', '-r', zip_file, graphml_file])
    
    # Manual cloud upload
    subprocess.run(['gsutil', 'cp', zip_file, cloud_path])

def new_approach():
    \"\"\"New @code_property_graph package approach\"\"\"
    # Create configuration
    config = CPGConfig(
        bucket_name="s2156631-thesis",
        google_credentials_path="keys/ml-80805-4893ba01f974.json"
    )
    
    # Initialize processor
    processor = CPGProcessor(config)
    
    # Process project with integrated workflow
    project_name = "commons-beanutils"
    results = processor.process_project_with_cloud_upload(project_name)
    
    if results['success']:
        print(f"Successfully processed and uploaded {len(results['zip_files'])} files")
        print(f"Uploaded files: {results['uploaded_files']}")
    else:
        print(f"Processing failed: {results['errors']}")

if __name__ == "__main__":
    print("Running new approach...")
    new_approach()
"""
    
    # Example 3: CPG analysis migration
    example3 = """#!/usr/bin/env python3
\"\"\"
Migration Example 3: CPG Analysis
Migrates from cpgs_script.py to the new package
\"\"\"

from code_property_graph import CPGAnalyzer, CPGConfig
import networkx as nx

def old_approach():
    \"\"\"Old cpgs_script.py approach (simplified)\"\"\"
    import re
    
    def clean_java_qualified_name(name):
        # Manual implementation
        name = re.sub(r'\\)[VZBSIFJD]$', ')', name)
        # ... more manual cleaning logic
        return name
    
    def filter_cpg_by_long_names(graph, long_names):
        # Manual filtering implementation
        ces_names = [clean_java_qualified_name(name) for name in long_names]
        start_nodes = [n for n, d in graph.nodes(data=True) if d.get('name') in ces_names]
        
        # Manual graph traversal
        related_nodes = set()
        for start_node in start_nodes:
            for node in nx.dfs_preorder_nodes(graph, source=start_node):
                related_nodes.add(node)
        
        return graph.subgraph(related_nodes)

def new_approach():
    \"\"\"New @code_property_graph package approach\"\"\"
    # Initialize analyzer
    analyzer = CPGAnalyzer()
    
    # Load CPG
    graph = analyzer.load_cpg_from_graphml("commit_hash.graphml")
    
    # Use built-in filtering
    long_names = ["com.example.Class.method"]
    filtered_graph = analyzer.filter_cpg_by_long_names(graph, long_names)
    
    # Generate comprehensive analysis
    report = analyzer.generate_cpg_report(filtered_graph, "project_name")
    
    # Access various analysis results
    print(f"Graph statistics: {report['graph_statistics']}")
    print(f"Code smells: {report['code_smells']}")
    print(f"Method complexity: {report['method_complexity']}")
    print(f"Dependencies: {report['dependencies']}")
    
    # Extract specific analyses
    call_graph = analyzer.extract_call_graph(filtered_graph)
    complexity_metrics = analyzer.analyze_method_complexity(filtered_graph)
    code_smells = analyzer.find_code_smells(filtered_graph)
    
    print(f"Call graph nodes: {call_graph.number_of_nodes()}")
    print(f"High complexity methods: {len([m for m in complexity_metrics.values() if m['complexity_level'] in ['HIGH', 'VERY_HIGH']])}")
    print(f"Total code smells: {sum(len(smells) for smells in code_smells.values())}")

if __name__ == "__main__":
    print("Running new approach...")
    new_approach()
"""
    
    # Write examples to files
    examples = [
        ("01_basic_cpg_generation.py", example1),
        ("02_graphml_processing.py", example2),
        ("03_cpg_analysis.py", example3)
    ]
    
    for filename, content in examples:
        example_file = examples_dir / filename
        with open(example_file, 'w') as f:
            f.write(content)
        print(f"Created: {example_file}")
    
    return True


def show_migration_checklist():
    """Show a comprehensive migration checklist."""
    print("\n=== Migration Checklist ===")
    
    checklist = [
        ("1. Install the new package", "pip install -r requirements.txt"),
        ("2. Update import statements", "from code_property_graph import ..."),
        ("3. Replace manual CPG generation", "Use CPGGenerator class"),
        ("4. Replace manual GraphML processing", "Use CPGProcessor class"),
        ("5. Replace manual CPG analysis", "Use CPGAnalyzer class"),
        ("6. Update configuration", "Use CPGConfig class or environment variables"),
        ("7. Test with a single project", "Verify functionality before batch processing"),
        ("8. Update batch processing scripts", "Use built-in parallel processing"),
        ("9. Update error handling", "Use package's built-in error handling"),
        ("10. Remove old scripts", "Delete or archive old genCpgs.py, gen_graphml.sh, etc.")
    ]
    
    for item, action in checklist:
        print(f"{item:35} → {action}")
    
    return True


def main():
    """Run the complete migration guide."""
    print("Code Property Graph Package - Migration Guide")
    print("=" * 60)
    print("This guide helps you migrate from the old CPG generation code")
    print("to the new @code_property_graph package.\n")
    
    # Run migration sections
    sections = [
        ("Basic CPG Generation", migrate_genCpgs_py),
        ("GraphML Processing", migrate_gen_graphml_sh),
        ("CPG Analysis", migrate_cpgs_script_py),
        ("Creating Examples", create_migration_examples),
        ("Migration Checklist", show_migration_checklist)
    ]
    
    for section_name, section_func in sections:
        try:
            success = section_func()
            if not success:
                print(f"Warning: Section '{section_name}' had issues")
        except Exception as e:
            print(f"Error in section '{section_name}': {e}")
    
    print("\n" + "=" * 60)
    print("Migration Guide Complete!")
    print("=" * 60)
    print("Next steps:")
    print("1. Review the migration examples in @code_property_graph/migration_examples/")
    print("2. Update your existing scripts one by one")
    print("3. Test with small projects before scaling up")
    print("4. Use the CLI for quick operations: python -m code_property_graph --help")
    print("5. Check the README.md for detailed usage information")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
