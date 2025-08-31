#!/usr/bin/env python3
"""
Basic usage example for the Code Property Graph package.

This script demonstrates how to use the main classes to generate, process, and analyze CPGs.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from code_property_graph import CPGGenerator, CPGProcessor, CPGAnalyzer, CPGConfig


def example_basic_cpg_generation():
    """Example of basic CPG generation for a single project."""
    print("=== Basic CPG Generation Example ===")
    
    # Create a custom configuration
    config = CPGConfig(
        base_project_path="/workspace/s2156631-thesis",
        repos_path="/workspace/repos",
        cpgs_output_path="/workspace/s2156631-thesis/cpgs",
        max_workers=5  # Use fewer workers for this example
    )
    
    # Initialize the generator
    generator = CPGGenerator(config)
    
    # Generate CPGs for a single project
    project_name = "commons-beanutils"
    try:
        print(f"Generating CPGs for project: {project_name}")
        cpg_files = generator.generate_cpgs_for_project(project_name)
        print(f"Successfully generated {len(cpg_files)} CPG files")
        
        # Check project status
        status = generator.get_project_status(project_name)
        print(f"Project status: {status['generated_cpgs']}/{status['total_commits']} CPGs generated")
        
    except Exception as e:
        print(f"Error generating CPGs: {e}")
        return False
    
    return True


def example_cpg_processing():
    """Example of processing CPGs to GraphML format."""
    print("\n=== CPG Processing Example ===")
    
    config = CPGConfig()
    processor = CPGProcessor(config)
    
    project_name = "commons-beanutils"
    try:
        print(f"Processing CPGs for project: {project_name}")
        
        # Process CPGs to GraphML and compress
        graphml_files, zip_files = processor.process_and_compress_project(project_name)
        
        print(f"Generated {len(graphml_files)} GraphML files")
        print(f"Created {len(zip_files)} ZIP files")
        
        # Clean up temporary GraphML files (keep ZIPs)
        cleaned_files = processor.cleanup_temp_files(project_name, keep_graphml=False, keep_zip=True)
        print(f"Cleaned up {len(cleaned_files)} temporary files")
        
    except Exception as e:
        print(f"Error processing CPGs: {e}")
        return False
    
    return True


def example_cpg_analysis():
    """Example of analyzing a CPG file."""
    print("\n=== CPG Analysis Example ===")
    
    config = CPGConfig()
    analyzer = CPGAnalyzer(config)
    
    # Look for a GraphML file to analyze
    project_name = "commons-beanutils"
    cpg_dir = config.get_cpg_output_path(project_name)
    
    if not os.path.exists(cpg_dir):
        print(f"CPG directory does not exist: {cpg_dir}")
        return False
    
    # Find the first GraphML file
    graphml_files = [f for f in os.listdir(cpg_dir) if f.endswith('.graphml')]
    
    if not graphml_files:
        print("No GraphML files found for analysis")
        return False
    
    # Analyze the first file
    graphml_file = os.path.join(cpg_dir, graphml_files[0])
    print(f"Analyzing CPG file: {graphml_files[0]}")
    
    try:
        # Load the CPG
        graph = analyzer.load_cpg_from_graphml(graphml_file)
        
        # Generate a comprehensive report
        report = analyzer.generate_cpg_report(graph, project_name)
        
        # Print summary
        print(f"\nCPG Analysis Summary for {project_name}")
        print("=" * 50)
        print(f"Total nodes: {report['graph_statistics']['total_nodes']}")
        print(f"Total edges: {report['graph_statistics']['total_edges']}")
        print(f"Methods: {report['graph_statistics']['method_count']}")
        print(f"Classes: {report['graph_statistics']['class_count']}")
        print(f"Variables: {report['graph_statistics']['variable_count']}")
        print(f"Code smells detected: {report['summary']['total_smells']}")
        
        # Show some code smells if any
        if report['code_smells']['long_methods']:
            print(f"\nLong methods found: {len(report['code_smells']['long_methods'])}")
            for smell in report['code_smells']['long_methods'][:3]:  # Show first 3
                print(f"  - {smell['method_name']}: {smell['line_count']} lines")
        
        # Save detailed report
        report_file = os.path.join(cpg_dir, f"{project_name}_analysis_report.json")
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {report_file}")
        
    except Exception as e:
        print(f"Error analyzing CPG: {e}")
        return False
    
    return True


def example_batch_processing():
    """Example of batch processing multiple projects."""
    print("\n=== Batch Processing Example ===")
    
    config = CPGConfig(max_workers=3)  # Use fewer workers for this example
    generator = CPGGenerator(config)
    processor = CPGProcessor(config)
    
    # Process a subset of projects
    projects = ["commons-beanutils", "commons-codec"]
    
    try:
        print(f"Batch processing projects: {projects}")
        
        # Generate CPGs for all projects
        print("Step 1: Generating CPGs...")
        generation_results = generator.generate_cpgs_for_projects(projects, parallel=True)
        
        for project, cpg_files in generation_results.items():
            print(f"  {project}: {len(cpg_files)} CPG files")
        
        # Process all projects to GraphML
        print("\nStep 2: Processing CPGs to GraphML...")
        processing_results = processor.batch_process_projects(projects, upload_to_cloud=False, parallel=True)
        
        for project, results in processing_results.items():
            if results['success']:
                print(f"  {project}: {len(results['zip_files'])} ZIP files created")
            else:
                print(f"  {project}: FAILED - {results['errors']}")
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return False
    
    return True


def main():
    """Run all examples."""
    print("Code Property Graph Package - Basic Usage Examples")
    print("=" * 60)
    
    examples = [
        ("Basic CPG Generation", example_basic_cpg_generation),
        ("CPG Processing", example_cpg_processing),
        ("CPG Analysis", example_cpg_analysis),
        ("Batch Processing", example_batch_processing)
    ]
    
    results = []
    for name, example_func in examples:
        try:
            success = example_func()
            results.append((name, success))
        except Exception as e:
            print(f"Example '{name}' failed with exception: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Example Execution Summary:")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name:25} {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} examples passed")
    
    if passed == total:
        print("All examples completed successfully!")
        return 0
    else:
        print("Some examples failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
