"""
Command-line interface for the Code Property Graph package.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional

from .generator import CPGGenerator
from .processor import CPGProcessor
from .analyzer import CPGAnalyzer
from .config import CPGConfig, default_config


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Code Property Graph (CPG) Generation and Processing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate CPGs for a single project
  python -m @code_property_graph generate --project commons-beanutils
  
  # Generate CPGs for multiple projects in parallel
  python -m @code_property_graph generate --projects commons-beanutils commons-codec --parallel
  
  # Process existing CPGs to GraphML
  python -m @code_property_graph process --project commons-beanutils
  
  # Analyze a CPG file
  python -m @code_property_graph analyze --file commit_hash.graphml --project commons-beanutils
  
  # Check project status
  python -m @code_property_graph status --project commons-beanutils
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate CPGs from source code')
    gen_parser.add_argument('--project', help='Single project name')
    gen_parser.add_argument('--projects', nargs='+', help='Multiple project names')
    gen_parser.add_argument('--parallel', action='store_true', help='Process projects in parallel')
    gen_parser.add_argument('--config', help='Path to configuration file')
    
    # Process command
    proc_parser = subparsers.add_parser('process', help='Process CPGs to GraphML')
    proc_parser.add_argument('--project', required=True, help='Project name')
    proc_parser.add_argument('--upload', action='store_true', help='Upload to cloud storage')
    proc_parser.add_argument('--compress', action='store_true', help='Compress GraphML files')
    
    # Analyze command
    anal_parser = subparsers.add_parser('analyze', help='Analyze CPG files')
    anal_parser.add_argument('--file', required=True, help='CPG file path (GraphML)')
    anal_parser.add_argument('--project', help='Project name for report')
    anal_parser.add_argument('--output', help='Output report file path')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check project status')
    status_parser.add_argument('--project', help='Project name (or all if not specified)')
    status_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up temporary files')
    cleanup_parser.add_argument('--project', required=True, help='Project name')
    cleanup_parser.add_argument('--failed', action='store_true', help='Clean up failed generations')
    cleanup_parser.add_argument('--temp', action='store_true', help='Clean up temporary files')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show or update configuration')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    config_parser.add_argument('--update', help='Update configuration from JSON file')
    config_parser.add_argument('--env', action='store_true', help='Show environment-based configuration')
    
    return parser


def load_config(config_path: Optional[str]) -> CPGConfig:
    """Load configuration from file or use default."""
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return CPGConfig.from_dict(config_dict)
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {e}")
            print("Using default configuration")
            return default_config
    else:
        return default_config


def handle_generate(args) -> int:
    """Handle the generate command."""
    config = load_config(args.config)
    
    if args.project and args.projects:
        print("Error: Cannot specify both --project and --projects")
        return 1
    
    if not args.project and not args.projects:
        print("Error: Must specify either --project or --projects")
        return 1
    
    generator = CPGGenerator(config)
    
    try:
        if args.project:
            print(f"Generating CPGs for project: {args.project}")
            cpg_files = generator.generate_cpgs_for_project(args.project)
            print(f"Generated {len(cpg_files)} CPG files")
        else:
            print(f"Generating CPGs for projects: {args.projects}")
            results = generator.generate_cpgs_for_projects(
                args.projects, 
                parallel=args.parallel
            )
            
            total_generated = 0
            for project, files in results.items():
                print(f"{project}: {len(files)} CPG files")
                total_generated += len(files)
            
            print(f"Total generated: {total_generated} CPG files")
        
        return 0
        
    except Exception as e:
        print(f"Error during CPG generation: {e}")
        return 1


def handle_process(args) -> int:
    """Handle the process command."""
    config = load_config(None)
    processor = CPGProcessor(config)
    
    try:
        print(f"Processing CPGs for project: {args.project}")
        
        if args.upload:
            results = processor.process_project_with_cloud_upload(args.project)
            if results['success']:
                print(f"Successfully processed and uploaded {len(results['zip_files'])} files")
            else:
                print(f"Processing failed: {results['errors']}")
                return 1
        else:
            graphml_files, zip_files = processor.process_and_compress_project(args.project)
            print(f"Generated {len(graphml_files)} GraphML files")
            print(f"Created {len(zip_files)} ZIP files")
        
        return 0
        
    except Exception as e:
        print(f"Error during CPG processing: {e}")
        return 1


def handle_analyze(args) -> int:
    """Handle the analyze command."""
    config = load_config(None)
    analyzer = CPGAnalyzer(config)
    
    try:
        if not Path(args.file).exists():
            print(f"Error: File does not exist: {args.file}")
            return 1
        
        print(f"Analyzing CPG file: {args.file}")
        graph = analyzer.load_cpg_from_graphml(args.file)
        
        # Generate comprehensive report
        project_name = args.project or "unknown"
        report = analyzer.generate_cpg_report(graph, project_name)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report saved to: {args.output}")
        else:
            # Print summary to console
            print(f"\nCPG Analysis Report for {project_name}")
            print("=" * 50)
            print(f"Total nodes: {report['graph_statistics']['total_nodes']}")
            print(f"Total edges: {report['graph_statistics']['total_edges']}")
            print(f"Methods: {report['graph_statistics']['method_count']}")
            print(f"Classes: {report['graph_statistics']['class_count']}")
            print(f"Variables: {report['graph_statistics']['variable_count']}")
            print(f"Code smells detected: {report['summary']['total_smells']}")
            print(f"High complexity methods: {report['summary']['high_complexity_methods']}")
        
        return 0
        
    except Exception as e:
        print(f"Error during CPG analysis: {e}")
        return 1


def handle_status(args) -> int:
    """Handle the status command."""
    config = load_config(None)
    generator = CPGGenerator(config)
    
    try:
        if args.project:
            status = generator.get_project_status(args.project)
            if args.json:
                print(json.dumps(status, indent=2, default=str))
            else:
                print(f"Status for project: {status['project_name']}")
                print(f"Dataset exists: {status['dataset_exists']}")
                print(f"Output path exists: {status['output_path_exists']}")
                print(f"Total commits: {status['total_commits']}")
                print(f"Generated CPGs: {status['generated_cpgs']}")
                print(f"Missing CPGs: {status['missing_cpgs']}")
        else:
            # Show status for all projects
            all_status = {}
            for project in config.default_projects:
                try:
                    all_status[project] = generator.get_project_status(project)
                except Exception as e:
                    all_status[project] = {'error': str(e)}
            
            if args.json:
                print(json.dumps(all_status, indent=2, default=str))
            else:
                print("Project Status Summary:")
                print("=" * 50)
                for project, status in all_status.items():
                    if 'error' in status:
                        print(f"{project}: ERROR - {status['error']}")
                    else:
                        print(f"{project}: {status['generated_cpgs']}/{status['total_commits']} CPGs")
        
        return 0
        
    except Exception as e:
        print(f"Error checking status: {e}")
        return 1


def handle_cleanup(args) -> int:
    """Handle the cleanup command."""
    config = load_config(None)
    generator = CPGGenerator(config)
    processor = CPGProcessor(config)
    
    try:
        print(f"Cleaning up project: {args.project}")
        
        if args.failed:
            cleaned_files = generator.cleanup_failed_generations(args.project)
            print(f"Cleaned up {len(cleaned_files)} failed generation files")
        
        if args.temp:
            cleaned_files = processor.cleanup_temp_files(args.project)
            print(f"Cleaned up {len(cleaned_files)} temporary files")
        
        if not args.failed and not args.temp:
            print("No cleanup options specified. Use --failed and/or --temp")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return 1


def handle_config(args) -> int:
    """Handle the config command."""
    try:
        if args.show:
            config = default_config
            print("Current Configuration:")
            print(json.dumps(config.to_dict(), indent=2))
        
        elif args.update:
            with open(args.update, 'r') as f:
                config_dict = json.load(f)
            config = CPGConfig.from_dict(config_dict)
            print(f"Configuration updated from: {args.update}")
            print(json.dumps(config.to_dict(), indent=2))
        
        elif args.env:
            config = CPGConfig.from_env()
            print("Environment-based Configuration:")
            print(json.dumps(config.to_dict(), indent=2))
        
        else:
            print("No config action specified. Use --show, --update, or --env")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error handling configuration: {e}")
        return 1


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate handler
    handlers = {
        'generate': handle_generate,
        'process': handle_process,
        'analyze': handle_analyze,
        'status': handle_status,
        'cleanup': handle_cleanup,
        'config': handle_config
    }
    
    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
