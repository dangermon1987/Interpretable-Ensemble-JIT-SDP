"""
Basic usage examples for the node embeddings package.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from node_embeddings import (
    EmbeddingGenerator, 
    EmbeddingConfig,
    Word2VecEmbedder,
    Node2VecEmbedder,
    CodeBERTEmbedder
)


def example_basic_embedding_generation():
    """Example: Basic embedding generation for a single commit."""
    print("=== Basic Embedding Generation Example ===")
    
    # Initialize with default configuration
    generator = EmbeddingGenerator()
    
    # Example file paths (replace with actual paths)
    before_file = "path/to/before.graphml"
    after_file = "path/to/after.graphml"
    
    # Check if files exist
    if not os.path.exists(before_file) or not os.path.exists(after_file):
        print(f"Example files not found. Please update the file paths.")
        print(f"Expected: {before_file}, {after_file}")
        return
    
    # Generate embeddings
    result = generator.generate_embeddings_for_commit(before_file, after_file)
    
    if result['status'] == 'success':
        print(f"✅ Successfully generated embeddings for {result['total_nodes_processed']} nodes")
        print(f"📁 Output files:")
        for file_path in result['hetero_data_files']:
            print(f"   - {file_path}")
        print(f"📊 Embedding statistics: {result['embedding_stats']}")
    else:
        print(f"❌ Error: {result['error']}")


def example_custom_configuration():
    """Example: Custom configuration and initialization."""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom configuration
    config = EmbeddingConfig(
        data_dir="/custom/data/path",
        output_dir="/custom/output/path",
        word2vec_vector_size=256,
        node2vec_dimensions=256,
        codebert_batch_size=16,
        device="cuda",
        max_workers=12
    )
    
    print(f"Custom configuration created:")
    print(f"  Data directory: {config.data_dir}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Word2Vec vector size: {config.word2vec_vector_size}")
    print(f"  Node2Vec dimensions: {config.node2vec_dimensions}")
    print(f"  CodeBERT batch size: {config.codebert_batch_size}")
    print(f"  Device: {config.device}")
    print(f"  Max workers: {config.max_workers}")
    
    # Initialize generator with custom config
    generator = EmbeddingGenerator(config)
    
    # Validate setup
    validation = generator.validate_setup()
    print(f"\nSetup validation: {'✅ Valid' if validation['is_valid'] else '❌ Invalid'}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")


def example_project_processing():
    """Example: Processing an entire project."""
    print("\n=== Project Processing Example ===")
    
    # Initialize generator
    generator = EmbeddingGenerator()
    
    # Process a specific project
    project_name = "commons-codec"
    
    print(f"Processing project: {project_name}")
    result = generator.generate_embeddings_for_project(project_name)
    
    print(f"Project {result['project']} completed:")
    print(f"  Total files: {result['total_files']}")
    print(f"  Successful: {result['successful']}")
    print(f"  Failed: {result['failed']}")
    print(f"  Status: {result['status']}")


def example_multi_project_processing():
    """Example: Processing multiple projects in parallel."""
    print("\n=== Multi-Project Processing Example ===")
    
    # Initialize generator
    generator = EmbeddingGenerator()
    
    # Define projects to process
    projects = ["commons-codec", "commons-lang", "commons-math"]
    
    print(f"Processing {len(projects)} projects in parallel: {projects}")
    
    # Process projects in parallel
    all_results = generator.generate_embeddings_for_projects(
        project_names=projects,
        parallel=True
    )
    
    print(f"\nAll projects completed:")
    for project, result in all_results['results'].items():
        status_icon = "✅" if result['status'] == 'completed' else "❌"
        print(f"  {status_icon} {project}: {result['status']}")


def example_individual_embedders():
    """Example: Using individual embedder classes."""
    print("\n=== Individual Embedders Example ===")
    
    # Initialize individual embedders
    word2vec = Word2VecEmbedder()
    node2vec = Node2VecEmbedder()
    codebert = CodeBERTEmbedder()
    
    print("Individual embedders initialized:")
    print(f"  Word2Vec: {word2vec.get_model_info()}")
    print(f"  Node2Vec: {node2vec.get_model_info()}")
    print(f"  CodeBERT: {codebert.get_model_info()}")


def example_word2vec_training():
    """Example: Training Word2Vec model."""
    print("\n=== Word2Vec Training Example ===")
    
    # Initialize generator
    generator = EmbeddingGenerator()
    
    # Example file paths for training
    file_paths = ["project1_commit1", "project1_commit2", "project2_commit1"]
    
    print(f"Training Word2Vec model on {len(file_paths)} files...")
    
    try:
        # Train the model
        model_path = generator.train_word2vec_model(file_paths)
        print(f"✅ Word2Vec model trained and saved to: {model_path}")
        
        # Load the model
        generator.load_word2vec_model(model_path)
        print("✅ Model loaded successfully")
        
        # Get model information
        model_info = generator.word2vec_embedder.get_model_info()
        print(f"📊 Model info: {model_info}")
        
    except Exception as e:
        print(f"❌ Error training Word2Vec model: {e}")


def example_embedding_analysis():
    """Example: Analyzing generated embeddings."""
    print("\n=== Embedding Analysis Example ===")
    
    # Initialize generator
    generator = EmbeddingGenerator()
    
    # Example: Get embedding summary
    summary = generator.get_embedding_summary()
    
    print("Embedding summary:")
    for component, info in summary.items():
        print(f"  {component}: {info}")


def example_validation_and_quality():
    """Example: Validation and quality checks."""
    print("\n=== Validation and Quality Example ===")
    
    # Initialize generator
    generator = EmbeddingGenerator()
    
    # Validate setup
    print("Validating setup...")
    validation = generator.validate_setup()
    
    if validation['is_valid']:
        print("✅ Setup is valid")
        for check, status in validation['checks'].items():
            print(f"  {check}: {status}")
    else:
        print("❌ Setup validation failed:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")


def main():
    """Run all examples."""
    print("🚀 Node Embeddings Package - Basic Usage Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_embedding_generation()
        example_custom_configuration()
        example_project_processing()
        example_multi_project_processing()
        example_individual_embedders()
        example_word2vec_training()
        example_embedding_analysis()
        example_validation_and_quality()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\n💡 Tips:")
        print("  - Update file paths in examples to match your environment")
        print("  - Check the README.md for detailed usage instructions")
        print("  - Use the package in your own scripts and notebooks")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
