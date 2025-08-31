"""
Code Property Graph processing and export functionality.
"""

import os
import subprocess
import zipfile
import shutil
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .config import CPGConfig
from .utils import load_cpg_from_graphml, save_cpg_to_graphml


class CPGProcessor:
    """
    Class for processing and exporting Code Property Graphs.
    
    Code Reference: gen_graphml.sh (lines 1-59)
    """
    
    def __init__(self, config: Optional[CPGConfig] = None):
        """
        Initialize the CPG processor.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or CPGConfig()
    
    def export_cpg_to_graphml(self, cpg_file_path: str, output_path: Optional[str] = None) -> str:
        """
        Export a CPG binary file to GraphML format.
        
        Args:
            cpg_file_path: Path to the CPG binary file
            output_path: Optional custom output path
            
        Returns:
            Path to the generated GraphML file
            
        Raises:
            FileNotFoundError: If CPG file doesn't exist
            subprocess.CalledProcessError: If joern-export fails
        """
        if not os.path.exists(cpg_file_path):
            raise FileNotFoundError(f"CPG file does not exist: {cpg_file_path}")
        
        if output_path is None:
            # Use same directory as input file
            output_path = cpg_file_path.replace('.bin', '.graphml')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Exporting CPG to GraphML: {cpg_file_path} -> {output_path}")
        
        # Use joern-export to generate GraphML
        subprocess.run([
            self.config.joern_export_cmd,
            cpg_file_path,
            self.config.joern_memory,
            '--repr', 'all',
            '--format', 'graphml',
            '-o', output_path
        ], check=True, capture_output=True)
        
        print(f"Successfully exported CPG to GraphML: {output_path}")
        return output_path
    
    def process_project_cpgs(self, project_name: str, 
                            cpg_directory: Optional[str] = None,
                            output_directory: Optional[str] = None) -> List[str]:
        """
        Process all CPG files for a project, converting them to GraphML.
        
        Args:
            project_name: Name of the project
            cpg_directory: Optional custom CPG directory
            output_directory: Optional custom output directory
            
        Returns:
            List of generated GraphML file paths
        """
        if cpg_directory is None:
            cpg_directory = self.config.get_cpg_output_path(project_name)
        
        if output_directory is None:
            output_directory = cpg_directory
        
        if not os.path.exists(cpg_directory):
            raise FileNotFoundError(f"CPG directory does not exist: {cpg_directory}")
        
        # Create output directory
        os.makedirs(output_directory, exist_ok=True)
        
        generated_files = []
        cpg_files = [f for f in os.listdir(cpg_directory) if f.endswith('.bin')]
        
        print(f"Processing {len(cpg_files)} CPG files for project {project_name}")
        
        for cpg_file in cpg_files:
            try:
                cpg_path = os.path.join(cpg_directory, cpg_file)
                commit_id = cpg_file.replace('.bin', '')
                graphml_path = os.path.join(output_directory, f"{commit_id}.graphml")
                
                # Export to GraphML
                exported_path = self.export_cpg_to_graphml(cpg_path, graphml_path)
                generated_files.append(exported_path)
                
            except Exception as e:
                print(f"Error processing CPG file {cpg_file}: {e}")
                continue
        
        print(f"Successfully processed {len(generated_files)} CPG files for project {project_name}")
        return generated_files
    
    def compress_graphml_files(self, graphml_directory: str, 
                              output_directory: Optional[str] = None,
                              remove_original: bool = False) -> List[str]:
        """
        Compress GraphML files into ZIP archives.
        
        Args:
            graphml_directory: Directory containing GraphML files
            output_directory: Optional custom output directory
            remove_original: Whether to remove original GraphML files after compression
            
        Returns:
            List of generated ZIP file paths
        """
        if not os.path.exists(graphml_directory):
            raise FileNotFoundError(f"GraphML directory does not exist: {graphml_directory}")
        
        if output_directory is None:
            output_directory = graphml_directory
        
        os.makedirs(output_directory, exist_ok=True)
        
        compressed_files = []
        graphml_files = [f for f in os.listdir(graphml_directory) if f.endswith('.graphml')]
        
        print(f"Compressing {len(graphml_files)} GraphML files")
        
        for graphml_file in graphml_files:
            try:
                graphml_path = os.path.join(graphml_directory, graphml_file)
                commit_id = graphml_file.replace('.graphml', '')
                zip_path = os.path.join(output_directory, f"{commit_id}.graphml.zip")
                
                # Create ZIP archive
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(graphml_path, graphml_file)
                
                compressed_files.append(zip_path)
                
                # Remove original if requested
                if remove_original:
                    os.remove(graphml_path)
                    print(f"Removed original file: {graphml_file}")
                
            except Exception as e:
                print(f"Error compressing {graphml_file}: {e}")
                continue
        
        print(f"Successfully compressed {len(compressed_files)} GraphML files")
        return compressed_files
    
    def process_and_compress_project(self, project_name: str) -> Tuple[List[str], List[str]]:
        """
        Process CPG files and compress GraphML files for a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Tuple of (GraphML files, ZIP files)
        """
        # First, export CPGs to GraphML
        graphml_files = self.process_project_cpgs(project_name)
        
        # Then compress GraphML files
        zip_files = self.compress_graphml_files(
            os.path.dirname(graphml_files[0]) if graphml_files else "",
            remove_original=True
        )
        
        return graphml_files, zip_files
    
    def upload_to_cloud_storage(self, local_files: List[str], 
                               project_name: str,
                               bucket_name: Optional[str] = None) -> List[str]:
        """
        Upload files to cloud storage (Google Cloud Storage).
        
        Args:
            local_files: List of local file paths to upload
            project_name: Name of the project (used for cloud path)
            bucket_name: Optional custom bucket name
            
        Returns:
            List of uploaded file URLs
        """
        if bucket_name is None:
            bucket_name = self.config.bucket_name
        
        # Check if gsutil is available
        try:
            subprocess.run(['gsutil', 'version'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("gsutil is not available. Please install Google Cloud SDK.")
        
        uploaded_files = []
        
        for local_file in local_files:
            try:
                if not os.path.exists(local_file):
                    print(f"File does not exist: {local_file}")
                    continue
                
                filename = os.path.basename(local_file)
                cloud_path = f"gs://{bucket_name}/{project_name}/{filename}"
                
                print(f"Uploading {local_file} to {cloud_path}")
                
                subprocess.run([
                    'gsutil', 'cp', local_file, cloud_path
                ], check=True, capture_output=True)
                
                uploaded_files.append(cloud_path)
                print(f"Successfully uploaded: {cloud_path}")
                
            except Exception as e:
                print(f"Error uploading {local_file}: {e}")
                continue
        
        return uploaded_files
    
    def process_project_with_cloud_upload(self, project_name: str,
                                        upload_to_cloud: bool = True) -> Dict[str, any]:
        """
        Complete workflow: process CPGs, compress, and optionally upload to cloud.
        
        Args:
            project_name: Name of the project
            upload_to_cloud: Whether to upload to cloud storage
            
        Returns:
            Dictionary containing processing results
        """
        results = {
            'project_name': project_name,
            'graphml_files': [],
            'zip_files': [],
            'uploaded_files': [],
            'success': True,
            'errors': []
        }
        
        try:
            # Process CPGs to GraphML
            print(f"Processing CPGs for project {project_name}")
            graphml_files, zip_files = self.process_and_compress_project(project_name)
            results['graphml_files'] = graphml_files
            results['zip_files'] = zip_files
            
            # Upload to cloud if requested
            if upload_to_cloud and zip_files:
                print(f"Uploading files to cloud for project {project_name}")
                uploaded_files = self.upload_to_cloud_storage(zip_files, project_name)
                results['uploaded_files'] = uploaded_files
            
        except Exception as e:
            results['success'] = False
            results['errors'].append(str(e))
            print(f"Error processing project {project_name}: {e}")
        
        return results
    
    def batch_process_projects(self, project_names: List[str],
                              upload_to_cloud: bool = True,
                              parallel: bool = True) -> Dict[str, Dict[str, any]]:
        """
        Process multiple projects in batch.
        
        Args:
            project_names: List of project names to process
            upload_to_cloud: Whether to upload to cloud storage
            parallel: Whether to process projects in parallel
            
        Returns:
            Dictionary mapping project names to processing results
        """
        if parallel:
            return self._batch_process_parallel(project_names, upload_to_cloud)
        else:
            return self._batch_process_sequential(project_names, upload_to_cloud)
    
    def _batch_process_parallel(self, project_names: List[str], 
                               upload_to_cloud: bool) -> Dict[str, Dict[str, any]]:
        """Process projects in parallel."""
        import concurrent.futures
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_project = {
                executor.submit(self.process_project_with_cloud_upload, project_name, upload_to_cloud): project_name
                for project_name in project_names
            }
            
            for future in concurrent.futures.as_completed(future_to_project):
                project_name = future_to_project[future]
                try:
                    project_results = future.result()
                    results[project_name] = project_results
                except Exception as e:
                    results[project_name] = {
                        'project_name': project_name,
                        'success': False,
                        'errors': [str(e)]
                    }
        
        return results
    
    def _batch_process_sequential(self, project_names: List[str], 
                                 upload_to_cloud: bool) -> Dict[str, Dict[str, any]]:
        """Process projects sequentially."""
        results = {}
        
        for project_name in project_names:
            try:
                project_results = self.process_project_with_cloud_upload(project_name, upload_to_cloud)
                results[project_name] = project_results
            except Exception as e:
                results[project_name] = {
                    'project_name': project_name,
                    'success': False,
                    'errors': [str(e)]
                }
        
        return results
    
    def cleanup_temp_files(self, project_name: str, 
                          keep_graphml: bool = False,
                          keep_zip: bool = True) -> List[str]:
        """
        Clean up temporary files after processing.
        
        Args:
            project_name: Name of the project
            keep_graphml: Whether to keep GraphML files
            keep_zip: Whether to keep ZIP files
            
        Returns:
            List of cleaned up file paths
        """
        cpg_directory = self.config.get_cpg_output_path(project_name)
        if not os.path.exists(cpg_directory):
            return []
        
        cleaned_files = []
        
        for file in os.listdir(cpg_directory):
            file_path = os.path.join(cpg_directory, file)
            
            try:
                if file.endswith('.graphml') and not keep_graphml:
                    os.remove(file_path)
                    cleaned_files.append(file_path)
                    print(f"Cleaned up GraphML file: {file}")
                elif file.endswith('.zip') and not keep_zip:
                    os.remove(file_path)
                    cleaned_files.append(file_path)
                    print(f"Cleaned up ZIP file: {file}")
            except Exception as e:
                print(f"Error cleaning up {file}: {e}")
        
        return cleaned_files
