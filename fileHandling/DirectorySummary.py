import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

"""
    Generate a summary of all files in a directory.
    
    What this script does:
    - Walks through all files in a directory (including subdirectories)
    - Counts file types by extension
    - Calculates total size of the directory
    - Calculates size per file type
    - Finds largest and smallest files
    - Outputs summary to console and saves to a txt file
    
    Author :        Martijn Folmer
    Date created :  06-01-2026
"""


def format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_directory_summary(target_dir: str | Path) -> dict:
    """
    Walk through directory and collect file statistics.
    
    Returns:
    - Dictionary containing all summary statistics
    """
    target_dir = Path(target_dir)
    
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {target_dir}")
    
    if not target_dir.is_dir():
        raise ValueError(f"Path is not a directory: {target_dir}")
    
    # Statistics collectors
    file_count_by_ext = defaultdict(int)
    file_size_by_ext = defaultdict(int)
    total_files = 0
    total_dirs = 0
    total_size = 0
    
    largest_file = None
    largest_file_size = 0
    smallest_file = None
    smallest_file_size = float('inf')
    
    all_files = []
    
    # Walk the directory
    for current_root, dirs, files in os.walk(target_dir):
        total_dirs += len(dirs)
        
        for f in files:
            file_path = Path(current_root) / f
            
            try:
                file_size = file_path.stat().st_size
            except (OSError, PermissionError):
                continue
            
            total_files += 1
            total_size += file_size
            
            # Get extension (lowercase, or 'no extension')
            ext = file_path.suffix.lower() if file_path.suffix else '(no extension)'
            
            file_count_by_ext[ext] += 1
            file_size_by_ext[ext] += file_size
            
            # Track largest and smallest
            if file_size > largest_file_size:
                largest_file_size = file_size
                largest_file = file_path
            
            if file_size < smallest_file_size:
                smallest_file_size = file_size
                smallest_file = file_path
            
            all_files.append((file_path, file_size, ext))
    
    # Sort extensions by count (descending)
    sorted_by_count = sorted(file_count_by_ext.items(), key=lambda x: x[1], reverse=True)
    sorted_by_size = sorted(file_size_by_ext.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'target_dir': target_dir,
        'total_files': total_files,
        'total_dirs': total_dirs,
        'total_size': total_size,
        'file_count_by_ext': dict(sorted_by_count),
        'file_size_by_ext': dict(sorted_by_size),
        'largest_file': largest_file,
        'largest_file_size': largest_file_size,
        'smallest_file': smallest_file,
        'smallest_file_size': smallest_file_size if smallest_file_size != float('inf') else 0,
        'all_files': all_files,
    }


def generate_summary_text(summary: dict) -> str:
    """Generate a formatted text summary from the statistics dictionary."""
    lines = []
    
    lines.append("=" * 70)
    lines.append("DIRECTORY SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Directory: {summary['target_dir'].absolute()}")
    lines.append("")
    
    # Overview
    lines.append("-" * 70)
    lines.append("OVERVIEW")
    lines.append("-" * 70)
    lines.append(f"Total files:       {summary['total_files']:,}")
    lines.append(f"Total directories: {summary['total_dirs']:,}")
    lines.append(f"Total size:        {format_size(summary['total_size'])} ({summary['total_size']:,} bytes)")
    lines.append("")
    
    # Largest and smallest files
    if summary['largest_file']:
        lines.append(f"Largest file:  {summary['largest_file']}")
        lines.append(f"               {format_size(summary['largest_file_size'])}")
    if summary['smallest_file']:
        lines.append(f"Smallest file: {summary['smallest_file']}")
        lines.append(f"               {format_size(summary['smallest_file_size'])}")
    lines.append("")
    
    # File count by extension
    lines.append("-" * 70)
    lines.append("FILE COUNT BY EXTENSION")
    lines.append("-" * 70)
    lines.append(f"{'Extension':<20} {'Count':>10} {'Percentage':>12}")
    lines.append("-" * 42)
    
    for ext, count in summary['file_count_by_ext'].items():
        percentage = (count / summary['total_files'] * 100) if summary['total_files'] > 0 else 0
        lines.append(f"{ext:<20} {count:>10,} {percentage:>11.1f}%")
    lines.append("")
    
    # File size by extension
    lines.append("-" * 70)
    lines.append("FILE SIZE BY EXTENSION")
    lines.append("-" * 70)
    lines.append(f"{'Extension':<20} {'Size':>15} {'Percentage':>12}")
    lines.append("-" * 47)
    
    for ext, size in summary['file_size_by_ext'].items():
        percentage = (size / summary['total_size'] * 100) if summary['total_size'] > 0 else 0
        lines.append(f"{ext:<20} {format_size(size):>15} {percentage:>11.1f}%")
    lines.append("")
    
    lines.append("=" * 70)
    lines.append("END OF SUMMARY")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def summarize_directory(
    target_dir: str | Path,
    output_file: str | Path | None = None,
    print_summary: bool = True,
) -> dict:
    """
    Generate and output a directory summary.
    
    Parameters:
    - target_dir: Directory to analyze
    - output_file: Path to save the summary txt file (None to skip saving)
    - print_summary: Whether to print the summary to console
    
    Returns:
    - Dictionary with all collected statistics
    """
    # Collect statistics
    summary = get_directory_summary(target_dir)
    
    # Generate text
    summary_text = generate_summary_text(summary)
    
    # Print if requested
    if print_summary:
        print(summary_text)
    
    # Save to file if requested
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(summary_text, encoding='utf-8')
        print(f"\nSummary saved to: {output_file.absolute()}")
    
    return summary


if __name__ == "__main__":
    
    TARGET_DIR = "."                            # Directory to analyze
    OUTPUT_FILE = "directory_summary.txt"       # Where to save the summary (set to None to skip)
    PRINT_SUMMARY = True                        # Whether to print to console
    
    summarize_directory(
        target_dir=TARGET_DIR,
        output_file=OUTPUT_FILE,
        print_summary=PRINT_SUMMARY,
    )



