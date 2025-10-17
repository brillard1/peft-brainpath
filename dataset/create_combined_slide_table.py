"""
Script to combine all category slide tables into a single table for multi-class classification.
"""

import pandas as pd
from pathlib import Path

def create_combined_slide_table():
    slide_tables_dir = Path("slide_tables/categories")
    output_dir = Path("slide_tables/")
    output_dir.mkdir(parents=True, exist_ok=True)

    slide_table_files = list(slide_tables_dir.glob("slide_table_*.csv"))
    
    print(f"Found {len(slide_table_files)} category slide tables")
    
    combined_data = []
    category_counts = {}
    
    for slide_table_file in slide_table_files:
        category = slide_table_file.stem.replace("slide_table_", "")
        df = pd.read_csv(slide_table_file)
        df['CATEGORY'] = category
        
        combined_data.append(df)
        category_counts[category] = len(df)
        
        print(f"Added {category}: {len(df)} slides")
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    output_file = output_dir / "combined_slide_table.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"\nCombined slide table saved to: {output_file}")
    print(f"Total slides: {len(combined_df)}")
    print(f"Total categories: {len(category_counts)}")
    
    print("\nCategory distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")
    
    summary_file = output_dir / "category_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Category Distribution Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total slides: {len(combined_df)}\n")
        f.write(f"Total categories: {len(category_counts)}\n\n")
        f.write("Category distribution:\n")
        for category, count in sorted(category_counts.items()):
            f.write(f"  {category}: {count}\n")
    
    print(f"Category summary saved to: {summary_file}")
    
    return output_file, category_counts

if __name__ == "__main__":
    create_combined_slide_table() 