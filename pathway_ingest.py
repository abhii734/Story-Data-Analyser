import os
import platform

# Check OS to avoid crashing on Windows when importing pathway
if os.name == 'nt' or platform.system() == 'Windows':
    # print("Detected Windows. Using 'mock_pathway' shim.")
    import mock_pathway as pw
else:
    try:
        import pathway as pw
    except ImportError:
        print("Warning: 'pathway' import failed. Using 'mock_pathway'.")
        import mock_pathway as pw

def ingest_books(books_dir):
    """
    Ingests book text files from the specified directory.
    Returns a dictionary mapping book title (filename stem) to full text.
    """
    print(f"Ingesting books from: {books_dir}")
    
    # Use Pathway to read the files
    # format="binary" because we want the raw text content
    files = pw.io.fs.read(books_dir, format="binary", mode="static", with_metadata=True)
    
    # Extract to memory (Mock simulation of materialization)
    books_map = {}
    
    # Handle Mock Table
    if hasattr(files, 'data'):
         for row in files.data:
            path_val = str(row.get('path', ''))
            content_val = row.get('data') or row.get('text')
            
            if isinstance(content_val, bytes):
                try:
                    content_val = content_val.decode('utf-8')
                except Exception:
                    content_val = str(content_val)
            
            # Extract basic filename as key (e.g., "The Count of Monte Cristo")
            filename = os.path.basename(path_val)
            book_name = os.path.splitext(filename)[0]
            books_map[book_name] = content_val
            
    return books_map

def ingest_csv(csv_path):
    """
    Ingests a CSV file.
    Returns a list of dictionaries representing the rows.
    """
    print(f"Ingesting CSV: {csv_path}")
    
    # Use Pathway to read CSV
    table = pw.io.csv.read(csv_path, mode="static")
    
    rows = []
    # Handle Mock Table
    if hasattr(table, 'data'):
        # Mock V2: data is a list of dicts
        rows = table.data
    elif hasattr(table, 'df'):
        # Mock V1: pandas df
        rows = table.df.to_dict('records')
        
    return rows

if __name__ == "__main__":
    # Test run
    curr_dir = os.getcwd()
    books = ingest_books(os.path.join(curr_dir, "data", "books"))
    print(f"Loaded {len(books)} books: {list(books.keys())}")
    
    train_data = ingest_csv(os.path.join(curr_dir, "data", "train.csv"))
    print(f"Loaded {len(train_data)} train rows.")

