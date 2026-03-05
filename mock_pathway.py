import os
import glob

# Mocking Pathway's basic components without Pandas

class Schema:
    pass

class This:
    """Mock for pw.this"""
    def __init__(self, name=None):
        self._name = name

    def __getattr__(self, name):
        if self._name:
            return This(f"{self._name}.{name}")
        return This(name)
        
    def __str__(self):
        return self._name or ""
        
    def __eq__(self, other):
        return ("eq", self, other)

this = This()

class MockTable:
    def __init__(self, data: list):
        self.data = data # List of dicts

    def select(self, **kwargs):
        """
        Simulates pw.Table.select()
        """
        new_data = []
        for row in self.data:
            new_row = {}
            for new_col, expression in kwargs.items():
                # Naive evaluation: assume expression is pw.this.colName or similar
                # Just extract the last part of the name
                col_name = str(expression).split('.')[-1]
                
                # Check directly
                val = row.get(col_name)
                
                # Fallback hacks
                if val is None and col_name == "data":
                    val = row.get("text")
                
                new_row[new_col] = val
            new_data.append(new_row)
            
        return MockTable(new_data)

    def filter(self, *args):
        return self 

class MockIOContext:
    class fs:
        @staticmethod
        def read(path, format="binary", mode="static", with_metadata=True, **kwargs):
            files = glob.glob(os.path.join(path, "*.txt"))
            data_list = []
            
            for file_path in files:
                abs_path = os.path.abspath(file_path)
                with open(abs_path, 'rb') as f:
                    content_bytes = f.read()
                    
                data_list.append({
                    "data": content_bytes.decode('utf-8'), 
                    "path": abs_path,
                    "_metadata": {"path": abs_path},
                    "text": content_bytes.decode('utf-8') # Helper duplicate
                })
                
            return MockTable(data_list)

    class csv:
        @staticmethod
        def read(path, mode="static", schema=None, **kwargs):
             import csv
             data_list = []
             with open(path, 'r', encoding='utf-8') as f:
                 reader = csv.DictReader(f)
                 for row in reader:
                     data_list.append(row)
             return MockTable(data_list)

        @staticmethod
        def write(table: MockTable, filename: str):
            # Write plain CSV logic
            if not table.data:
                return
            keys = table.data[0].keys()
            import csv
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(table.data)
            print(f"[Mock] Written to {filename}")

io = MockIOContext()
debug = None

def run(eval_strategy=None):
    print("[Mock] Pipeline run simulated.")
