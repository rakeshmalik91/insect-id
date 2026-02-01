import os
import json

NOTEBOOKS_DIR = r"D:\Projects\insect-id\notebooks"
EXCLUDE_DIRS = ["utils"]

# Strings to remove
TARGET_STRINGS = [
    'os.chdir("D:/Projects/insect-id")',
    "os.chdir('D:/Projects/insect-id')",
    'os.chdir("d:/Projects/insect-id")',
    "os.chdir('d:/Projects/insect-id')",
]

# Cell to insert
cell_source = [
    "import os\n",
    "import sys\n",
    "\n",
    "# Standardize working directory to project root\n",
    "try:\n",
    "    if os.path.basename(os.getcwd()) in ['modeling', 'scraping', 'data-prep', 'analysis', 'setup', 'misc']:\n",
    "        os.chdir('../..')\n",
    "    elif os.path.basename(os.getcwd()) == 'notebooks':\n",
    "        os.chdir('..')\n",
    "        \n",
    "    if os.getcwd() not in sys.path:\n",
    "        sys.path.append(os.getcwd())\n",
    "        \n",
    "    print(f\"Working Directory set to: {os.getcwd()}\")\n",
    "except Exception as e:\n",
    "    print(f\"Failed to set working directory: {e}\")"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "antigravity-path-fix",
    "metadata": {},
    "outputs": [],
    "source": cell_source
}

processed_count = 0

for root, dirs, files in os.walk(NOTEBOOKS_DIR):
    # Skip excluded directories
    is_excluded = False
    for ex in EXCLUDE_DIRS:
        if ex in os.path.basename(root) or f"\\{ex}\\" in root:
             is_excluded = True
    if is_excluded:
        continue
        
    for file in files:
        if file.endswith(".ipynb"):
            path = os.path.join(root, file)
            file_modified = False
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 1. Remove hardcoded paths
                if 'cells' in data:
                    for cell in data['cells']:
                        if cell['cell_type'] == 'code':
                            new_source = []
                            source_changed = False
                            for line in cell.get('source', []):
                                stripped_line = line.strip()
                                is_target = False
                                for target in TARGET_STRINGS:
                                    if target in stripped_line:
                                        is_target = True
                                        break
                                
                                if is_target:
                                    print(f"Removing '{stripped_line}' from {file}")
                                    source_changed = True
                                    continue 
                                
                                new_source.append(line)
                            
                            if source_changed:
                                cell['source'] = new_source
                                file_modified = True

                # 2. Add path fix cell if missing
                path_fixed = False
                if data.get('cells') and len(data['cells']) > 0:
                    first_cell = data['cells'][0]
                    # Check ID
                    if first_cell.get('id') == "antigravity-path-fix":
                        path_fixed = True
                    # Check Source Content
                    source_str = "".join(first_cell.get('source', []))
                    if "Standardize working directory" in source_str:
                         path_fixed = True

                if not path_fixed:
                    print(f"Patching {file} with standardized path setup...")
                    data['cells'].insert(0, new_cell)
                    file_modified = True

                # Save if any changes
                if file_modified:
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=1)
                    processed_count += 1
                    
            except Exception as e:
                print(f"Error processing {file}: {e}")

print(f"Standardization Complete. Modified {processed_count} notebooks.")
