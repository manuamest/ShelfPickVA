import json
import os

CONFIG_FILE = 'bottle_config.json'

def repair_config():
    if not os.path.exists(CONFIG_FILE):
        print("Config file not found.")
        return

    with open(CONFIG_FILE, 'r') as f:
        content = f.read()
    
    print(f"Original length: {len(content)}")
    print(f"Tail: {content[-50:]}")
    
    # Try to parse to see error
    try:
        json.loads(content)
        print("JSON is valid.")
        return
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}")
    
    # Attempt simple repair by closing brackets
    # The tail looked like numbers in a list, so we likely need ]}]
    # But let's be safer.
    
    # Trim whitespace from end
    content = content.rstrip()
    
    # If it ends with a comma, remove it
    if content.endswith(','):
        content = content[:-1]
        
    # Try appending closing sequences
    closers = [
        "}",
        "]}",
        "}]}",
        "]}]}"
    ]
    
    repaired = False
    for closer in closers:
        try:
            test_content = content + closer
            json.loads(test_content)
            print(f"Fixed with closer: {closer}")
            with open(CONFIG_FILE, 'w') as f:
                f.write(test_content)
            repaired = True
            break
        except json.JSONDecodeError:
            continue
            
    if repaired:
        print("File repaired successfully.")
    else:
        print("Could not auto-repair. Manual inspection needed.")

if __name__ == "__main__":
    repair_config()
