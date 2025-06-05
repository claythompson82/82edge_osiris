import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

errors_found = []
successful_imports = []

def attempt_import(module_name, import_statement=None):
    global errors_found, successful_imports
    if import_statement is None:
        import_statement = f"import {module_name}"
    try:
        exec(import_statement, globals())
        print(f"Successfully executed: {import_statement}")
        successful_imports.append(module_name)
        # Optionally, print version for key libraries
        if module_name == "torch":
            import torch
            print(f"  Torch version: {torch.__version__}")
            print(f"  Torch CUDA available: {torch.cuda.is_available()}")
        elif module_name == "transformers":
            import transformers
            print(f"  Transformers version: {transformers.__version__}")
        elif module_name == "optimum": # Check parent for optimum.onnxruntime
            import optimum
            print(f"  Optimum version: {optimum.__version__}")
    except Exception as e:
        print(f"Error: Failed to execute '{import_statement}': {e}")
        errors_found.append(f"Failed: {import_statement} -> {e}")

# Attempting imports as requested
attempt_import("torch")
attempt_import("transformers", "from transformers import AutoTokenizer")
attempt_import("optimum", "from optimum.onnxruntime import ORTModelForCausalLM") # Check parent ns for version
attempt_import("osiris")
# osiris.scripts.harvest_feedback will be implicitly tested by osiris import if __init__ structure is correct
# and osiris/__init__.py tries to load it or makes it available.
# A more direct test will occur after checking osiris/scripts/__init__.py.

print("\n--- Summary ---")
if not errors_found:
    print("All key imports verified successfully!")
else:
    print("Found errors during import verification:")
    for err in errors_found:
        print(err)

# Check for osiris.scripts.harvest_feedback specifically if osiris imported
if "osiris" in successful_imports:
    print("\nAttempting direct import of osiris.scripts.harvest_feedback...")
    attempt_import("osiris.scripts.harvest_feedback")
    if "osiris.scripts.harvest_feedback" in successful_imports:
         print("Successfully imported osiris.scripts.harvest_feedback directly.")
    else:
         print("Failed to import osiris.scripts.harvest_feedback directly (see errors above if any).")
