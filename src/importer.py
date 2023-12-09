def import_library(library_name):
    try:
        import importlib
        importlib.import_module(library_name)
    except ImportError:
        print("The " + library_name + " library is not installed. Installing...")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])
            print("Library " + library_name + " installed successfully!")
        except Exception as e:
            print("Error installing the library:", e)
            
    # return the library so i can use it in a variable like pd = import_library("pandas")
    return __import__(library_name)