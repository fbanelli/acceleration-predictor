def import_library(library_name):
    try:
        import importlib
        importlib.import_module(library_name)
    except ImportError:
        print("The " + library_name + " library is not installed. Installing...")
        try:
            # if the libary name is a subpackage, install the parent package
            if "." in library_name:
                library_name_install = library_name.split(".")[0]
            else:
                library_name_install = library_name

            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", library_name_install])
            print("Library " +  library_name_install+ " installed successfully!")
        except Exception as e:
            print("Error installing the library:", e)
            
    if library_name == "scipy.interpolate.interp1d":
        from scipy.interpolate import interp1d
        return interp1d
    else:
        return __import__(library_name)