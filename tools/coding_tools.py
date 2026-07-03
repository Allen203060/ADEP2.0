import os
import sys
import subprocess
import traceback
import io

from utils.shared_environment import SHARED_GLOBALS

IMPORT_TO_PIP = {
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "jinja2": "Jinja2",
    "bs4": "beautifulsoup4",
    "sqlalchemy": "SQLAlchemy",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
}

def install_package(package_name: str) -> bool:
    """Installs a package using pip and appends it to requirements.txt."""
    # Find the pip binary in the virtual environment
    pip_path = os.path.join("venv", "bin", "pip")
    if not os.path.exists(pip_path):
        # Fallback to sys.executable's folder
        pip_path = os.path.join(os.path.dirname(sys.executable), "pip")
    if not os.path.exists(pip_path):
        pip_path = "pip"

    print(f"[SYSTEM] 📦 Installing missing dependency: {package_name}...")
    try:
        subprocess.run([pip_path, "install", package_name], check=True)
        
        # Append to requirements.txt if not already present
        req_file = "requirements.txt"
        existing = set()
        if os.path.exists(req_file):
            with open(req_file, "r") as f:
                for line in f:
                    name = line.strip().split("==")[0].split(">=")[0].strip().lower()
                    if name:
                        existing.add(name)
                        
        if package_name.lower() not in existing:
            with open(req_file, "a") as f:
                f.write(f"{package_name}\n")
            print(f"[SYSTEM] 📝 Added {package_name} to requirements.txt")
        return True
    except Exception as err:
        print(f"[SYSTEM] ❌ Failed to install package {package_name}: {err}")
        return False

def execute_python_code(code: str) -> str:
    """
    Executes Python code in the local environment.
    Runs in-memory with access to SHARED_GLOBALS.
    If the code raises a ModuleNotFoundError, it attempts to install the missing package and retries.
    """
    max_retries = 3
    for attempt in range(max_retries):
        # Setup captured stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = sys.stdout = io.StringIO()
        redirected_error = sys.stderr = io.StringIO()

        exec_globals = {
            "SHARED_GLOBALS": SHARED_GLOBALS,
            "os": os,
            "sys": sys,
            "pd": sys.modules.get("pandas"),
            "np": sys.modules.get("numpy")
        }

        try:
            # Execute python code string
            exec(code, exec_globals)
            
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            stdout_str = redirected_output.getvalue()
            stderr_str = redirected_error.getvalue()
            
            return f"Execution Success.\nStdout:\n{stdout_str}\nStderr:\n{stderr_str}"

        except ModuleNotFoundError as e:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            missing_module = e.name
            if not missing_module:
                # Extract from error message if e.name is None
                parts = str(e).split("'")
                if len(parts) >= 2:
                    missing_module = parts[1]
            
            if missing_module:
                pip_package = IMPORT_TO_PIP.get(missing_module, missing_module)
                installed = install_package(pip_package)
                if installed:
                    # Retry execution in next loop iteration
                    continue
            
            return f"Execution Failed:\n{traceback.format_exc()}"
            
        except Exception as e:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            return f"Execution Failed:\n{traceback.format_exc()}"
            
    return "Execution Failed: Maximum retries exceeded due to missing dependencies."

def execute_python_subprocess(code: str) -> str:
    """
    Writes code to a temporary file and runs it in a python subprocess.
    Includes the same self-healing dependency installer mechanism.
    """
    temp_file = "temp_worker_script.py"
    max_retries = 3
    
    for attempt in range(max_retries):
        with open(temp_file, "w") as f:
            f.write(code)
            
        python_path = os.path.join("venv", "bin", "python")
        if not os.path.exists(python_path):
            python_path = sys.executable

        try:
            res = subprocess.run(
                [python_path, temp_file],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            return f"Execution Success.\nStdout:\n{res.stdout}\nStderr:\n{res.stderr}"
            
        except subprocess.CalledProcessError as e:
            # Analyze stderr for ModuleNotFoundError
            err_output = e.stderr
            if "ModuleNotFoundError" in err_output:
                # Find package name
                import re
                match = re.search(r"ModuleNotFoundError: No module named '([^']+)'", err_output)
                if match:
                    missing_module = match.group(1)
                    pip_package = IMPORT_TO_PIP.get(missing_module, missing_module)
                    installed = install_package(pip_package)
                    if installed:
                        continue
            
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return f"Subprocess Execution Failed (exit code {e.returncode}):\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}"
            
        except Exception as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return f"Subprocess Run Error: {e}"

    if os.path.exists(temp_file):
        os.remove(temp_file)
    return "Execution Failed: Maximum retries exceeded due to missing dependencies in subprocess."
