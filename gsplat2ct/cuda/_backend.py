"""
Backend selection logic (CUDA vs Torch fallback)

This module handles loading the compiled CUDA extension and provides
JIT compilation fallback if the extension is not pre-compiled.

Following gsplat's robust JIT compilation pattern with TRUE lazy loading:
- The CUDA extension is NOT loaded at module import time
- Loading/compilation only happens when _C is first accessed
- This allows `import gsplat2ct` to be fast even without CUDA

Trigger compiling (for debugging):
    VERBOSE=1 FAST_COMPILE=1 python -c "from gsplat2ct.cuda._backend import _C"

Author: Dy164
Date: 2025-12-03
Updated: 2025-12-16 (True lazy loading implementation)
"""

import glob
import os
import shutil
import time
from subprocess import DEVNULL, call
from typing import Optional

import torch
from rich.console import Console

PATH = os.path.dirname(os.path.abspath(__file__))
NO_FAST_MATH = os.getenv("NO_FAST_MATH", "0") == "1"
FAST_COMPILE = os.getenv("FAST_COMPILE", "0") == "1"
VERBOSE = os.getenv("VERBOSE", "0") == "1"
MAX_JOBS = os.getenv("MAX_JOBS")


def _set_max_jobs():
    """Set MAX_JOBS environment variable if not already set."""
    if not os.getenv("MAX_JOBS"):
        os.environ["MAX_JOBS"] = "10"


def cuda_toolkit_available() -> bool:
    """
    Check if the CUDA toolkit is available WITHOUT triggering compilation.
    
    This function only checks if nvcc exists, it does NOT try to compile.
    """
    try:
        from torch.utils.cpp_extension import _find_cuda_home
        
        cuda_home = _find_cuda_home()
        if not cuda_home:
            return False

        nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
        if not os.path.isfile(nvcc_path):
            try:
                call(["nvcc", "--version"], stdout=DEVNULL, stderr=DEVNULL)
                return True
            except FileNotFoundError:
                return False
        return True
    except Exception:
        return False


def _load_extension(
    name,
    sources,
    extra_cflags=None,
    extra_cuda_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    build_directory=None,
    verbose=False,
):
    """Load a JIT compiled extension."""
    from torch.utils.cpp_extension import (
        _get_build_directory,
        _import_module_from_library,
        _jit_compile,
    )
    
    if build_directory:
        os.makedirs(build_directory, exist_ok=True)

    try:
        try:
            compiled = _jit_compile(
                name,
                sources,
                extra_cflags,
                extra_cuda_cflags,
                extra_ldflags,
                extra_include_paths,
                build_directory,
                verbose,
                with_cuda=None,
                is_python_module=True,
                is_standalone=False,
                keep_intermediates=True,
            )
        except TypeError as e:
            # torch>=2.7.0 added SYCL support with new arguments
            if "_jit_compile() missing" in str(e):
                compiled = _jit_compile(
                    name,
                    sources,
                    extra_cflags,
                    extra_cuda_cflags,
                    None,  # extra_ldflags for SYCL
                    extra_ldflags,
                    extra_include_paths,
                    build_directory,
                    verbose,
                    with_cuda=None,
                    with_sycl=None,
                    is_python_module=True,
                    is_standalone=False,
                    keep_intermediates=True,
                )
            else:
                raise e
        return compiled
    except OSError:
        return _import_module_from_library(name, build_directory, True)


# ============================================================================
# Lazy Loading Implementation
# ============================================================================

class _LazyModule:
    """
    Lazy loader for CUDA extension.
    
    The extension is only loaded when first accessed, not at import time.
    This allows:
    - Fast `import gsplat2ct` even without CUDA
    - Automatic fallback to Torch backend when CUDA is not available
    - JIT compilation only when CUDA functions are actually needed
    """
    
    def __init__(self):
        self._module = None
        self._loaded = False
        self._load_attempted = False
    
    def _try_load(self):
        """Attempt to load the CUDA extension (called on first access)."""
        if self._load_attempted:
            return
        
        self._load_attempted = True
        _set_max_jobs()
        
        # Step 1: Try pre-compiled module
        try:
            from gsplat2ct import csrc
            self._module = csrc
            self._loaded = True
            return
        except ImportError:
            pass
        
        # Step 2: Try JIT compilation if CUDA toolkit is available
        if not cuda_toolkit_available():
            Console().print(
                "[yellow]gsplat2ct: No CUDA toolkit found. CUDA extension disabled.[/yellow]"
            )
            return
        
        # Step 3: JIT compile
        try:
            from torch.utils.cpp_extension import _get_build_directory
            
            name = "gsplat2ct_cuda"
            build_dir = _get_build_directory(name, verbose=False)
            
            extra_include_paths = [os.path.join(PATH, "include/")]
            opt_level = "-O0" if FAST_COMPILE else "-O3"
            extra_cflags = [opt_level, "-std=c++17"]
            extra_cuda_cflags = [opt_level, "-std=c++17"]
            if not NO_FAST_MATH:
                extra_cuda_cflags += ["-use_fast_math"]
            
            sources = (
                list(glob.glob(os.path.join(PATH, "csrc/*.cu")))
                + list(glob.glob(os.path.join(PATH, "csrc/*.cpp")))
            )
            
            # Clean up lock file if exists
            try:
                os.remove(os.path.join(build_dir, "lock"))
            except OSError:
                pass
            
            # Check if already built
            already_built = (
                os.path.exists(os.path.join(build_dir, f"{name}.so")) or
                os.path.exists(os.path.join(build_dir, f"{name}.lib"))
            )
            
            if already_built:
                self._module = _load_extension(
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                    extra_include_paths=extra_include_paths,
                    build_directory=build_dir,
                    verbose=VERBOSE,
                )
            else:
                # Build from scratch
                if os.path.exists(build_dir):
                    shutil.rmtree(build_dir)
                
                tic = time.time()
                with Console().status(
                    f"[bold yellow]gsplat2ct: Setting up CUDA with MAX_JOBS={os.environ.get('MAX_JOBS', '10')} "
                    "(This may take a few minutes the first time)",
                    spinner="bouncingBall",
                ):
                    self._module = _load_extension(
                        name=name,
                        sources=sources,
                        extra_cflags=extra_cflags,
                        extra_cuda_cflags=extra_cuda_cflags,
                        extra_include_paths=extra_include_paths,
                        build_directory=build_dir,
                        verbose=VERBOSE,
                    )
                toc = time.time()
                Console().print(
                    f"[green]gsplat2ct: CUDA extension has been set up successfully in {toc - tic:.2f} seconds.[/green]"
                )
            
            self._loaded = True
            
        except Exception as e:
            Console().print(
                f"[red]gsplat2ct: Failed to load CUDA extension: {e}[/red]"
            )
            self._module = None
    
    def __getattr__(self, name):
        """Intercept attribute access to trigger lazy loading."""
        self._try_load()
        if self._module is None:
            raise RuntimeError(
                f"CUDA extension is not available. Cannot access '{name}'. "
                "Use MLPClassificationTorch or MLPClassificationAutograd instead."
            )
        return getattr(self._module, name)
    
    def is_available(self) -> bool:
        """Check if extension is loaded without triggering load."""
        return self._loaded and self._module is not None


# Global lazy module instance
_lazy_C = _LazyModule()


# Create a module-level _C that acts as proxy
class _CProxy:
    """Proxy object that forwards all attribute access to lazy module."""
    
    def __getattr__(self, name):
        return getattr(_lazy_C, name)
    
    def __bool__(self):
        return _lazy_C.is_available()


_C = _CProxy()


def is_cuda_available() -> bool:
    """
    Check if CUDA extension is available.
    
    This function does NOT trigger JIT compilation.
    It only returns True if:
    1. The extension has already been loaded successfully, OR
    2. A pre-compiled module exists
    
    To check if CUDA toolkit exists (without loading), use cuda_toolkit_available().
    """
    # First check if already loaded
    if _lazy_C.is_available():
        return True
    
    # Check if pre-compiled module exists (without loading)
    try:
        import importlib.util
        spec = importlib.util.find_spec("gsplat2ct.csrc")
        if spec is not None:
            return torch.cuda.is_available()
    except Exception:
        pass
    
    # Check if CUDA toolkit available for potential JIT
    return cuda_toolkit_available() and torch.cuda.is_available()


def get_cuda_module():
    """
    Explicitly get the CUDA module, triggering load if necessary.
    
    Returns:
        The loaded CUDA module, or None if not available.
    """
    _lazy_C._try_load()
    return _lazy_C._module


__all__ = ["is_cuda_available", "_C", "cuda_toolkit_available", "get_cuda_module"]
