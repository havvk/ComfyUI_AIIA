
import inspect
import sys
import torch

try:
    from resemble_enhance.enhancer.enhancer import Enhancer
    print("=== Source of Enhancer.forward ===")
    print(inspect.getsource(Enhancer.forward))
    
    print("\n=== Source of Enhancer (init to check modules) ===")
    # Print first 50 lines of Init to see components
    lines = inspect.getsource(Enhancer.__init__).split('\n')
    for line in lines[:50]:
        print(line)

except ImportError:
    print("Could not import Enhancer. Trying to locate file...")
    import resemble_enhance
    print(f"Resemble Enhance path: {resemble_enhance.__file__}")

except Exception as e:
    print(f"Error: {e}")
