"""
marker_test.py

Standalone check: can Python launch a custom-named no-op GPU kernel that
ncu captures? This is the prerequisite for the marker-kernel scheme in
the roofline measurement pipeline.

Uses numba.cuda -- compiles a Python function to a real GPU kernel. The
kernel does nothing, but it IS a genuine kernel launch, so ncu sees it.

Run directly to confirm it launches:
    ../myenv/bin/python marker_test.py

Then profile it with ncu (see the command printed at the end) and check
that a kernel named 'roofline_marker' appears in the output.
"""

import sys

try:
    from numba import cuda
except ImportError:
    print("numba is NOT installed in this environment.")
    print("Try:  pip install numba   (into ../myenv)")
    print("Or tell me and we'll use cupy / fall back to the 'last K' scheme.")
    sys.exit(1)


# A no-op kernel. The Python function name becomes the kernel name that
# ncu reports, so this will show up as 'roofline_marker'.
@cuda.jit
def roofline_marker():
    # genuinely nothing -- but this is still a real kernel launch
    return


def main():
    print("numba.cuda imported OK")
    try:
        dev = cuda.get_current_device()
        print(f"GPU detected: {dev.name.decode() if isinstance(dev.name, bytes) else dev.name}")
    except Exception as e:
        print(f"could not query device: {e}")
        sys.exit(1)

    # launch the marker a few times so it is unmistakable in the profile
    n_launches = 5
    print(f"launching roofline_marker kernel {n_launches} times...")
    for _ in range(n_launches):
        roofline_marker[1, 1]()      # <<<1 block, 1 thread>>>
    cuda.synchronize()
    print("done -- kernel launches completed")
    print()
    print("Now profile with ncu and look for 'roofline_marker' in the output:")
    print()
    print("  ncu --target-processes all \\")
    print("      --metrics dram__bytes_read.sum,dram__bytes_write.sum \\")
    print("      --csv \\")
    print(f"      {sys.executable} {__file__}")


if __name__ == "__main__":
    main()