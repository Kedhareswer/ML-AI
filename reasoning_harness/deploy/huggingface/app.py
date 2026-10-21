# Entry point for the free Gradio-SDK Space: Spaces run `python app.py` and
# route traffic to port 7860, so this starts the arena's own server there.
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

try:  # ZeroGPU (the only free Gradio hardware) refuses to start without one
    import spaces

    @spaces.GPU
    def _unused():
        pass

    # spaces normally reports this from gr.Blocks.launch, which this app never calls
    from spaces.zero import client
    client.startup_report()
except ImportError:
    pass

from arena.cli import main

main(["ui", "--host", "0.0.0.0", "--port", "7860",
      "--artifacts", "artifacts/arena", "--tests", "benchmarks/arena"])
