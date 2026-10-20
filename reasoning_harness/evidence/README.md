# Actual running application

These are real Playwright captures of Chromium interacting with the locally running model server. Browser assertions checked that every displayed solve returned `solved: true`. The model was the packaged 8,024,004-parameter policy trained for 1,500 steps. No generated imagery, substituted model responses or mock API was used.

- [Screen recording, MP4](solving-demo.mp4) / [original WebM](solving-demo.webm): neural-only 4x4, neural-only forced detour, then guided 8x8.
- [Neural-only 4x4 screenshot](neural-4x4.png).
- [Neural-only forced-detour screenshot](neural-detour.png): route `RRRDDLLLD`, nine moves, independently verified and optimal.
- [Guided 8x8 screenshot](guided-8x8.png): explicit backtracking mode, displayed separately from the neural-only examples.
- [Actual API responses, grids and modes](browser-evidence.json).

The recordings show route playback **after** the model responds. Playback is visual evidence of the verified route, not a claim of streamed model reasoning. These are selected demonstrations; use the held-out benchmark for aggregate accuracy. The UI measured request time while other benchmark/training work shared this machine, so the displayed latency is not an isolated speed guarantee.

To repeat: start `sys2 ui --run artifacts/policy-v1 --temperature 0 --port 7860`, then run `scripts/capture_demo.py` with Playwright, Chromium and video encoding installed. This environment used the existing trusted system Chromium/ffmpeg binaries; downloading Playwright's ffmpeg was blocked by the network policy. The provided MP4 is a format conversion of the actual WebM capture.
