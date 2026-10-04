# Demo video script (~90 s)

**Before recording**

```bash
uv run lidar sim --seq 08 --checkpoint models/spunet-10cm-mix.pt   # wait for "warm-up … open http://127.0.0.1:8000"
```

Open `http://127.0.0.1:8000/?frame=1380&view=chase`, full screen (`F11`), close other GPU-heavy apps.

| # | Time | On screen | Say |
|---|---|---|---|
| 1 | 0:00–0:12 | Chase view, playing. | "This is FOVEA running live: a recorded LiDAR drive, segmented by our neural network and turned into a 2.5D map, every frame." |
| 2 | 0:12–0:25 | Point at the top-right chips, then the latency panel. | "It runs at about 30 frames per second, three times faster than the 10 Hz sensor." |
| 3 | 0:25–0:40 | Press `4` (Resolution), click **Top-down**. | "The map is foveated: 5 cm cells next to the vehicle, coarsening to 40 cm out to 100 m." |
| 4 | 0:40–0:50 | Point at **Memory vs uniform maps**. | "That makes it 8 MB: 22 times smaller than a uniform grid, 319 times smaller than 3D voxels." |
| 5 | 0:50–1:05 | Click **Chase**, press `1` (Semantic), then `2` (Drivability). | "Every cell knows what is there — road, terrain, obstacles, vehicles, pedestrians — and whether we can drive on it." |
| 6 | 1:05–1:15 | Hover a few road and curb cells. | "Each cell stores its 2.5D record: ground height, step to neighbours for curbs, clearance under overhangs." |
| 7 | 1:15–1:30 | Point at the boxes and the **Tracked objects** list. | "Vehicles and pedestrians are detected and tracked, and moving ones are told apart from parked ones. 67.7 % mIoU on the held-out sequence." |

**Cut list if time is short:** drop shot 6, then shot 4 (say the memory line over shot 3).

**Backup:** `docs/demo.mp4` is a pre-recorded 70 s walkthrough if the live run fails.
