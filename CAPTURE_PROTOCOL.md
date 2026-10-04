# Stock Capture Protocol

This is the current Route 2 protocol for the supported LiDAR tier. It is intentionally
literal so a non-engineer can reproduce the benchmark capture.

## Install

Install **Stray Scanner** from the App Store on an iPhone Pro or iPad Pro with LiDAR. Use the
latest version available on the capture date. Grant camera, motion, and location permissions;
location is not used by roomscan. In the app, enable depth, confidence, colour, intrinsics,
and pose export if those switches are present.

## Walk

1. Open every interior door that should count as an opening and turn on ordinary room lights.
2. Start one recording at the property entrance.
3. Walk at normal walking speed, holding the device at chest height and pointing forward.
4. Keep walls and the floor in view. Pause for two seconds in each room and rotate slowly once.
5. Pass through the connector and return to the entrance to create a loop closure.
6. Do not scan through mirrors or glass, move furniture, cover the camera, or walk faster than
   the app can track.
7. Stop only after the final room has a two-second pause and the return path is recorded.

## Hand-off

Export the raw capture, not a screenshot or rendered plan. Copy the exported folder without
renaming files into `data/<capture-id>/`. It must contain `camera_matrix.csv`, `imu.csv`,
`odometry.csv`, `depth/`, and `confidence/`. Run:

```powershell
.venv\Scripts\python.exe -m roomscan run data\<capture-id> --out out\<capture-id>
```

The output contract is `plan.json` plus `plan.png`. Preserve the raw folder and command output
for reproduction. For photo capture, place 2-8 JPG or HEIC images in one folder per room. For
video capture, place one `.mov` or `.mp4` in the capture folder. Both use the same command, but
metric RGB output remains provisional unless `scale.json` supplies room dimensions. An optional
`video_rooms` array in `scale.json` can identify frame ranges when automatic scene grouping is
ambiguous.