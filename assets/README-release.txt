IntelliProctor 1.0.0 - Windows x64
===================================

IntelliProctor is an offline desktop application for exam-monitoring research.
It uses a webcam or video source and runs the detection models locally.

Requirements
------------
- Windows 10 or Windows 11, 64-bit
- No Python installation is required
- NVIDIA GPU (optional): CUDA inference is faster; CPU inference also works
- Keep IntelliProctor.exe and the _internal folder together

First run
---------
1. Extract the whole archive to a normal folder. Do not run the app from
   inside the ZIP file.
2. Double-click "Install IntelliProctor.cmd" to create a Desktop shortcut.
   Alternatively, double-click IntelliProctor.exe directly.
3. In the application, choose a webcam, video file, or CCTV stream and start
   a session.

The first launch can take longer while Windows loads the bundled libraries.
The executable is not a standalone file: _internal contains the models,
Python runtime, Qt libraries, and CUDA libraries it needs.

Data and privacy
----------------
- Processing is local and offline.
- Recorded feature CSV:
  %LOCALAPPDATA%\IntelliProctor\data\proctoring_features.csv
- Nothing is uploaded by the application.

Troubleshooting
---------------
- Windows SmartScreen may warn because this is an unsigned academic build.
  Choose "More info" then "Run anyway" only if you obtained the archive from
  a trusted source.
- If the app does not start, verify that the complete folder was extracted
  and that IntelliProctor.exe remains next to _internal.
- If a session cannot open the camera, close other applications that may be
  using it and check Windows camera permissions.
- The bundled COCO fallback detector covers phone/laptop/book only. The
  optional custom desk model is used automatically when present.

Uninstall
---------
Delete the Desktop shortcut and the IntelliProctor folder. No system-wide
files or registry entries are installed.
