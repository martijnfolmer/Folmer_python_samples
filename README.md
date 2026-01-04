# Folmer_python_samples

A collection of practical python sample scripts. It is meant to be reusable, copy+pastable functions that I often use
in projects I'm working on. Feel free to use anything you find here in any way you like.

This is an ongoing process and I will be regularly adding to this repository

## Audio
### TrimWav.py
Shorten a .wav file by trimming the start and end of the audio file

## File Handling
### DeleteEverything.py
Delete a directory, including all files, directories, subdirectories, sub-subdirectories etc.

## Images

### ConvertImagesTypeDir.py
Batch converts images from one format to another.

### ORBComparison.py
Compare 2 images using ORB to check for feature matching and alignment issues
![2 images with matches between where we see similar features](readme_img/ORBComparison.png)

### ResizeImagesDir.py
Batch converts images from one size to another.



## Video

### ConcatenateVideos.py
Concatenates multiple videos into a single output video.
### StackVideo.py
Stacks multiple videos into one frame, either horizontally or vertically.

---

## Requirements

- Python 3.10+
- OpenCV (`pip install opencv-python`)
- NumPy 
- pydub (for audio processing)
- tqdm (for progress bars)
---

## Notes

- All scripts use hardcoded paths and settings inside `__main__` for clarity
- Modify paths and parameters directly in the script before running
- Designed for clarity over maximum performance. 
- Not all quality of life functionality exists over every script, so do not be alarmed if they for example have different verbose printing

Feel free to adapt or extend these samples for your own workflows.