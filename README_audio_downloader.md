# Audio Downloader and Converter

This script downloads MP3 files from URLs, converts them to WAV format, and saves them to the `Audios` folder.

## Features

- Downloads MP3 files from a list of URLs
- Converts MP3 to WAV format
- Saves files to the `Audios` folder
- Includes retry logic for failed downloads
- Skips files that already exist
- Provides detailed progress information
- Shows audio file information (duration, sample rate, file size)

## Setup

### 1. Install Dependencies

Make sure you have the required Python packages:

```bash
# Activate your virtual environment
source venv/bin/activate

# Install required packages
pip install pydub requests
```

### 2. Install FFmpeg (Required for audio conversion)

The `pydub` library requires FFmpeg for audio conversion. You can install it using one of these methods:

**Option A: System-wide installation (requires sudo)**
```bash
sudo apt update
sudo apt install -y ffmpeg
```

**Option B: Using conda (if you have conda installed)**
```bash
conda install ffmpeg
```

**Option C: Download and install manually**
- Visit https://ffmpeg.org/download.html
- Download the appropriate version for your system
- Follow the installation instructions

## Usage

### Method 1: Using the main function directly

```python
from audio_downloader import download_and_convert_audio

# Your list of 20 URLs
urls = [
    "https://erpnoveloffice.in/files/ATSID00897933_introduction.mp3",
    "https://example.com/audio2.mp3",
    "https://example.com/audio3.mp3",
    # ... add all your URLs here
]

# Process the audio files
processed_files = download_and_convert_audio(urls)
```

### Method 2: Using the example script

1. Edit `example_usage.py` and replace the URLs with your actual 20 URLs
2. Run the script:
```bash
python example_usage.py
```

### Method 3: Running the main script directly

1. Edit `audio_downloader.py` and replace the example URLs with your actual URLs
2. Run the script:
```bash
python audio_downloader.py
```

## Function Parameters

The `download_and_convert_audio` function accepts these parameters:

- `urls` (List[str]): List of URLs pointing to MP3 files
- `output_folder` (str): Folder to save the converted WAV files (default: "Audios")
- `download_timeout` (int): Timeout for download requests in seconds (default: 30)
- `retry_attempts` (int): Number of retry attempts for failed downloads (default: 3)

## Output

- WAV files are saved to the `Audios` folder
- Original filenames are preserved (with .wav extension)
- Temporary MP3 files are automatically cleaned up
- Progress information is displayed during processing

## Example Output

```
Starting download and conversion of 20 audio files...

Processing file 1/20: https://erpnoveloffice.in/files/ATSID00897933_introduction.mp3
  Downloading MP3 (attempt 1/3)...
  MP3 downloaded successfully: ATSID00897933_introduction.mp3
  Converting MP3 to WAV...
  Conversion successful: ATSID00897933_introduction.wav

Processing file 2/20: https://example.com/audio2.mp3
  Downloading MP3 (attempt 1/3)...
  MP3 downloaded successfully: audio2.mp3
  Converting MP3 to WAV...
  Conversion successful: audio2.wav

...

Processing complete!
Successfully processed: 18/20 files
Files saved to: /home/novel/Audios
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Make sure FFmpeg is installed and accessible in your PATH
2. **Download timeouts**: Increase the `download_timeout` parameter for slow connections
3. **Permission errors**: Make sure you have write permissions to the output folder
4. **Network errors**: The script includes retry logic, but check your internet connection

### Error Messages

- `"Download attempt X failed"`: Network or server issues, will retry automatically
- `"Conversion failed"`: Usually indicates FFmpeg is not installed or the file is corrupted
- `"WAV file already exists"`: File was already processed, skipping to save time

## File Structure

```
your_project/
├── audio_downloader.py      # Main script with functions
├── example_usage.py         # Example usage script
├── Audios/                  # Output folder for WAV files
├── temp_mp3/               # Temporary folder (auto-created and cleaned)
└── README_audio_downloader.md  # This file
```

## Notes

- The script automatically creates the `Audios` folder if it doesn't exist
- Files that already exist as WAV files are skipped to avoid re-processing
- Temporary MP3 files are automatically deleted after conversion
- The script provides detailed progress information and error handling 