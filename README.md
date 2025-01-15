# Image to Video Using Prompt

This project transforms a static image into a video based on a text prompt using Stable Diffusion and frame interpolation techniques. For example, you can turn an image of a sitting cat into a video of the cat starting to run.

## Features

- Image-to-image transformation using Stable Diffusion
- Smooth frame interpolation for fluid motion
- Customizable video duration and FPS
- Support for various image formats
- GPU acceleration (if available)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SanjeevJMT/image_to_video.git
cd Image_To_Video
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your input image in the `input` directory
2. Run the script:

```python
from main import ImageToVideo

converter = ImageToVideo()
converter.create_video(
    image_path="input/image.jpg",
    prompt="input/prompt.txt",
    output_path="output/video_out.mp4"
)
```

### Parameters

- `image_path`: Path to the input image
- `prompt`: Text description of the desired motion/transformation
- `output_path`: Path where the generated video will be saved
- `duration`: Video duration in seconds (default: 4)

## Project Structure

```
Image_To_Video_Using_Prompt/
├── input/              # Input images directory
├── output/            # Generated videos directory
├── app.py           # Main implementation
├── requirements.txt  # Project dependencies
└── README.md        # Project documentation
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)
- See requirements.txt for full list of dependencies

## Limitations

- The quality of the output depends on the input image quality and prompt clarity
- Processing time varies based on hardware capabilities
- Video transitions work best with natural, gradual movements
- May require significant GPU memory for processing

## Contributing

Feel free to open issues or submit pull requests with improvements.

## License

No License yet.