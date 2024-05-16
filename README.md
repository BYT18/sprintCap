# Markerless Sprinting Biomechanics Analysis

## Overview

This project aims to analyze sprinting biomechanics using markerless tracking techniques and produce a kinogram of key positions. Markerless tracking eliminates the need for physical markers on athletes, making it non-intrusive and suitable for real-world biomechanical analysis during sprinting.

Analysis is based on following papers ...


## Key Features

- **Markerless Tracking**: Utilizes computer vision techniques to track key points on the athlete's body during sprinting.
- **Biomechanical Analysis**: Analyzes the tracked data to extract key biomechanical metrics such as joint angles, velocity, and acceleration.
- **Kinogram Generation**: Produces a kinogram, a graphical representation of key positions and movements over time during the sprint.
- **Visualization**: Provides interactive visualizations to explore biomechanical data and kinograms.

![full_kinogram](https://github.com/BYT18/sprintCap/assets/68192622/9a37e814-f027-4cb2-838f-e5f180615a7d)

![animation](https://github.com/BYT18/sprintCap/assets/68192622/da933110-6d4a-4e0b-8b12-7ccba36c0fbb)

![initial_ground_contacts](https://github.com/BYT18/sprintCap/assets/68192622/156e3be2-41a9-4472-8d59-44d887af8156)

![time_on_ground](https://github.com/BYT18/sprintCap/assets/68192622/49ad8cab-589a-481a-8258-db985488b544)

![ground_vs_flight](https://github.com/BYT18/sprintCap/assets/68192622/4ccdd8ab-c00a-4868-b377-f46b524961a4)


## Dependencies

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Pandas

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/sprinting-biomechanics-analysis.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare video footage of sprinting sessions for analysis.
2. Run the main script to perform markerless tracking and biomechanical analysis:

```bash
python analyze_sprinting.py --video <path_to_video_file>
```

3. Explore the generated kinograms and biomechanical metrics to gain insights into sprinting performance.

## Project Structure

- `analyze_sprinting.py`: Main script for markerless tracking and biomechanical analysis.
- `utils.py`: Utility functions for video processing, tracking, and analysis.
- `visualizations.py`: Module for generating kinograms and visualizing biomechanical data.
- `README.md`: Project documentation.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

This project was inspired by [source/reference], and we thank [author/contributor] for their valuable insights and code contributions.
