# LSF Recognition - Real-Time Sign Language Translation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project aims to develop a real-time recognition system for **French Sign Language (LSF)**. It detects signs from video input, converts them into raw text, reformulates the text into natural French using NLP, and generates audio output with text-to-speech (TTS). Subtitles are displayed for accessibility. The goal is to bridge communication gaps, with plans to support multiple sign languages for global impact.

### Key Features
- Sign detection using **MediaPipe Hands**.
- Sign classification with a **LSTM model** (TensorFlow/Keras).
- Text reformulation with **T5-small** (Hugging Face Transformers).
- Audio synthesis using **gTTS**.
- Interactive web interface with **Streamlit**.
- Compatible with the **LSF-Data** dataset ([parlr/lsf-data](https://github.com/parlr/lsf-data)) or custom datasets.

## Tech Stack
- **Python**: Core backend.
- **MediaPipe**: Real-time hand tracking.
- **TensorFlow/Keras**: LSTM for sign classification.
- **Hugging Face Transformers**: T5-small for text reformulation.
- **gTTS**: Text-to-speech.
- **Streamlit**: Web interface.
- **OpenCV**: Video processing.
- **Docker**: Deployment.

## Project Status
Under active development. Initial setup includes repository structure and documentation. Next steps: dataset integration and model training.

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/fless-lab/lsf-recognition.git
   ```
2. Further setup instructions coming soon.

## Repository Structure
```
lsf-recognition/
├── data/               # Raw and processed datasets
├── models/             # Trained and pretrained models
├── src/                # Source code (detection, classification, NLP, UI)
├── notebooks/          # Data exploration notebooks
├── tests/              # Unit tests
├── Dockerfile          # Containerization
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── LICENSE             # MIT License
```

## Contributing
Contributions are welcome! Check the [issues](https://github.com/<your-username>/lsf-recognition/issues) for open tasks or submit your ideas. Follow standard practices (tests, documentation, PEP8).

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For questions, open an issue or reach out via GitHub.

---
*Stay tuned for updates as the project grows!*