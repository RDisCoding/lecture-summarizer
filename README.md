# Lecture Summarizer Project

An AI-powered system that automatically summarizes video lectures and discovers relevant learning resources.

## Features
- Speech-to-text transcription using OpenAI Whisper
- Intelligent lecture summarization with fine-tuned LLMs
- Automatic discovery of relevant educational resources
- Topic-focused content extraction

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run setup test: `python test_setup.py`
3. Test Whisper: `python whisper_example.py`

## Project Structure
- `data/`: All data files (raw audio, transcripts, summaries)
- `src/`: Source code modules
- `notebooks/`: Jupyter notebooks for experimentation
- `models/`: Trained model files
- `config/`: Configuration files
- `tests/`: Unit tests

## Next Steps
1. Start with audio transcription (Whisper)
2. Build summarization pipeline
3. Implement resource discovery
4. Fine-tune models for better performance
