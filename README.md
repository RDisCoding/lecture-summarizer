# 🎧 Lecture Transcript Analyzer

Transform your recorded lectures into detailed study summaries and curated learning resources with the power of AI. Perfect for college students who want to maximize their learning from audio and video lectures.

[![Live Demo](https://img.shields.io/badge/🚀%20Try%20It%20Live-thelazyone.streamlit.app-FF4B4B?style=for-the-badge)](https://thelazyone.streamlit.app/)

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## ✨ Features

### 🎯 Perfect for Students
- **Smart Transcription**: Convert any lecture audio/video to accurate text using OpenAI's Whisper
- **Intelligent Summarization**: Get comprehensive study-ready summaries that capture all key concepts
- **Curated Learning Resources**: Automatically find relevant books, courses, tutorials, and documentation
- **Progress Tracking**: Visual progress bar with ETA estimation for long lectures
- **Analysis History**: Save and revisit past lecture analyses
- **Export Options**: Download summaries as text files for offline study

### 🚀 Technical Highlights
- **CPU Optimized**: Works efficiently on laptop hardware (no GPU required)
- **Smart Audio Processing**: Intelligent chunking based on silence detection
- **Multiple Formats**: Supports MP3, WAV, MP4, AVI, MOV, MKV, WebM, and more
- **Cost Effective**: Uses Perplexity's cheapest "sonar" model for analysis
- **Robust Error Handling**: Graceful fallbacks and retry mechanisms

## 🎓 Why Students Love This Tool

**Before**: Struggling to review hours of lecture recordings, missing key points, spending time searching for additional resources.

**After**: Get a comprehensive summary in minutes, plus a curated list of resources to deepen your understanding.

### Real Student Use Cases
- **Missed Lectures**: Quickly catch up on classes you couldn't attend
- **Exam Preparation**: Create study guides from recorded review sessions
- **Research**: Summarize academic talks, conferences, and webinars
- **Language Learning**: Get transcripts of foreign language content
- **Accessibility**: Convert audio content to text for better accessibility

## 🚀 Quick Start

### 🌐 Try It Online (Recommended)
**No installation required!** Just visit: **[thelazyone.streamlit.app](https://thelazyone.streamlit.app/)**

Simply upload your lecture file and enter your Perplexity API key to get started instantly.

### 💻 Local Installation

#### Prerequisites
- Python 3.8 or higher
- Perplexity API key ([Get one here](https://www.perplexity.ai/settings/api))

#### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/lecture-transcript-analyzer.git
   cd lecture-transcript-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your API key** (Optional - can also enter in the app)
   ```bash
   export PERPLEXITY_API_KEY="your_api_key_here"
   ```

4. **Launch the app**
   ```bash
   python launch.py
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run lecture_analyzer_app.py
   ```

## 📋 Requirements

```
streamlit>=1.28.0
openai-whisper>=20231117
requests>=2.31.0
moviepy>=1.0.3
pydub>=0.25.1
torch>=2.0.0
```

## 🎮 How to Use

### 🌐 Using the Live App (Easiest)
1. **Visit** [thelazyone.streamlit.app](https://thelazyone.streamlit.app/)
2. **Enter your Perplexity API key** in the sidebar
3. **Upload your lecture file** (audio or video)
4. **Adjust settings** if needed:
   - Whisper model size (tiny/base/small/medium)
   - Audio chunk duration for processing
5. **Click "Analyze"** and watch the progress tracker
6. **Review results** in two tabs:
   - 📝 **Summary**: Comprehensive lecture summary
   - 🔗 **Resources**: Curated learning materials
7. **Download or save** your results for later study

### 💻 Using Local Installation
1. **Start the app** using `python launch.py`
2. **Follow steps 2-7** from above

## ⚙️ Configuration Options

### Whisper Model Selection
- **Tiny** (39MB): Fastest, basic accuracy - great for quick previews
- **Base** (74MB): Good balance of speed and accuracy - recommended default
- **Small** (244MB): Better accuracy, slower processing
- **Medium** (769MB): High accuracy for important lectures

### Audio Processing
- **Chunk Duration**: Adjust based on your system's memory (60-600 seconds)
- **Intelligent Chunking**: Automatically splits on silence for natural breaks
- **Audio Preprocessing**: Normalizes volume and converts to optimal format

## 📊 Performance Tips

### For Best Results
- **Clear Audio**: Use recordings with minimal background noise
- **Steady Internet**: Required for Perplexity API calls
- **Sufficient RAM**: 4GB+ recommended for medium Whisper models
- **Audio Quality**: 16kHz mono WAV files process fastest

### Processing Times (Approximate)
- 1-hour lecture with base model: ~10-15 minutes on typical laptop
- Processing speed: 4-6x real-time with base model
- Summary generation: ~30-60 seconds
- Resource finding: ~30-45 seconds

## 🔧 Troubleshooting

### Common Issues

**"No speech detected"**
- Check audio volume levels
- Ensure the file isn't corrupted
- Try preprocessing with audio editing software

**API Rate Limits**
- The app automatically handles rate limiting with exponential backoff
- Consider upgrading your Perplexity plan for higher limits

**Memory Issues**
- Use smaller Whisper models (tiny/base)
- Reduce chunk duration
- Close other applications while processing

**Slow Processing**
- Use faster Whisper models for quick previews
- Process shorter segments first
- Ensure sufficient free disk space for temporary files

## 🛠️ Architecture

```
lecture_analyzer_app.py     # Main Streamlit application
├── transcript_generator.py # Whisper transcription engine
├── launch.py              # Dependency checker and launcher
└── requirements.txt       # Python dependencies
```

### Key Components
- **WhisperTranscriber**: Handles audio processing and transcription
- **PerplexityAPI**: Manages AI-powered summarization and resource discovery
- **ProgressTracker**: Provides real-time progress updates with ETA
- **AudioProcessor**: Optimizes audio for transcription

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and test thoroughly
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Areas for Contribution
- Support for additional languages
- Integration with more AI providers
- Improved audio preprocessing
- Better error handling
- UI/UX improvements
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper** for state-of-the-art speech recognition
- **Perplexity AI** for intelligent summarization and resource discovery
- **Streamlit** for the beautiful and intuitive web interface
- **The Open Source Community** for the amazing libraries that make this possible

## 📞 Support

Having issues? We're here to help!

- 🐛 **Bug Reports**: [Open an issue](https://github.com/yourusername/lecture-transcript-analyzer/issues)
- 💡 **Feature Requests**: [Start a discussion](https://github.com/yourusername/lecture-transcript-analyzer/discussions)
- 📧 **Email**: rdiscoding@gmail.com

## 🎯 Roadmap

### Coming Soon
- [ ] Batch processing for multiple files
- [ ] Integration with popular learning management systems
- [ ] Mobile app version
- [ ] Support for live lecture transcription
- [ ] Advanced analytics and insights
- [ ] Collaborative note-taking features

### Future Ideas
- [ ] Integration with note-taking apps (Notion, Obsidian)
- [ ] Automatic flashcard generation
- [ ] Quiz creation from lecture content
- [ ] Multi-language support
- [ ] Advanced search across all transcripts

---

<div align="center">

**⭐ If this tool helped you ace your studies, please star this repo! ⭐**

**🚀 [Try it now at thelazyone.streamlit.app](https://thelazyone.streamlit.app/)**

Made with ❤️ for students who want to learn smarter, not harder.

[⬆ Back to Top](#-lecture-transcript-analyzer)

</div>
