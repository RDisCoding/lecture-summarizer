
"""
Configuration file for the Lecture Summarizer Transcript Generator
Modify these settings according to your needs and system capabilities
"""

import os
from pathlib import Path

# Model Configuration
MODEL_CONFIG = {
    # Default model size - adjust based on your system's capabilities
    "default_model": "base",  # Options: tiny, base, small, medium, large
    
    # Model recommendations based on system specs
    "recommendations": {
        "low_memory": "tiny",      # <8GB RAM
        "medium_memory": "base",   # 8-16GB RAM  
        "high_memory": "small",    # 16-32GB RAM
        "very_high_memory": "medium"  # >32GB RAM
    },
    
    # Automatic model selection based on available memory
    "auto_select": True
}

# Audio Processing Configuration
AUDIO_CONFIG = {
    # Chunk duration for long audio files (seconds)
    "default_chunk_duration": 300,  # 5 minutes
    
    # Audio preprocessing settings
    "sample_rate": 16000,  # Whisper's native sample rate
    "channels": 1,         # Mono audio
    "normalize_audio": True,
    
    # Silence detection for intelligent chunking
    "silence_threshold": -40,  # dB
    "min_silence_duration": 1000,  # milliseconds
    "keep_silence": 500,    # milliseconds
}

# File Handling Configuration
FILE_CONFIG = {
    # Supported file extensions
    "video_extensions": ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'],
    "audio_extensions": ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'],
    
    # Output directories
    "temp_dir": "temp_audio",
    "transcripts_dir": "transcripts",
    "logs_dir": "logs",
    
    # Default output formats
    "default_formats": ["txt", "json", "srt"],
    
    # File naming
    "include_timestamp": True,
    "timestamp_format": "%Y%m%d_%H%M%S"
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    # CPU optimization
    "use_cpu": True,
    "cpu_threads": os.cpu_count(),
    
    # Memory management
    "max_memory_usage": "80%",  # Percentage of available RAM
    "garbage_collect": True,
    
    # Processing limits
    "max_file_size_mb": 1000,  # Maximum input file size
    "max_duration_minutes": 180,  # 3 hours
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_logging": True,
    "console_logging": True,
    "max_log_size_mb": 10,
    "backup_count": 5
}

# Feature Flags
FEATURES = {
    "word_timestamps": True,
    "language_detection": True,
    "automatic_chunking": True,
    "silence_removal": True,
    "audio_normalization": True,
    "progress_tracking": True,
    "batch_processing": True
}

# System-specific optimizations
def get_optimal_settings():
    """Get optimal settings based on current system"""
    import psutil
    
    # Get system memory
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Get optimal model based on memory
    if memory_gb < 8:
        model = "tiny"
        chunk_duration = 180  # Smaller chunks for low memory
    elif memory_gb < 16:
        model = "base"
        chunk_duration = 300
    elif memory_gb < 32:
        model = "small"
        chunk_duration = 600
    else:
        model = "medium"
        chunk_duration = 900
    
    return {
        "recommended_model": model,
        "optimal_chunk_duration": chunk_duration,
        "memory_available": f"{memory_gb:.1f}GB"
    }

# Validation functions
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check if temp directory is writable
    temp_path = Path(FILE_CONFIG["temp_dir"])
    try:
        temp_path.mkdir(exist_ok=True)
        test_file = temp_path / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        errors.append(f"Cannot write to temp directory: {e}")
    
    # Check model validity
    valid_models = ["tiny", "base", "small", "medium", "large"]
    if MODEL_CONFIG["default_model"] not in valid_models:
        errors.append(f"Invalid model: {MODEL_CONFIG['default_model']}")
    
    # Check chunk duration
    if AUDIO_CONFIG["default_chunk_duration"] < 30:
        errors.append("Chunk duration too small (minimum 30 seconds)")
    
    return errors

# Export commonly used settings
DEFAULT_MODEL = MODEL_CONFIG["default_model"]
DEFAULT_CHUNK_DURATION = AUDIO_CONFIG["default_chunk_duration"]
TEMP_DIR = FILE_CONFIG["temp_dir"]
TRANSCRIPTS_DIR = FILE_CONFIG["transcripts_dir"]

if __name__ == "__main__":
    print("🔧 Lecture Summarizer Configuration")
    print("=" * 40)
    
    # Show current settings
    print(f"Default model: {DEFAULT_MODEL}")
    print(f"Chunk duration: {DEFAULT_CHUNK_DURATION}s")
    print(f"Temp directory: {TEMP_DIR}")
    print(f"Output directory: {TRANSCRIPTS_DIR}")
    
    # Show system-specific recommendations
    print("\n📊 System Analysis:")
    optimal = get_optimal_settings()
    for key, value in optimal.items():
        print(f"   {key}: {value}")
    
    # Validate configuration
    print("\n✅ Configuration Validation:")
    errors = validate_config()
    if errors:
        print("❌ Issues found:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ All settings valid!")
