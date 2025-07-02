
"""
Lecture Summarizer - Transcript Generator Module
Optimized for CPU-only systems with intelligent chunking and error handling
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

import whisper
import torch
from moviepy import VideoFileClip
import subprocess
from pydub import AudioSegment
from pydub.silence import split_on_silence

@dataclass
class TranscriptionResult:
    """Data class to store transcription results"""
    text: str
    segments: List[Dict]
    language: str
    duration: float
    model_used: str
    processing_time: float
    word_count: int
    confidence_score: Optional[float] = None

class AudioProcessor:
    """Handles audio extraction and preprocessing"""
    
    def __init__(self, temp_dir: str = "temp_audio"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def extract_audio_from_video(self, video_path: str, output_path: str = None) -> str:
        """Extract audio from video file using MoviePy"""
        try:
            if output_path is None:
                video_name = Path(video_path).stem
                output_path = self.temp_dir / f"{video_name}_audio.wav"
            
            self.logger.info(f"Extracting audio from {video_path}")
            
            # Use MoviePy for reliable extraction
            video = VideoFileClip(video_path)
            audio = video.audio
            
            # Extract as WAV for best compatibility with Whisper
            audio.write_audiofile(
                str(output_path),
                codec='pcm_s16le',  # 16-bit PCM
                ffmpeg_params=['-ac', '1'],  # Convert to mono
                logger=None
            )
            
            # Clean up
            audio.close()
            video.close()
            
            self.logger.info(f"Audio extracted to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error extracting audio: {str(e)}")
            raise
    
    def preprocess_audio(self, audio_path: str) -> str:
        """Preprocess audio for optimal Whisper performance"""
        try:
            self.logger.info(f"Preprocessing audio: {audio_path}")
            
            # Load audio with pydub
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
                self.logger.info("Converted audio to mono")
            
            # Normalize sample rate to 16kHz (Whisper's native rate)
            if audio.frame_rate != 16000:
                audio = audio.set_frame_rate(16000)
                self.logger.info(f"Converted sample rate to 16kHz")
            
            # Normalize volume
            audio = audio.normalize()
            
            # Save preprocessed audio
            processed_path = self.temp_dir / f"processed_{Path(audio_path).name}"
            audio.export(str(processed_path), format="wav")
            
            self.logger.info(f"Preprocessed audio saved to {processed_path}")
            return str(processed_path)
            
        except Exception as e:
            self.logger.error(f"Error preprocessing audio: {str(e)}")
            return audio_path  # Return original if preprocessing fails

class WhisperTranscriber:
    """Main transcription class optimized for CPU systems"""
    
    def __init__(self, model_size: str = "base"):
        self.logger = logging.getLogger(__name__)
        self.model_size = model_size
        self.model = None
        self.audio_processor = AudioProcessor()
        
        # Model selection based on system capabilities
        self.model_recommendations = {
            "tiny": {"size_mb": 39, "speed": "fastest", "accuracy": "lowest"},
            "base": {"size_mb": 74, "speed": "fast", "accuracy": "good"},
            "small": {"size_mb": 244, "speed": "medium", "accuracy": "better"},
            "medium": {"size_mb": 769, "speed": "slow", "accuracy": "very good"},
            "large": {"size_mb": 1550, "speed": "very slow", "accuracy": "best"}
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model with error handling"""
        try:
            self.logger.info(f"Loading Whisper model: {self.model_size}")
            start_time = time.time()
            
            # Load model with CPU optimization
            self.model = whisper.load_model(
                self.model_size,
                device="cpu"
            )
            
            load_time = time.time() - start_time
            model_info = self.model_recommendations.get(self.model_size, {})
            
            self.logger.info(f"Model loaded in {load_time:.2f}s")
            self.logger.info(f"Model size: ~{model_info.get('size_mb', 'unknown')}MB")
            self.logger.info(f"Expected speed: {model_info.get('speed', 'unknown')}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            # Fallback to tiny model
            if self.model_size != "tiny":
                self.logger.warning("Falling back to tiny model")
                self.model_size = "tiny"
                self._load_model()
            else:
                raise
    
    def chunk_audio_intelligent(self, audio_path: str, chunk_duration: int = 300) -> List[str]:
        """Intelligently chunk audio based on silence detection"""
        try:
            self.logger.info(f"Chunking audio: {audio_path}")
            
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            total_duration = len(audio) / 1000  # Convert to seconds
            
            # If audio is short enough, don't chunk
            if total_duration <= chunk_duration:
                self.logger.info(f"Audio duration {total_duration:.1f}s is within chunk limit")
                return [audio_path]
            
            # Split on silence for natural breakpoints
            chunks = split_on_silence(
                audio,
                min_silence_len=1000,  # 1 second of silence
                silence_thresh=-40,    # dB threshold
                keep_silence=500       # Keep 0.5s of silence
            )
            
            # If no good silence breaks found, use time-based chunking
            if len(chunks) <= 1:
                self.logger.info("No silence breaks found, using time-based chunking")
                return self._chunk_by_time(audio_path, chunk_duration)
            
            # Combine small chunks and split large ones
            chunk_paths = []
            current_chunk = AudioSegment.empty()
            chunk_count = 0
            
            for i, chunk in enumerate(chunks):
                current_chunk += chunk
                
                # If current chunk is long enough or we're at the end
                if len(current_chunk) >= chunk_duration * 1000 or i == len(chunks) - 1:
                    chunk_path = self.audio_processor.temp_dir / f"chunk_{chunk_count:03d}.wav"
                    current_chunk.export(str(chunk_path), format="wav")
                    chunk_paths.append(str(chunk_path))
                    
                    chunk_count += 1
                    current_chunk = AudioSegment.empty()
            
            self.logger.info(f"Audio split into {len(chunk_paths)} chunks")
            return chunk_paths
            
        except Exception as e:
            self.logger.error(f"Error chunking audio: {str(e)}")
            # Fallback to simple time-based chunking
            return self._chunk_by_time(audio_path, chunk_duration)
    
    def _chunk_by_time(self, audio_path: str, chunk_duration: int) -> List[str]:
        """Simple time-based audio chunking"""
        try:
            audio = AudioSegment.from_file(audio_path)
            total_duration = len(audio)
            chunk_duration_ms = chunk_duration * 1000
            
            chunk_paths = []
            for i, start in enumerate(range(0, total_duration, chunk_duration_ms)):
                end = min(start + chunk_duration_ms, total_duration)
                chunk = audio[start:end]
                
                chunk_path = self.audio_processor.temp_dir / f"time_chunk_{i:03d}.wav"
                chunk.export(str(chunk_path), format="wav")
                chunk_paths.append(str(chunk_path))
            
            return chunk_paths
            
        except Exception as e:
            self.logger.error(f"Error in time-based chunking: {str(e)}")
            return [audio_path]  # Return original if chunking fails
    
    def transcribe_chunk(self, audio_path: str, previous_text: str = "") -> Dict:
        """Transcribe a single audio chunk"""
        try:
            # Use previous text as context (first 224 tokens)
            prompt = previous_text[-224:] if previous_text else ""
            
            # Transcribe with word-level timestamps
            result = self.model.transcribe(
                audio_path,
                language="en",  # Can be auto-detected by removing this
                prompt=prompt,
                word_timestamps=True,
                verbose=False
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error transcribing chunk {audio_path}: {str(e)}")
            return {"text": "", "segments": [], "language": "en"}
    
    def transcribe_file(self, input_path: str, chunk_duration: int = 300) -> TranscriptionResult:
        """Main transcription method"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting transcription of: {input_path}")
            
            # Determine if input is video or audio
            file_ext = Path(input_path).suffix.lower()
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
            
            # Extract audio if video file
            if file_ext in video_extensions:
                audio_path = self.audio_processor.extract_audio_from_video(input_path)
            else:
                audio_path = input_path
            
            # Preprocess audio
            processed_audio = self.audio_processor.preprocess_audio(audio_path)
            
            # Get audio duration
            audio = AudioSegment.from_file(processed_audio)
            total_duration = len(audio) / 1000
            
            # Chunk audio if necessary
            chunk_paths = self.chunk_audio_intelligent(processed_audio, chunk_duration)
            
            # Transcribe each chunk
            all_text = ""
            all_segments = []
            total_offset = 0
            
            for i, chunk_path in enumerate(chunk_paths):
                self.logger.info(f"Transcribing chunk {i+1}/{len(chunk_paths)}")
                
                chunk_result = self.transcribe_chunk(chunk_path, all_text)
                
                # Adjust timestamps based on chunk position
                for segment in chunk_result.get("segments", []):
                    segment["start"] += total_offset
                    segment["end"] += total_offset
                    
                    # Adjust word timestamps if available
                    if "words" in segment:
                        for word in segment["words"]:
                            word["start"] += total_offset
                            word["end"] += total_offset
                
                all_text += " " + chunk_result.get("text", "")
                all_segments.extend(chunk_result.get("segments", []))
                
                # Update offset for next chunk
                chunk_audio = AudioSegment.from_file(chunk_path)
                total_offset += len(chunk_audio) / 1000
            
            # Clean up text
            all_text = all_text.strip()
            
            # Calculate metrics
            processing_time = time.time() - start_time
            word_count = len(all_text.split())
            
            # Create result object
            result = TranscriptionResult(
                text=all_text,
                segments=all_segments,
                language=chunk_result.get("language", "en"),
                duration=total_duration,
                model_used=self.model_size,
                processing_time=processing_time,
                word_count=word_count
            )
            
            self.logger.info(f"Transcription completed in {processing_time:.1f}s")
            self.logger.info(f"Total duration: {total_duration:.1f}s")
            self.logger.info(f"Word count: {word_count}")
            self.logger.info(f"Processing speed: {total_duration/processing_time:.1f}x real-time")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during transcription: {str(e)}")
            raise
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            temp_files = list(self.audio_processor.temp_dir.glob("*"))
            for file_path in temp_files:
                file_path.unlink()
            self.logger.info(f"Cleaned up {len(temp_files)} temporary files")
        except Exception as e:
            self.logger.warning(f"Error cleaning up temp files: {str(e)}")

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "transcript_generator.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logger = setup_logging()
    
    # Initialize transcriber
    transcriber = WhisperTranscriber(model_size="base")
    
    # Example file path (you would replace this with your actual file)
    # test_file = "path/to/your/lecture.mp4"
    
    print("Transcript Generator Module Loaded Successfully!")
    print(f"Available Whisper models: {list(transcriber.model_recommendations.keys())}")
    print(f"Current model: {transcriber.model_size}")
    print("\nReady to transcribe audio/video files.")
    print("\nUsage example:")
    print("transcriber = WhisperTranscriber('base')")
    print("result = transcriber.transcribe_file('your_lecture.mp4')")
    print("print(result.text)")
