import os 
import time
import tempfile
import multiprocessing as mp
from openai import OpenAI
from tqdm import tqdm
from pydub import AudioSegment 
from pathlib import Path


class AudioHandler: 
    def __init__(self, filepath: str | Path): 
        if isinstance(filepath, Path): 
            filepath = str(filepath)
        self.fp = str(filepath) if isinstance(filepath, Path) else filepath
    
    def _split_audio(self, chunk_length_ms=60_000) -> list:
        print(f"Loading audio file: {self.fp}")
        audio = AudioSegment.from_file(self.fp)

        chunks = []
        temp_dir = tempfile.mkdtemp()

        for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
            end = start + chunk_length_ms
            chunk = audio[start:end]
            chunk_path = os.path.join(temp_dir, f"chunk_{i}.mp3")
            chunk.export(chunk_path, format="mp3")
            chunks.append((i, chunk_path))
    
        print(f"Split audio into {len(chunks)} chunks")
        return chunks
    
    def _transcribe_chunk(self, args: tuple) -> tuple: 
        chunk_index, chunk_path = args
        try:
            client = OpenAI()
            audio_file = open(chunk_path, "rb")
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
            return chunk_index, transcription.text
        except Exception as e:
            print(f"Error processing chunk {chunk_index}: {str(e)}")
            return chunk_index, ""
        finally:
            try:
                os.remove(chunk_path)
            except:
                pass

    def process_audio(self): 
        try:
            start_time = time.time()
            
            chunks = self._split_audio()
            process_args = [(i, path) for i, path in chunks]
            num_processes = max(1, mp.cpu_count() - 1)

            print(f"Starting parallel transcription with {num_processes} processes...")
            with mp.Pool(num_processes) as pool:
                results = list(tqdm(
                    pool.imap(self._transcribe_chunk, process_args),
                    total=len(process_args),
                    desc="Transcribing chunks"
                ))
            
            results.sort(key=lambda x: x[0])
            full_transcript = "\n".join(text for _, text in results if text)
            
            output_path = Path(self.fp).with_suffix('.txt')
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_transcript)
            
            end_time = time.time()
            duration = end_time - start_time
        
            print(f"\nTranscription completed in {duration:.2f} seconds")
            print(f"Transcript saved to: {output_path}")
        
            return output_path
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise
        finally:
            for _, chunk_path in chunks:
                try:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                except:
                    pass

