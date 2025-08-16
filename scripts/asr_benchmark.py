#!/usr/bin/env python3
"""
🏆 ASR Performance Benchmark
Compare CPU vs GPU ASR processing speeds
"""

import os
import sys
import time
import numpy as np

# Add core directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(current_dir), 'core')
sys.path.insert(0, core_dir)

import config
from asr_module import FunASREngine

def generate_test_audio(duration_seconds=3, sample_rate=16000):
    """Generate synthetic test audio for benchmarking"""
    # Create realistic audio-like noise (better than pure sine wave for ASR testing)
    samples = int(duration_seconds * sample_rate)
    
    # Mix of frequencies to simulate speech
    t = np.linspace(0, duration_seconds, samples)
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +  # Low frequency
        0.2 * np.sin(2 * np.pi * 800 * t) +  # Mid frequency  
        0.1 * np.sin(2 * np.pi * 1600 * t) + # High frequency
        0.05 * np.random.normal(0, 1, samples)  # Background noise
    )
    
    # Normalize and convert to int16
    audio = np.clip(audio, -1, 1)
    audio = (audio * 32767).astype(np.int16)
    return audio

def benchmark_asr_engine(engine, test_audio_chunks, chunk_names):
    """Benchmark an ASR engine with multiple audio chunks"""
    if not engine.model:
        return None
        
    device_name = f"GPU ({engine.device})" if engine.gpu_available else "CPU"
    print(f"\n=== Benchmarking {device_name} ASR ===")
    
    results = {
        'device': device_name,
        'gpu_available': engine.gpu_available,
        'times': [],
        'texts': [],
        'total_time': 0,
        'avg_time': 0,
        'chunks_per_second': 0
    }
    
    # Warm-up run (not counted in results)
    if test_audio_chunks:
        print("   Warming up...")
        engine.transcribe_chunk(test_audio_chunks[0])
    
    # Benchmark runs
    print("   Running benchmark...")
    start_total = time.time()
    
    for i, (audio_chunk, chunk_name) in enumerate(zip(test_audio_chunks, chunk_names)):
        start_time = time.time()
        text = engine.transcribe_chunk(audio_chunk)
        end_time = time.time()
        
        processing_time = end_time - start_time
        results['times'].append(processing_time)
        results['texts'].append(text[:50] + "..." if len(text) > 50 else text)
        
        print(f"   Chunk {i+1}/{len(test_audio_chunks)}: {processing_time:.3f}s")
        if text and not config.MINIMAL_LOGGING:
            print(f"      Text: {text[:30]}...")
    
    end_total = time.time()
    results['total_time'] = end_total - start_total
    results['avg_time'] = np.mean(results['times'])
    results['chunks_per_second'] = len(test_audio_chunks) / results['total_time']
    
    return results

def print_benchmark_results(cpu_results, gpu_results):
    """Print comparison of benchmark results"""
    print("\n" + "=" * 60)
    print("🏆 ASR PERFORMANCE BENCHMARK RESULTS")
    print("=" * 60)
    
    if cpu_results:
        print(f"\n📊 CPU Performance:")
        print(f"   Average processing time: {cpu_results['avg_time']:.3f}s per chunk")
        print(f"   Total processing time:   {cpu_results['total_time']:.3f}s")
        print(f"   Throughput:             {cpu_results['chunks_per_second']:.2f} chunks/second")
        
    if gpu_results:
        print(f"\n🚀 GPU Performance:")
        print(f"   Average processing time: {gpu_results['avg_time']:.3f}s per chunk")
        print(f"   Total processing time:   {gpu_results['total_time']:.3f}s")
        print(f"   Throughput:             {gpu_results['chunks_per_second']:.2f} chunks/second")
        
    if cpu_results and gpu_results:
        speedup = cpu_results['avg_time'] / gpu_results['avg_time']
        throughput_improvement = gpu_results['chunks_per_second'] / cpu_results['chunks_per_second']
        
        print(f"\n📈 Performance Improvement:")
        print(f"   Speed improvement:      {speedup:.2f}x faster")
        print(f"   Throughput improvement: {throughput_improvement:.2f}x more chunks/second")
        
        if speedup > 1:
            print(f"   ✅ GPU is {speedup:.1f}x faster than CPU")
        else:
            print(f"   ⚠️ CPU is {1/speedup:.1f}x faster than GPU (unexpected)")
            
        # Real-time factor analysis
        chunk_duration = 3 if config.OPTIMIZED_CHUNK_SIZE else 5
        cpu_rtf = cpu_results['avg_time'] / chunk_duration
        gpu_rtf = gpu_results['avg_time'] / chunk_duration
        
        print(f"\n⏱️ Real-time Factor Analysis:")
        print(f"   CPU real-time factor:   {cpu_rtf:.3f} ({'✅ real-time' if cpu_rtf < 1 else '❌ slower than real-time'})")
        print(f"   GPU real-time factor:   {gpu_rtf:.3f} ({'✅ real-time' if gpu_rtf < 1 else '❌ slower than real-time'})")
        
    print("\n" + "=" * 60)

def main():
    """Main benchmark function"""
    print("🏆 S2T2SS ASR Performance Benchmark")
    print("=" * 50)
    
    # Configuration
    num_test_chunks = 5
    chunk_duration = 3 if config.OPTIMIZED_CHUNK_SIZE else 5
    
    print(f"Benchmark Configuration:")
    print(f"   Number of test chunks: {num_test_chunks}")
    print(f"   Chunk duration: {chunk_duration} seconds")
    print(f"   Total test audio: {num_test_chunks * chunk_duration} seconds")
    
    # Generate test audio
    print(f"\nGenerating test audio...")
    test_audio_chunks = []
    chunk_names = []
    
    for i in range(num_test_chunks):
        audio = generate_test_audio(duration_seconds=chunk_duration)
        test_audio_chunks.append(audio)
        chunk_names.append(f"Test_{i+1}_{chunk_duration}s")
        
    print(f"   Generated {len(test_audio_chunks)} audio chunks")
    
    # Test CPU ASR
    print(f"\n1️⃣ Testing CPU ASR...")
    cpu_engine = FunASREngine(use_gpu=False)
    cpu_results = benchmark_asr_engine(cpu_engine, test_audio_chunks, chunk_names)
    
    # Test GPU ASR
    print(f"\n2️⃣ Testing GPU ASR...")
    gpu_engine = FunASREngine(use_gpu=True)
    gpu_results = benchmark_asr_engine(gpu_engine, test_audio_chunks, chunk_names)
    
    # Print comparison
    print_benchmark_results(cpu_results, gpu_results)
    
    # Recommendations
    print("\n💡 Recommendations:")
    if gpu_results and gpu_results['gpu_available']:
        if cpu_results and gpu_results['avg_time'] < cpu_results['avg_time']:
            improvement = cpu_results['avg_time'] / gpu_results['avg_time']
            print(f"   ✅ Enable GPU ASR for {improvement:.1f}x performance improvement")
            print(f"   ✅ Better real-time performance for live transcription")
            print(f"   ✅ Reduced CPU load allows better coordination")
        else:
            print(f"   ⚠️ GPU ASR not significantly faster - CPU mode may be sufficient")
    else:
        print(f"   ❌ GPU ASR not available - stick with CPU mode")
    
    return cpu_results, gpu_results

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()