# GPU ASR Migration Plan

## Current Setup Analysis
- **ASR**: FunASR Paraformer-large (CPU)
- **TTS**: XTTS v2 (GPU - RTX 3060)
- **LLM**: External HTTP API (CPU/Network)
- **Bottleneck**: CPU processing ASR chunks sequentially

## Migration Benefits

### Performance Gains
- **3-10x speed improvement** for ASR inference
- **Parallel chunk processing** instead of sequential
- **Reduced CPU load** allowing better coordination
- **Lower latency** for real-time applications

### Resource Optimization
```
Current:  CPU[ASR+Coord] → Network[LLM] → GPU[TTS]
Optimized: CPU[Coord] → GPU[ASR] → Network[LLM] → GPU[TTS]
```

### Memory Management
- RTX 3060: 12GB VRAM total
- Current TTS usage: ~2-4GB
- Available for ASR: ~6-8GB (sufficient for Paraformer-large)

## Implementation Options

### Option 1: FunASR GPU Mode (Recommended)
```python
# Enable GPU support in FunASR
model = AutoModel(
    model=model_dir,
    vad_model=vad_model_dir,
    device="cuda:0",  # Use GPU
    disable_update=True
)
```

### Option 2: Alternative GPU-Optimized Models
- **Whisper with faster-whisper**: GPU-optimized
- **SenseVoice**: Alibaba's newer GPU-optimized model
- **OpenAI Whisper v3**: Native GPU support

### Option 3: Hybrid Approach
- Keep FunASR for accuracy-critical tasks
- Add GPU Whisper for speed-critical tasks
- Dynamic switching based on performance mode

## Implementation Steps

### Phase 1: GPU Support Detection
1. Check CUDA availability and memory
2. Test FunASR GPU initialization
3. Benchmark CPU vs GPU performance

### Phase 2: Memory Management
1. Implement smart GPU memory allocation
2. Coordinate ASR and TTS GPU usage
3. Add memory cleanup between tasks

### Phase 3: Pipeline Optimization
1. Enable parallel chunk processing
2. Implement GPU-based preprocessing
3. Optimize audio format conversions

### Phase 4: Fallback Strategy
1. Automatic CPU fallback on GPU errors
2. Memory pressure detection
3. Performance monitoring

## Configuration Changes Needed

### config.py additions:
```python
# GPU ASR Configuration
ENABLE_GPU_ASR = True          # Use GPU for ASR when available
GPU_ASR_DEVICE = "cuda:0"      # GPU device for ASR
GPU_MEMORY_FRACTION = 0.6      # Fraction of GPU memory for ASR
ENABLE_ASR_TTS_INTERLEAVING = True  # Smart GPU scheduling
ASR_BATCH_SIZE = 4             # Process multiple chunks together
```

### asr_module.py modifications:
```python
class FunASREngine:
    def __init__(self, use_gpu=True, device="cuda:0"):
        self.device = device if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = AutoModel(
            model=model_dir,
            vad_model=vad_model_dir,
            device=self.device,
            disable_update=True
        )
```

## Expected Performance Improvements

### Chunk Processing Speed
- **3-second chunks**: 0.8s → 0.2s (4x improvement)
- **5-second chunks**: 1.2s → 0.3s (4x improvement)

### System Throughput
- **Real-time factor**: 0.4 → 0.1 (better than real-time)
- **Parallel processing**: 3 chunks simultaneously
- **CPU utilization**: 80% → 40% (freed for coordination)

### Latency Reduction
- **End-to-end**: 2.5s → 1.2s per chunk
- **Pipeline efficiency**: Better overlap between ASR/LLM/TTS

## Risk Mitigation

### Memory Management
- Monitor GPU memory usage
- Implement graceful fallback to CPU
- Smart memory cleanup between chunks

### Compatibility
- Test with existing model weights
- Validate accuracy against CPU baseline
- Ensure TTS compatibility

### Stability
- GPU error recovery mechanisms
- Temperature monitoring
- Automatic performance scaling

## Testing Plan

### Benchmark Tests
1. **Speed comparison**: CPU vs GPU processing times
2. **Accuracy validation**: Identical outputs verification
3. **Memory usage**: GPU VRAM consumption monitoring
4. **Stability test**: Long-running session reliability

### Integration Tests
1. **ASR-TTS coordination**: No memory conflicts
2. **Pipeline throughput**: End-to-end performance
3. **Error handling**: Graceful degradation
4. **Configuration switching**: Dynamic CPU/GPU switching

## Implementation Priority

### High Priority (Immediate)
- [ ] GPU detection and initialization
- [ ] Basic FunASR GPU mode
- [ ] Memory management framework

### Medium Priority (Next Phase)
- [ ] Parallel chunk processing
- [ ] Performance monitoring
- [ ] Advanced memory optimization

### Low Priority (Future)
- [ ] Alternative model support
- [ ] Automatic model selection
- [ ] Advanced GPU scheduling