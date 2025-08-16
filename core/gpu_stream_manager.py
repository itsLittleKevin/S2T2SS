#!/usr/bin/env python3
"""
🚀 GPU Stream Manager for S2T2SS
Manages concurrent GPU operations using CUDA streams to prevent conflicts
"""

import torch
import threading
import queue
import time
from typing import Optional, Dict, Any
import sys
import os

# Add core directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

class GPUStreamManager:
    """
    Manages GPU operations using CUDA streams for safe concurrent execution
    """
    
    def __init__(self):
        """Initialize GPU stream manager"""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.is_cuda = torch.cuda.is_available()
        
        if self.is_cuda:
            # Create separate streams for ASR and TTS
            self.asr_stream = torch.cuda.Stream()
            self.tts_stream = torch.cuda.Stream()
            self.default_stream = torch.cuda.default_stream()
            
            print(f"GPU Stream Manager initialized with CUDA streams")
            print(f"   ASR Stream: {self.asr_stream}")
            print(f"   TTS Stream: {self.tts_stream}")
        else:
            self.asr_stream = None
            self.tts_stream = None
            print("⚠️ GPU Stream Manager: CUDA not available, using CPU")
        
        # Operation tracking
        self.active_operations = {
            'asr': 0,
            'tts': 0
        }
        self.operation_lock = threading.Lock()
        
    def execute_asr_operation(self, operation_func, *args, **kwargs):
        """
        Execute ASR operation in dedicated stream
        
        Args:
            operation_func: Function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            Result of the operation
        """
        if not self.is_cuda:
            return operation_func(*args, **kwargs)
        
        with self.operation_lock:
            self.active_operations['asr'] += 1
        
        try:
            with torch.cuda.stream(self.asr_stream):
                # Synchronize with previous operations in this stream
                self.asr_stream.synchronize()
                
                # Execute the operation
                result = operation_func(*args, **kwargs)
                
                # Ensure operation completes before returning
                self.asr_stream.synchronize()
                
                return result
                
        finally:
            with self.operation_lock:
                self.active_operations['asr'] -= 1
    
    def execute_tts_operation(self, operation_func, *args, **kwargs):
        """
        Execute TTS operation in dedicated stream
        
        Args:
            operation_func: Function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            Result of the operation
        """
        if not self.is_cuda:
            return operation_func(*args, **kwargs)
        
        with self.operation_lock:
            self.active_operations['tts'] += 1
        
        try:
            with torch.cuda.stream(self.tts_stream):
                # Synchronize with previous operations in this stream
                self.tts_stream.synchronize()
                
                # Execute the operation
                result = operation_func(*args, **kwargs)
                
                # Ensure operation completes before returning
                self.tts_stream.synchronize()
                
                return result
                
        finally:
            with self.operation_lock:
                self.active_operations['tts'] -= 1
    
    def synchronize_all_streams(self):
        """Synchronize all GPU streams"""
        if not self.is_cuda:
            return
            
        self.asr_stream.synchronize()
        self.tts_stream.synchronize()
        torch.cuda.synchronize()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if not self.is_cuda:
            return {'allocated': 0, 'reserved': 0, 'total': 0}
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return {
            'allocated': allocated,
            'reserved': reserved, 
            'total': total,
            'free': total - allocated
        }
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        if not self.is_cuda:
            return
            
        # Synchronize all streams before cleanup
        self.synchronize_all_streams()
        
        # Empty cache
        torch.cuda.empty_cache()
        
        print(f"GPU streams cleaned up successfully")
    
    def cleanup(self):
        """
        Clean up GPU streams and resources (alias for cleanup_memory)
        """
        self.cleanup_memory()
    
    def __del__(self):
        """
        Destructor to ensure cleanup when object is destroyed
        """
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of GPU operations"""
        memory = self.get_memory_usage()
        
        return {
            'cuda_available': self.is_cuda,
            'device': str(self.device),
            'active_asr_ops': self.active_operations['asr'],
            'active_tts_ops': self.active_operations['tts'],
            'memory_allocated_gb': memory['allocated'],
            'memory_free_gb': memory['free'],
            'memory_total_gb': memory['total']
        }

# Global GPU stream manager instance
_gpu_manager = None

def get_gpu_manager() -> GPUStreamManager:
    """Get or create the global GPU stream manager"""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUStreamManager()
    return _gpu_manager

def asr_gpu_operation(func):
    """Decorator for ASR GPU operations"""
    def wrapper(*args, **kwargs):
        manager = get_gpu_manager()
        return manager.execute_asr_operation(func, *args, **kwargs)
    return wrapper

def tts_gpu_operation(func):
    """Decorator for TTS GPU operations"""
    def wrapper(*args, **kwargs):
        manager = get_gpu_manager()
        return manager.execute_tts_operation(func, *args, **kwargs)
    return wrapper

if __name__ == "__main__":
    print("🚀 Testing GPU Stream Manager")
    print("=" * 50)
    
    manager = GPUStreamManager()
    status = manager.get_status()
    
    print("GPU Manager Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    if manager.is_cuda:
        print(f"\n🧪 Testing stream operations...")
        
        def dummy_asr_operation():
            time.sleep(0.1)
            return "ASR result"
        
        def dummy_tts_operation():
            time.sleep(0.1) 
            return "TTS result"
        
        # Test concurrent operations
        import threading
        
        def test_asr():
            result = manager.execute_asr_operation(dummy_asr_operation)
            print(f"ASR completed: {result}")
        
        def test_tts():
            result = manager.execute_tts_operation(dummy_tts_operation)
            print(f"TTS completed: {result}")
        
        # Run both simultaneously
        asr_thread = threading.Thread(target=test_asr)
        tts_thread = threading.Thread(target=test_tts)
        
        start_time = time.time()
        asr_thread.start()
        tts_thread.start()
        
        asr_thread.join()
        tts_thread.join()
        
        end_time = time.time()
        print(f"✅ Concurrent operations completed in {end_time - start_time:.2f}s")
    else:
        print("⚠️ CUDA not available for testing")