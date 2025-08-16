# 📁 S2T2SS Archive

This folder contains historical scripts and tools that have been integrated into the main system or are no longer actively used.

## 🧹 cleanup_chinese_spaces.py
- **Purpose**: Standalone script to remove unnecessary spaces between Chinese characters in caption files
- **Status**: Archived (functionality integrated into `caption_manager.py`)
- **Date Archived**: 2025-08-15
- **Integration**: Chinese space cleaning is now automatic via `CaptionManager._clean_chinese_spaces()` method
- **Usage**: Can still be run manually for historical file cleanup if needed

## 📝 Notes
- Scripts in this archive are preserved for reference and emergency use
- The main system now handles these functions automatically
- Check the main codebase before using archived scripts