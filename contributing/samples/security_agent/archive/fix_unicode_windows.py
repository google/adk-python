#!/usr/bin/env python3
"""
Script to fix Unicode issues for Windows compatibility.
Replaces problematic Unicode emojis with ASCII equivalents.
"""

import os
import re
from pathlib import Path

# Define Unicode emoji replacements for Windows compatibility
UNICODE_REPLACEMENTS = {
    "🎯": "[TARGET]",
    "✅": "[OK]",
    "❌": "[ERROR]",
    "⚠️": "[WARNING]", 
    "🔄": "[REFRESH]",
    "🚀": "[STARTING]",
    "🛑": "[STOPPED]",
    "🔍": "[SEARCH]",
    "📊": "[STATS]",
    "🔐": "[SECURITY]",
    "📋": "[INFO]",
    "🏥": "[HEALTH]",
    "🔧": "[CONFIG]",
    "🌐": "[NETWORK]",
    "🛡️": "[SHIELD]",
    "🚫": "[BLOCKED]",
    "🔗": "[LINK]",
    "📂": "[FOLDER]",
    "🌍": "[GLOBE]",
    "💻": "[LOCAL]",
    "☁️": "[CLOUD]",
    "📦": "[PACKAGE]",
    "🆕": "[NEW]",
    "🔴": "[CRITICAL]",
    "🟠": "[HIGH]",
    "🟡": "[MEDIUM]",
    "🟢": "[LOW]",
    "ℹ️": "[INFO]",
    "💡": "[TIP]",
    "📈": "[TRENDING]",
    "🎉": "[SUCCESS]",
    "📱": "[MOBILE]",
    "♿": "[ACCESSIBLE]",
    "🕐": "[TIME]",
    "📅": "[DATE]",
    "💾": "[SAVE]",
    "📄": "[FILE]",
    "🎭": "[TESTING]",
    "•": "*"  # Unicode bullet to ASCII asterisk
}

def fix_file_unicode(file_path: Path):
    """Fix Unicode issues in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace Unicode characters
        for unicode_char, replacement in UNICODE_REPLACEMENTS.items():
            content = content.replace(unicode_char, replacement)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed Unicode in: {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Fix Unicode issues in all Python files."""
    backend_dir = Path(__file__).parent / "backend"
    
    if not backend_dir.exists():
        print("Backend directory not found!")
        return
    
    fixed_count = 0
    total_count = 0
    
    # Process all Python files in backend
    for py_file in backend_dir.rglob("*.py"):
        total_count += 1
        if fix_file_unicode(py_file):
            fixed_count += 1
    
    print(f"\nProcessed {total_count} files, fixed {fixed_count} files")
    
    if fixed_count > 0:
        print("\nUnicode emojis have been replaced with ASCII equivalents for Windows compatibility.")
        print("Your Windows teammate should now be able to deploy without Unicode errors.")
    else:
        print("\nNo Unicode replacements needed.")

if __name__ == "__main__":
    main()