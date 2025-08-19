import os
import shutil

def sync_files(source_file, dest_file):
    """Copy content from source to destination if they differ."""
    if not os.path.exists(source_file):
        print(f"Source file {source_file} not found.")
        return

    # If dest_file doesn't exist, just copy
    if not os.path.exists(dest_file):
        shutil.copy2(source_file, dest_file)
        print(f"Copied {source_file} to {dest_file}")
        return

    # Compare files
    with open(source_file, 'r') as f1, open(dest_file, 'r') as f2:
        if f1.read() == f2.read():
            print(f"{source_file} and {dest_file} are already in sync.")
            return

    # If different, copy from source to dest
    shutil.copy2(source_file, dest_file)
    print(f"Synced {source_file} to {dest_file}")

if __name__ == "__main__":
    crush_md = "CRUSH.md"
    claude_md = "CLAUDE.md"
    docs_dir = "docs"

    # Ensure docs directory exists
    if not os.path.isdir(docs_dir):
        os.makedirs(docs_dir)

    # Sync CRUSH.md
    sync_files(crush_md, os.path.join(docs_dir, crush_md))
    sync_files(os.path.join(docs_dir, crush_md), crush_md)

    # Sync CLAUDE.md
    sync_files(claude_md, os.path.join(docs_dir, claude_md))
    sync_files(os.path.join(docs_dir, claude_md), claude_md)
