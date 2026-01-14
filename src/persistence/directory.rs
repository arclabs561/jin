//! Directory abstraction for persistence.
//!
//! Provides a trait-based abstraction over storage backends (filesystem, memory, S3, etc.)
//! enabling flexible persistence implementations.

use crate::persistence::error::{PersistenceError, PersistenceResult};
use std::io::{Read, Write};
use std::path::PathBuf;

/// Trait for directory-like storage backends.
///
/// This abstraction allows ordino-retrieve to work with different storage backends:
/// - Filesystem (local disk)
/// - Memory (for testing, ephemeral indexes)
/// - Cloud storage (S3, GCS - future)
/// - Network filesystems (NFS, etc.)
pub trait Directory: Send + Sync {
    /// Create a new file for writing.
    ///
    /// Returns a writer that will write to the specified path.
    /// The file should not exist yet (or will be overwritten).
    fn create_file(&self, path: &str) -> PersistenceResult<Box<dyn Write>>;

    /// Open an existing file for reading.
    ///
    /// Returns a reader for the specified path.
    fn open_file(&self, path: &str) -> PersistenceResult<Box<dyn Read>>;

    /// Check if a file or directory exists.
    fn exists(&self, path: &str) -> bool;

    /// Delete a file or directory.
    ///
    /// For directories, should recursively delete all contents.
    fn delete(&self, path: &str) -> PersistenceResult<()>;

    /// Atomically rename/move a file.
    ///
    /// This operation should be atomic (all-or-nothing) on the underlying storage.
    /// On POSIX systems, this uses `rename()` which is atomic.
    /// On other systems, may require copy + delete (less safe).
    fn atomic_rename(&self, from: &str, to: &str) -> PersistenceResult<()>;

    /// Create a directory (and parent directories if needed).
    fn create_dir_all(&self, path: &str) -> PersistenceResult<()>;

    /// List files in a directory.
    ///
    /// Returns paths relative to the directory root.
    fn list_dir(&self, path: &str) -> PersistenceResult<Vec<String>>;

    /// Open a file for appending.
    ///
    /// Returns a writer that will append to the existing file.
    /// If the file doesn't exist, creates it.
    fn append_file(&self, path: &str) -> PersistenceResult<Box<dyn Write>>;

    /// Atomically write data to a file using the fsync + rename pattern.
    ///
    /// This ensures crash-safe writes by:
    /// 1. Writing to a temporary file
    /// 2. Flushing and syncing the temporary file
    /// 3. Atomically renaming the temporary file to the final path
    ///
    /// This pattern is used for critical files like checkpoints, metadata, and segment footers.
    fn atomic_write(&self, path: &str, data: &[u8]) -> PersistenceResult<()>;

    /// Get the file path for memory mapping (optional).
    ///
    /// Returns `None` if the directory doesn't support memory mapping (e.g., MemoryDirectory).
    /// For FsDirectory, returns the absolute path to the file.
    ///
    /// This allows callers to use memory mapping for efficient read-only access to large files.
    fn file_path(&self, path: &str) -> Option<PathBuf>;
}

/// Filesystem-based directory implementation.
///
/// Stores all files on the local filesystem under a root directory.
pub struct FsDirectory {
    root: PathBuf,
}

impl FsDirectory {
    /// Create a new filesystem directory at the specified path.
    ///
    /// The directory will be created if it doesn't exist.
    pub fn new<P: Into<PathBuf>>(root: P) -> PersistenceResult<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// Get the root path.
    pub fn root(&self) -> &PathBuf {
        &self.root
    }

    /// Resolve a relative path to an absolute path.
    fn resolve_path(&self, path: &str) -> PathBuf {
        self.root.join(path)
    }
}

impl Directory for FsDirectory {
    fn create_file(&self, path: &str) -> PersistenceResult<Box<dyn Write>> {
        let full_path = self.resolve_path(path);
        // Create parent directories if needed
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(full_path)?;
        Ok(Box::new(file))
    }

    fn open_file(&self, path: &str) -> PersistenceResult<Box<dyn Read>> {
        let full_path = self.resolve_path(path);
        let file = std::fs::File::open(full_path)?;
        Ok(Box::new(file))
    }

    fn exists(&self, path: &str) -> bool {
        self.resolve_path(path).exists()
    }

    fn delete(&self, path: &str) -> PersistenceResult<()> {
        let full_path = self.resolve_path(path);
        if full_path.is_dir() {
            std::fs::remove_dir_all(full_path)?;
        } else {
            std::fs::remove_file(full_path)?;
        }
        Ok(())
    }

    fn atomic_rename(&self, from: &str, to: &str) -> PersistenceResult<()> {
        let from_path = self.resolve_path(from);
        let to_path = self.resolve_path(to);
        // Create parent directories for destination if needed
        if let Some(parent) = to_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::rename(from_path, to_path)?;
        Ok(())
    }

    fn create_dir_all(&self, path: &str) -> PersistenceResult<()> {
        let full_path = self.resolve_path(path);
        std::fs::create_dir_all(full_path)?;
        Ok(())
    }

    fn list_dir(&self, path: &str) -> PersistenceResult<Vec<String>> {
        let full_path = self.resolve_path(path);
        let entries = std::fs::read_dir(full_path)?;
        let mut paths = Vec::new();
        for entry in entries {
            let entry = entry?;
            let file_name = entry.file_name();
            paths.push(file_name.to_string_lossy().to_string());
        }
        Ok(paths)
    }

    fn append_file(&self, path: &str) -> PersistenceResult<Box<dyn Write>> {
        let full_path = self.resolve_path(path);
        // Create parent directories if needed
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(full_path)?;
        Ok(Box::new(file))
    }

    fn atomic_write(&self, path: &str, data: &[u8]) -> PersistenceResult<()> {
        let temp_path = format!("{}.tmp", path);
        let full_temp_path = self.resolve_path(&temp_path);

        // Create parent directories if needed
        if let Some(parent) = full_temp_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Write to temporary file
        let mut temp_file = std::fs::File::create(&full_temp_path)?;
        temp_file.write_all(data)?;
        temp_file.sync_all()?; // fsync for durability

        // Atomically rename
        let full_path = self.resolve_path(path);
        std::fs::rename(&full_temp_path, &full_path)?;

        // Sync parent directory (ensures rename is durable on some filesystems)
        if let Some(parent) = full_path.parent() {
            if let Ok(parent_file) = std::fs::File::open(parent) {
                let _ = parent_file.sync_all(); // Best effort
            }
        }

        Ok(())
    }

    fn file_path(&self, path: &str) -> Option<PathBuf> {
        Some(self.resolve_path(path))
    }
}

/// In-memory directory implementation (for testing).
///
/// Stores all files in memory as byte vectors.
/// Useful for testing and ephemeral indexes.
#[derive(Clone)]
pub struct MemoryDirectory {
    files: std::sync::Arc<std::sync::RwLock<std::collections::HashMap<String, Vec<u8>>>>,
}

impl MemoryDirectory {
    pub fn new() -> Self {
        Self {
            files: std::sync::Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
        }
    }
}

impl Default for MemoryDirectory {
    fn default() -> Self {
        Self::new()
    }
}

impl Directory for MemoryDirectory {
    fn create_file(&self, path: &str) -> PersistenceResult<Box<dyn Write>> {
        let files = self.files.clone();
        let path = path.to_string();
        Ok(Box::new(MemoryWriter {
            files,
            path,
            buffer: Vec::new(),
        }))
    }

    fn open_file(&self, path: &str) -> PersistenceResult<Box<dyn Read>> {
        let files = self
            .files
            .read()
            .map_err(|_| PersistenceError::LockFailed {
                resource: "memory files".into(),
                reason: "lock poisoned".into(),
            })?;
        let data = files
            .get(path)
            .ok_or_else(|| PersistenceError::NotFound(path.to_string()))?
            .clone();
        Ok(Box::new(std::io::Cursor::new(data)))
    }

    fn exists(&self, path: &str) -> bool {
        self.files
            .read()
            .map(|f| f.contains_key(path))
            .unwrap_or(false)
    }

    fn delete(&self, path: &str) -> PersistenceResult<()> {
        self.files
            .write()
            .map_err(|_| PersistenceError::LockFailed {
                resource: "memory files".into(),
                reason: "lock poisoned".into(),
            })?
            .remove(path);
        Ok(())
    }

    fn atomic_rename(&self, from: &str, to: &str) -> PersistenceResult<()> {
        let mut files = self
            .files
            .write()
            .map_err(|_| PersistenceError::LockFailed {
                resource: "memory files".into(),
                reason: "lock poisoned".into(),
            })?;
        if let Some(data) = files.remove(from) {
            files.insert(to.to_string(), data);
        }
        Ok(())
    }

    fn create_dir_all(&self, _path: &str) -> PersistenceResult<()> {
        // No-op for memory directory
        Ok(())
    }

    fn list_dir(&self, path: &str) -> PersistenceResult<Vec<String>> {
        // Return files that start with the path prefix
        let files = self
            .files
            .read()
            .map_err(|_| PersistenceError::LockFailed {
                resource: "memory files".into(),
                reason: "lock poisoned".into(),
            })?;
        let prefix = if path.is_empty() {
            "".to_string()
        } else {
            format!("{}/", path)
        };
        let mut result: Vec<String> = files
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .map(|k| {
                // Remove the prefix and return just the filename
                k.strip_prefix(&prefix).unwrap_or(k).to_string()
            })
            .collect();
        result.sort();
        Ok(result)
    }

    fn append_file(&self, path: &str) -> PersistenceResult<Box<dyn Write>> {
        // For memory directory, append means read existing, append, write back
        let files = self.files.clone();
        let path = path.to_string();
        let existing = self
            .files
            .read()
            .map_err(|_| PersistenceError::LockFailed {
                resource: "memory files".into(),
                reason: "lock poisoned".into(),
            })?
            .get(&path)
            .cloned()
            .unwrap_or_default();
        Ok(Box::new(MemoryAppendWriter {
            files,
            path,
            buffer: existing,
        }))
    }

    fn atomic_write(&self, path: &str, data: &[u8]) -> PersistenceResult<()> {
        // For memory directory, atomic_write is just a regular write
        // (no need for temp file + rename in memory)
        let mut files = self
            .files
            .write()
            .map_err(|_| PersistenceError::LockFailed {
                resource: "memory files".into(),
                reason: "lock poisoned".into(),
            })?;
        files.insert(path.to_string(), data.to_vec());
        Ok(())
    }

    fn file_path(&self, _path: &str) -> Option<PathBuf> {
        // Memory directory doesn't have real file paths
        None
    }
}

struct MemoryWriter {
    files: std::sync::Arc<std::sync::RwLock<std::collections::HashMap<String, Vec<u8>>>>,
    path: String,
    buffer: Vec<u8>,
}

impl Write for MemoryWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let mut files = self
            .files
            .write()
            .map_err(|_| std::io::Error::other("lock poisoned"))?;
        files.insert(self.path.clone(), self.buffer.clone());
        Ok(())
    }
}

impl Drop for MemoryWriter {
    fn drop(&mut self) {
        // Ensure file is written on drop
        let _ = self.flush();
    }
}

struct MemoryAppendWriter {
    files: std::sync::Arc<std::sync::RwLock<std::collections::HashMap<String, Vec<u8>>>>,
    path: String,
    buffer: Vec<u8>,
}

impl Write for MemoryAppendWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let mut files = self
            .files
            .write()
            .map_err(|_| std::io::Error::other("lock poisoned"))?;
        files.insert(self.path.clone(), self.buffer.clone());
        Ok(())
    }
}

impl Drop for MemoryAppendWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fs_directory() {
        // Use a unique temp directory with timestamp/pid to avoid conflicts
        let temp_dir = std::env::temp_dir().join(format!("plesio_test_{}", std::process::id()));
        // Clean up any leftover from previous runs
        let _ = std::fs::remove_dir_all(&temp_dir);
        let dir = FsDirectory::new(&temp_dir).unwrap();

        // Test create and write
        let mut file = dir.create_file("test.txt").unwrap();
        file.write_all(b"hello").unwrap();
        drop(file);

        // Test read
        let mut file = dir.open_file("test.txt").unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        assert_eq!(contents, "hello");

        // Test exists
        assert!(dir.exists("test.txt"));

        // Test atomic rename
        dir.atomic_rename("test.txt", "renamed.txt").unwrap();
        assert!(!dir.exists("test.txt"));
        assert!(dir.exists("renamed.txt"));

        // Test delete
        dir.delete("renamed.txt").unwrap();
        assert!(!dir.exists("renamed.txt"));

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_memory_directory() {
        let dir = MemoryDirectory::new();

        // Test create and write
        let mut file = dir.create_file("test.txt").unwrap();
        file.write_all(b"hello").unwrap();
        file.flush().unwrap();

        // Test read
        let mut file = dir.open_file("test.txt").unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        assert_eq!(contents, "hello");

        // Test exists
        assert!(dir.exists("test.txt"));

        // Test atomic rename
        dir.atomic_rename("test.txt", "renamed.txt").unwrap();
        assert!(!dir.exists("test.txt"));
        assert!(dir.exists("renamed.txt"));

        // Test delete
        dir.delete("renamed.txt").unwrap();
        assert!(!dir.exists("renamed.txt"));
    }
}
