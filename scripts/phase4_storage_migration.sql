-- Phase 4: Multi-Modal Storage and Indexing Migration
-- This script creates tables for multi-modal file storage and indexing

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- ============================================================
-- MULTI-MODAL FILES TABLE
-- ============================================================
-- Stores metadata for uploaded files (images, documents, audio, video)

CREATE TABLE IF NOT EXISTS multimodal_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT UNIQUE NOT NULL,
    session_id TEXT NOT NULL,
    file_type TEXT NOT NULL,  -- 'image', 'pdf', 'docx', 'txt', 'audio', 'video'
    original_filename TEXT,
    stored_path TEXT NOT NULL,
    file_size INTEGER,
    mime_type TEXT,
    file_hash TEXT,  -- SHA256 hash for deduplication
    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT,  -- JSON: dimensions, pages, duration, etc.
    FOREIGN KEY (session_id) REFERENCES captures(session_id)
);

-- ============================================================
-- FILE ANALYSIS TABLE
-- ============================================================
-- Stores analysis results (vision, OCR, document extraction, transcription)

CREATE TABLE IF NOT EXISTS file_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT NOT NULL,
    analysis_type TEXT NOT NULL,  -- 'vision', 'ocr', 'extraction', 'transcription'
    analysis_result TEXT,  -- JSON: analysis output
    analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES multimodal_files(file_id) ON DELETE CASCADE
);

-- ============================================================
-- FILE-CONCEPT LINKS TABLE
-- ============================================================
-- Links files to knowledge graph concepts

CREATE TABLE IF NOT EXISTS file_concept_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT NOT NULL,
    concept_name TEXT NOT NULL,
    link_type TEXT,  -- 'extracted', 'mentioned', 'related'
    confidence REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES multimodal_files(file_id) ON DELETE CASCADE
);

-- ============================================================
-- FULL-TEXT SEARCH TABLE (FTS5)
-- ============================================================
-- Full-text search on file analysis results

CREATE VIRTUAL TABLE IF NOT EXISTS file_analysis_fts
USING fts5(
    file_id UNINDEXED,
    analysis_type UNINDEXED,
    analysis_text,
    content=file_analysis,
    content_rowid=id
);

-- ============================================================
-- TRIGGERS FOR FTS SYNC
-- ============================================================
-- Keep FTS5 table synchronized with file_analysis

-- Trigger: Insert
CREATE TRIGGER IF NOT EXISTS file_analysis_ai
AFTER INSERT ON file_analysis BEGIN
    INSERT INTO file_analysis_fts(rowid, file_id, analysis_type, analysis_text)
    VALUES (new.id, new.file_id, new.analysis_type, new.analysis_result);
END;

-- Trigger: Delete
CREATE TRIGGER IF NOT EXISTS file_analysis_ad
AFTER DELETE ON file_analysis BEGIN
    DELETE FROM file_analysis_fts WHERE rowid = old.id;
END;

-- Trigger: Update
CREATE TRIGGER IF NOT EXISTS file_analysis_au
AFTER UPDATE ON file_analysis BEGIN
    UPDATE file_analysis_fts
    SET analysis_text = new.analysis_result
    WHERE rowid = new.id;
END;

-- ============================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================

-- Index: Files by session and upload date
CREATE INDEX IF NOT EXISTS idx_file_session
ON multimodal_files(session_id, uploaded_at DESC);

-- Index: Files by type and upload date
CREATE INDEX IF NOT EXISTS idx_file_type
ON multimodal_files(file_type, uploaded_at DESC);

-- Index: Files by hash (for deduplication)
CREATE INDEX IF NOT EXISTS idx_file_hash
ON multimodal_files(file_hash);

-- Index: Analysis by file and type
CREATE INDEX IF NOT EXISTS idx_analysis_file
ON file_analysis(file_id, analysis_type);

-- Index: Concept links by file
CREATE INDEX IF NOT EXISTS idx_concept_links
ON file_concept_links(file_id, concept_name);

-- Index: Concept links by concept
CREATE INDEX IF NOT EXISTS idx_concept_name
ON file_concept_links(concept_name, file_id);

-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================

-- Verify tables created
SELECT name FROM sqlite_master
WHERE type='table'
AND name LIKE 'multimodal_files' OR name LIKE 'file_analysis' OR name LIKE 'file_concept_links';

-- Verify indexes created
SELECT name FROM sqlite_master
WHERE type='index'
AND name LIKE 'idx_file_%' OR name LIKE 'idx_analysis_%' OR name LIKE 'idx_concept_%';

-- Verify triggers created
SELECT name FROM sqlite_master
WHERE type='trigger'
AND name LIKE 'file_analysis_%';

-- ============================================================
-- SAMPLE QUERIES
-- ============================================================

-- Get all files for a session
-- SELECT * FROM multimodal_files WHERE session_id = ? ORDER BY uploaded_at DESC;

-- Get all analysis for a file
-- SELECT * FROM file_analysis WHERE file_id = ? ORDER BY analyzed_at DESC;

-- Full-text search on analysis results
-- SELECT fa.*, mf.original_filename, mf.file_type
-- FROM file_analysis fa
-- JOIN file_analysis_fts ON fa.id = file_analysis_fts.rowid
-- JOIN multimodal_files mf ON fa.file_id = mf.file_id
-- WHERE file_analysis_fts MATCH ?
-- ORDER BY rank;

-- Get files linked to a concept
-- SELECT mf.*, fcl.link_type, fcl.confidence
-- FROM multimodal_files mf
-- JOIN file_concept_links fcl ON mf.file_id = fcl.file_id
-- WHERE fcl.concept_name = ?
-- ORDER BY fcl.confidence DESC;

-- Find duplicate files (by hash)
-- SELECT file_hash, COUNT(*) as count, GROUP_CONCAT(file_id) as file_ids
-- FROM multimodal_files
-- WHERE file_hash IS NOT NULL
-- GROUP BY file_hash
-- HAVING count > 1;

-- Storage statistics
-- SELECT
--     file_type,
--     COUNT(*) as file_count,
--     SUM(file_size) as total_bytes,
--     ROUND(SUM(file_size) / 1024.0 / 1024.0, 2) as total_mb
-- FROM multimodal_files
-- GROUP BY file_type;
