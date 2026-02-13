"""
Plot Analysis Cache System

Caches vision LLM plot analyses to avoid re-analyzing the same plots.
Uses content-based hashing to detect if a plot has changed.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from loguru import logger


class PlotAnalysisCache:
    """
    Cache system for plot analyses.
    
    Features:
    - Content-based caching (using file hash)
    - Persistent storage (survives restarts)
    - Automatic invalidation when plots change
    - Cache statistics and management
    """
    
    def __init__(self, cache_dir: str = "cache/plot_analyses"):
        """
        Initialize the plot analysis cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = self.cache_dir / "plot_cache.json"
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()
        
        logger.info(f"📦 PlotAnalysisCache initialized: {self.cache_dir}")
        logger.info(f"   Cached analyses: {len(self.cache)}")
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                logger.info(f"✅ Loaded {len(cache)} cached analyses")
                return cache
            except Exception as e:
                logger.warning(f"⚠️ Failed to load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save cache: {e}")
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of file content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hex string of file hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks for memory efficiency
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def get(self, plot_path: str) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis for a plot.
        
        Args:
            plot_path: Path to the plot file
            
        Returns:
            Cached analysis dict or None if not cached/invalid
        """
        plot_path = Path(plot_path).resolve()
        
        if not plot_path.exists():
            logger.warning(f"⚠️ Plot file not found: {plot_path}")
            return None
        
        # Compute current file hash
        current_hash = self._compute_file_hash(plot_path)
        cache_key = str(plot_path)
        
        # Check if in cache
        if cache_key in self.cache:
            cached_entry = self.cache[cache_key]
            
            # Verify hash matches (plot hasn't changed)
            if cached_entry.get("file_hash") == current_hash:
                logger.info(f"✅ Cache HIT: {plot_path.name}")
                
                # Update access time
                cached_entry["last_accessed"] = datetime.now().isoformat()
                self._save_cache()
                
                return cached_entry["analysis"]
            else:
                logger.info(f"🔄 Cache INVALID (plot changed): {plot_path.name}")
                # Remove stale entry
                del self.cache[cache_key]
                self._save_cache()
        else:
            logger.info(f"❌ Cache MISS: {plot_path.name}")
        
        return None
    
    def set(
        self, 
        plot_path: str, 
        analysis: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Cache a plot analysis.
        
        Args:
            plot_path: Path to the plot file
            analysis: Analysis result to cache
            metadata: Optional metadata about the analysis
        """
        plot_path = Path(plot_path).resolve()
        
        if not plot_path.exists():
            logger.warning(f"⚠️ Cannot cache - plot not found: {plot_path}")
            return
        
        # Compute file hash
        file_hash = self._compute_file_hash(plot_path)
        cache_key = str(plot_path)
        
        # Create cache entry
        cache_entry = {
            "file_hash": file_hash,
            "analysis": analysis,
            "cached_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "plot_name": plot_path.name,
            "file_size": plot_path.stat().st_size,
            "metadata": metadata or {}
        }
        
        self.cache[cache_key] = cache_entry
        self._save_cache()
        
        logger.info(f"💾 Cached analysis: {plot_path.name}")
    
    def invalidate(self, plot_path: str) -> bool:
        """
        Invalidate (remove) a cached analysis.
        
        Args:
            plot_path: Path to the plot file
            
        Returns:
            True if cache entry was removed, False if not found
        """
        cache_key = str(Path(plot_path).resolve())
        
        if cache_key in self.cache:
            del self.cache[cache_key]
            self._save_cache()
            logger.info(f"🗑️ Invalidated cache: {Path(plot_path).name}")
            return True
        
        return False
    
    def clear(self):
        """Clear all cached analyses."""
        self.cache = {}
        self._save_cache()
        logger.info("🗑️ Cleared all cached analyses")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        if not self.cache:
            return {
                "total_entries": 0,
                "cache_size_mb": 0,
                "oldest_entry": None,
                "newest_entry": None
            }
        
        cached_times = [
            datetime.fromisoformat(entry["cached_at"])
            for entry in self.cache.values()
        ]
        
        total_size = sum(
            entry.get("file_size", 0)
            for entry in self.cache.values()
        )
        
        return {
            "total_entries": len(self.cache),
            "cache_size_mb": total_size / (1024 * 1024),
            "oldest_entry": min(cached_times).isoformat(),
            "newest_entry": max(cached_times).isoformat(),
            "cache_dir": str(self.cache_dir),
            "cache_file": str(self.cache_file)
        }
    
    def cleanup_old_entries(self, days: int = 30):
        """
        Remove cache entries older than specified days.
        
        Args:
            days: Remove entries older than this many days
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        old_keys = []
        for key, entry in self.cache.items():
            cached_at = datetime.fromisoformat(entry["cached_at"])
            if cached_at < cutoff_date:
                old_keys.append(key)
        
        for key in old_keys:
            del self.cache[key]
        
        if old_keys:
            self._save_cache()
            logger.info(f"🗑️ Cleaned up {len(old_keys)} old cache entries")
        else:
            logger.info("✅ No old cache entries to clean up")
    
    def get_all_cached_paths(self) -> list[str]:
        """Get list of all cached plot paths."""
        return list(self.cache.keys())
    
    def export_cache(self, export_path: str):
        """
        Export cache to a different location.
        
        Args:
            export_path: Path to export cache file
        """
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path, 'w') as f:
            json.dump(self.cache, f, indent=2, default=str)
        
        logger.info(f"📤 Exported cache to: {export_path}")


# ============================================================================
# STANDALONE TESTING
# ============================================================================

def test_cache():
    """Test the plot analysis cache."""
    import tempfile
    import shutil
    from PIL import Image
    import numpy as np
    
    print("\n" + "="*70)
    print("TESTING PLOT ANALYSIS CACHE")
    print("="*70)
    
    # Create temporary directory for testing
    temp_dir = Path(tempfile.mkdtemp())
    cache_dir = temp_dir / "cache"
    
    try:
        # Initialize cache
        cache = PlotAnalysisCache(cache_dir=str(cache_dir))
        
        # Create a test plot
        test_plot = temp_dir / "test_plot.png"
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(test_plot)
        
        # Test 1: Cache miss
        print("\n1. Testing cache miss...")
        result = cache.get(str(test_plot))
        assert result is None, "Should be cache miss"
        print("✅ Cache miss works")
        
        # Test 2: Set cache
        print("\n2. Testing cache set...")
        analysis = {
            "plot_name": "test_plot.png",
            "analysis": "This is a test analysis",
            "insights": ["Insight 1", "Insight 2"]
        }
        cache.set(str(test_plot), analysis, metadata={"model": "test-model"})
        print("✅ Cache set works")
        
        # Test 3: Cache hit
        print("\n3. Testing cache hit...")
        result = cache.get(str(test_plot))
        assert result is not None, "Should be cache hit"
        assert result["analysis"] == "This is a test analysis"
        print("✅ Cache hit works")
        
        # Test 4: Invalidation on file change
        print("\n4. Testing cache invalidation on file change...")
        # Modify the plot
        img2 = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img2.save(test_plot)
        result = cache.get(str(test_plot))
        assert result is None, "Should be cache miss after file change"
        print("✅ Cache invalidation works")
        
        # Test 5: Cache statistics
        print("\n5. Testing cache statistics...")
        cache.set(str(test_plot), analysis)  # Re-cache
        stats = cache.get_stats()
        print(f"   Total entries: {stats['total_entries']}")
        assert stats['total_entries'] == 1
        print("✅ Cache statistics work")
        
        # Test 6: Cache persistence
        print("\n6. Testing cache persistence...")
        cache2 = PlotAnalysisCache(cache_dir=str(cache_dir))
        result = cache2.get(str(test_plot))
        assert result is not None, "Cache should persist across instances"
        print("✅ Cache persistence works")
        
        # Test 7: Manual invalidation
        print("\n7. Testing manual invalidation...")
        success = cache.invalidate(str(test_plot))
        assert success, "Should successfully invalidate"
        result = cache.get(str(test_plot))
        assert result is None, "Should be cache miss after invalidation"
        print("✅ Manual invalidation works")
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✅")
        print("="*70)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_cache()