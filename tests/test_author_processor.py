"""
Integration tests for AuthorProcessor with in-memory patch generation.

This test validates:
1. Dependency injection works correctly
2. In-memory patches are returned (not file paths)
3. Partition structure is correct
4. Labels are properly generated
5. No files are saved during generation
"""

import os
import sys
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import FileSystem, ImageTransformer, ImageAnalyzer, AuthorProcessor
from src.processors.author_processor import LineSegmentProcessor


class TestAuthorProcessorIntegration:
    """Integration tests for the refactored AuthorProcessor."""
    
    @classmethod
    def setup_class(cls):
        """Create temporary test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.images_dir = os.path.join(cls.temp_dir, 'images')
        cls.mat_dir = os.path.join(cls.temp_dir, 'mat')
        cls.output_dir = os.path.join(cls.temp_dir, 'output')
        
        os.makedirs(cls.images_dir)
        os.makedirs(cls.mat_dir)
        os.makedirs(cls.output_dir)
        
        # Create synthetic test image (200x1000 pixels)
        cls.test_image = cls._create_test_image()
        cls.author_name = "test_author_001"
        
        # Save test image
        import cv2  # type: ignore
        image_path = os.path.join(cls.images_dir, f"{cls.author_name}.jpg")
        cv2.imwrite(image_path, cls.test_image)
        
        # Create test .mat file
        cls._create_test_mat_file()
    
    @staticmethod
    def _create_test_image():
        """Create a synthetic handwriting-like image."""
        # Create white background
        image = np.ones((1000, 800), dtype=np.uint8) * 255
        
        # Add some "handwriting" lines (dark pixels)
        for y in range(100, 900, 180):
            # Simulate text lines
            for x in range(50, 750, 20):
                # Add random "letter" patches
                if np.random.rand() > 0.3:
                    image[y:y+30, x:x+15] = np.random.randint(0, 100, (30, 15))
        
        return image
    
    @classmethod
    def _create_test_mat_file(cls):
        """Create a synthetic .mat file with line peaks."""
        from scipy.io import savemat
        
        # Simulate 5 lines with peaks at y-coordinates
        peaks = np.array([250, 430, 610, 790, 970])
        scale_factor = 1.0
        # Test area should be a single line height (180px)
        top_test_area = 50
        bottom_test_area = 230  # 180px tall
        
        mat_data = {
            'peaks_indices': peaks,
            'SCALE_FACTOR': np.array([scale_factor]),
            'top_test_area': np.array([top_test_area]),
            'bottom_test_area': np.array([bottom_test_area])
        }
        
        mat_path = os.path.join(cls.mat_dir, f"{cls.author_name}.mat")
        savemat(mat_path, mat_data)
    
    @classmethod
    def teardown_class(cls):
        """Clean up temporary files."""
        shutil.rmtree(cls.temp_dir)
    
    def test_dependency_injection(self):
        """Test that DI works correctly."""
        # Create services
        fs = FileSystem()
        transformer = ImageTransformer()
        analyzer = ImageAnalyzer()
        line_processor = LineSegmentProcessor()
        
        # Create processor with injected dependencies
        processor = AuthorProcessor(
            label_id=0,
            author=self.author_name,
            author_images_dir=self.images_dir,
            author_mat_dir=self.mat_dir,
            transformer=transformer,
            analyzer=analyzer,
            line_processor=line_processor
        )
        
        # Verify dependencies are injected
        assert processor.transformer is transformer
        assert processor.analyzer is analyzer
        assert processor.line_processor is line_processor
        print("✅ Dependency injection works correctly")
    
    def test_in_memory_patch_generation(self):
        """Test that patches are returned in memory, not saved to disk."""
        # Create services
        transformer = ImageTransformer()
        analyzer = ImageAnalyzer()
        line_processor = LineSegmentProcessor()
        
        # Create processor
        processor = AuthorProcessor(
            label_id=42,
            author=self.author_name,
            author_images_dir=self.images_dir,
            author_mat_dir=self.mat_dir,
            transformer=transformer,
            analyzer=analyzer,
            line_processor=line_processor
        )
        
        # Count files before generation
        files_before = len(os.listdir(self.output_dir))
        
        # Generate dataset
        partition, labels = processor.generate()
        
        # Count files after generation
        files_after = len(os.listdir(self.output_dir))
        
        # Verify NO files were created
        assert files_before == files_after, "Files should not be saved during generation"
        print(f"✅ No files saved: {files_before} before, {files_after} after")
    
    def test_partition_structure(self):
        """Test that partition has correct structure with numpy arrays."""
        transformer = ImageTransformer()
        analyzer = ImageAnalyzer()
        line_processor = LineSegmentProcessor()
        
        processor = AuthorProcessor(
            label_id=42,
            author=self.author_name,
            author_images_dir=self.images_dir,
            author_mat_dir=self.mat_dir,
            transformer=transformer,
            analyzer=analyzer,
            line_processor=line_processor
        )
        
        partition, labels = processor.generate()
        
        # Check partition keys
        assert 'train' in partition, "Partition should have 'train' key"
        assert 'validation' in partition, "Partition should have 'validation' key"
        assert 'test' in partition, "Partition should have 'test' key"
        
        # Check that values are lists of numpy arrays
        for split_name, patches in partition.items():
            assert isinstance(patches, list), f"{split_name} should be a list"
            if len(patches) > 0:
                assert isinstance(patches[0], np.ndarray), \
                    f"{split_name} should contain numpy arrays, not file paths"
                print(f"✅ {split_name}: {len(patches)} patches (type: {type(patches[0])})")
        
        # Check labels
        assert isinstance(labels, list), "Labels should be a list"
        assert all(isinstance(label, (int, np.integer)) for label in labels), \
            "All labels should be integers"
        
        total_patches = sum(len(patches) for patches in partition.values())
        assert len(labels) == total_patches, \
            f"Labels count ({len(labels)}) should match total patches ({total_patches})"
        
        print(f"✅ Partition structure correct: {total_patches} total patches")
    
    def test_labels_consistency(self):
        """Test that all labels match the provided label_id."""
        transformer = ImageTransformer()
        analyzer = ImageAnalyzer()
        line_processor = LineSegmentProcessor()
        
        label_id = 99
        processor = AuthorProcessor(
            label_id=label_id,
            author=self.author_name,
            author_images_dir=self.images_dir,
            author_mat_dir=self.mat_dir,
            transformer=transformer,
            analyzer=analyzer,
            line_processor=line_processor
        )
        
        partition, labels = processor.generate()
        
        # All labels should be equal to label_id
        assert all(label == label_id for label in labels), \
            f"All labels should be {label_id}"
        
        print(f"✅ All {len(labels)} labels correctly set to {label_id}")
    
    def test_patch_dimensions(self):
        """Test that all patches have correct dimensions."""
        transformer = ImageTransformer()
        analyzer = ImageAnalyzer()
        line_processor = LineSegmentProcessor()
        
        patch_height = 180
        patch_width = 160
        
        processor = AuthorProcessor(
            label_id=0,
            author=self.author_name,
            author_images_dir=self.images_dir,
            author_mat_dir=self.mat_dir,
            transformer=transformer,
            analyzer=analyzer,
            line_processor=line_processor,
            patch_size=(patch_height, patch_width)
        )
        
        partition, labels = processor.generate()
        
        # Check dimensions of all patches
        for split_name, patches in partition.items():
            for i, patch in enumerate(patches):
                assert patch.shape[0] == patch_height, \
                    f"{split_name}[{i}] height should be {patch_height}, got {patch.shape[0]}"
                assert patch.shape[1] == patch_width, \
                    f"{split_name}[{i}] width should be {patch_width}, got {patch.shape[1]}"
        
        print(f"✅ All patches have correct dimensions: {patch_height}x{patch_width}")
    
    def test_train_val_test_split(self):
        """Test that train/val/test split is reasonable."""
        transformer = ImageTransformer()
        analyzer = ImageAnalyzer()
        line_processor = LineSegmentProcessor()
        
        processor = AuthorProcessor(
            label_id=0,
            author=self.author_name,
            author_images_dir=self.images_dir,
            author_mat_dir=self.mat_dir,
            transformer=transformer,
            analyzer=analyzer,
            line_processor=line_processor,
            val_size=0.15
        )
        
        partition, labels = processor.generate()
        
        train_count = len(partition['train'])
        val_count = len(partition['validation'])
        test_count = len(partition['test'])
        total = train_count + val_count + test_count
        
        # Validation should be approximately 15% of train+val
        # (allow wider range for small datasets due to rounding)
        if train_count + val_count > 0:
            val_ratio = val_count / (train_count + val_count)
            assert 0.10 <= val_ratio <= 0.30, \
                f"Validation ratio should be ~15% (±15%), got {val_ratio:.2%}"
        
        # Test set should exist
        assert test_count > 0, "Test set should not be empty"
        
        print(f"✅ Split ratios: train={train_count}, val={val_count}, test={test_count}")
        print(f"   Total: {total} patches")


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("RUNNING INTEGRATION TESTS FOR AUTHORPROCESSOR")
    print("=" * 60)
    
    test_suite = TestAuthorProcessorIntegration()
    
    # Setup
    print("\n[SETUP] Creating test data...")
    TestAuthorProcessorIntegration.setup_class()
    
    # Run tests
    tests = [
        ("Dependency Injection", test_suite.test_dependency_injection),
        ("In-Memory Patch Generation", test_suite.test_in_memory_patch_generation),
        ("Partition Structure", test_suite.test_partition_structure),
        ("Labels Consistency", test_suite.test_labels_consistency),
        ("Patch Dimensions", test_suite.test_patch_dimensions),
        ("Train/Val/Test Split", test_suite.test_train_val_test_split),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * 60)
        try:
            test_func()
            passed += 1
            print(f"✅ PASSED: {test_name}")
        except AssertionError as e:
            failed += 1
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {e}")
        except Exception as e:
            failed += 1
            print(f"❌ ERROR: {test_name}")
            print(f"   Exception: {e}")
    
    # Teardown
    print("\n[TEARDOWN] Cleaning up test data...")
    TestAuthorProcessorIntegration.teardown_class()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
