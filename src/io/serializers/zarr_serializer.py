from pathlib import Path
from typing import Any, Dict, List
import zarr
import numpy as np
import numcodecs

from .base_serializer import BaseSerializer
from ..exceptions import SerializationError, DeserializationError
from ..logging import LoggerFactory

# Compatibilidad con Zarr 2.x y 3.x
try:
    # Zarr 2.x
    DirectoryStore = zarr.DirectoryStore
except AttributeError:
    # Zarr 3.x
    try:
        from zarr.storage import DirectoryStore
    except ImportError:
        # Fallback para otras versiones
        DirectoryStore = None


class ZarrSerializer(BaseSerializer):
    """
    Zarr serializer con logging integrado para datasets grandes (10GB+).
    
    Features:
    - Logging detallado del progreso de guardado/carga
    - Estadísticas de compresión en tiempo real
    - Tracking de memoria y tamaño en disco
    - Manejo optimizado para datasets grandes (42GB → 4-8GB)
    
    Compresión esperada: 80-90% reducción con blosc+zstd
    """
    
    def __init__(
        self, 
        file_system, 
        compressor='blosc', 
        compression_level=5,
        log_dir=None
    ):
        """
        Args:
            file_system: FileSystem instance
            compressor: 'blosc' (recomendado), 'zstd', 'gzip', None
            compression_level: 1-9 (5 es buen balance velocidad/tamaño)
            log_dir: Directorio para logs (None = solo console)
        """
        super().__init__(file_system)
        
        # Setup logger
        self.logger = LoggerFactory.get_logger(
            name=self.__class__.__name__,
            log_dir=log_dir,
            console=True,
            file_prefix="zarr_serializer"
        )
        
        # Setup compressor
        if compressor == 'blosc':
            self.compressor = numcodecs.Blosc(
                cname='zstd',
                clevel=compression_level,
                shuffle=numcodecs.Blosc.BITSHUFFLE  # Óptimo para imágenes
            )
        elif compressor == 'zstd':
            self.compressor = numcodecs.Zstd(level=compression_level)
        elif compressor == 'gzip':
            self.compressor = numcodecs.GZip(level=compression_level)
        elif compressor is None:
            self.compressor = None
        else:
            raise ValueError(f"Unknown compressor: {compressor}")
        
        self.logger.info(f"Initialized with {compressor} compression (level {compression_level})")
    
    def extensions(self) -> list[str]:
        return [".zarr"]
    
    def save(self, directory: str | Path, data: Any, filename: str) -> None:
        """
        Guarda dataset con logging detallado del progreso.
        
        Args:
            directory: Directorio destino
            data: Dataset dict
            filename: Nombre del archivo (e.g., "dataset.zarr")
        """
        dir_path = self.fs.ensure_directory(directory)
        file_path = dir_path / filename
        
        self.logger.info("="*70)
        self.logger.info(f"Starting Zarr serialization: {filename}")
        self.logger.info("="*70)
        
        try:
            # Limpiar zarr existente
            if file_path.exists():
                import shutil
                self.logger.warning(f"Removing existing zarr store: {file_path}")
                shutil.rmtree(file_path)
            
            # Crear zarr store (compatible con 2.x y 3.x)
            if DirectoryStore is None:
                raise SerializationError(
                    "Zarr DirectoryStore not available. "
                    "Please install zarr>=2.16.0: pip install 'zarr>=2.16.0,<3.0.0'"
                )
            
            store = DirectoryStore(str(file_path))
            root = zarr.group(store=store, overwrite=True)
            
            # Guardar dataset
            if isinstance(data, dict):
                self._save_dataset_dict(root, data, file_path)
            else:
                raise SerializationError(f"Expected dict, got {type(data)}")
            
            # Log final stats
            self._log_final_stats(file_path)
            self.logger.info("="*70)
            self.logger.info("✅ Zarr serialization completed successfully")
            self.logger.info("="*70)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save Zarr: {e}", exc_info=True)
            raise SerializationError(f"Failed to save Zarr to {file_path}: {e}") from e
    
    def _save_dataset_dict(self, root: zarr.Group, data: Dict[str, Any], file_path: Path) -> None:
        """Guarda estructura de dataset con logging por componente."""
        
        for key, value in data.items():
            self.logger.info(f"📝 Processing key: '{key}'")
            
            if key == 'partition':
                # Partition es el componente más grande (99% del dataset)
                self._save_partition(root, value)
                
            elif key == 'labels':
                # Labels array
                self._save_labels(root, value)
                
            elif isinstance(value, np.ndarray):
                # Otros arrays
                size_mb = value.nbytes / (1024 * 1024)
                self.logger.info(f"   └─ Saving array: shape={value.shape}, size={size_mb:.1f}MB")
                root.create_dataset(key, data=value, compressor=self.compressor)
                
            elif isinstance(value, dict):
                # Metadata, author_indices, etc.
                if self._is_json_serializable(value):
                    self.logger.info(f"   └─ Saving as JSON metadata: {len(value)} items")
                    root.attrs[key] = value
                else:
                    self.logger.info(f"   └─ Saving as nested group")
                    subgroup = root.create_group(key)
                    self._save_nested_dict(subgroup, value)
                    
            else:
                # Primitivos
                self.logger.info(f"   └─ Saving primitive: {type(value).__name__}")
                root.attrs[key] = value
    
    def _save_labels(self, root: zarr.Group, labels: Any) -> None:
        """Guarda labels con logging."""
        if isinstance(labels, list):
            labels = np.array(labels, dtype=np.int32)
        
        size_mb = labels.nbytes / (1024 * 1024)
        self.logger.info(f"   └─ Labels: {len(labels)} items, {size_mb:.1f}MB")
        
        root.create_dataset(
            'labels',
            data=labels,
            compressor=self.compressor,
            chunks=10000
        )
    
    def _save_partition(self, root: zarr.Group, partition: Dict[str, List[np.ndarray]]) -> None:
        """
        Guarda partition con logging detallado del progreso.
        
        Esta es la parte crítica para datasets grandes (42GB).
        """
        partition_group = root.create_group('partition')
        
        self.logger.info("   ╔════════════════════════════════════════════════════════")
        self.logger.info("   ║ PARTITION (Main Dataset Component)")
        self.logger.info("   ╚════════════════════════════════════════════════════════")
        
        total_patches = sum(len(patches) for patches in partition.values())
        self.logger.info(f"   Total patches: {total_patches:,}")
        
        for split, patches in partition.items():
            if not patches:
                partition_group.attrs[f'{split}_count'] = 0
                self.logger.info(f"   ├─ {split}: EMPTY")
                continue
            
            num_patches = len(patches)
            
            # Calcular tamaño estimado
            sample_size = patches[0].nbytes if patches else 0
            estimated_mb = (sample_size * num_patches) / (1024 * 1024)
            
            self.logger.info(f"   ├─ {split}: {num_patches:,} patches (~{estimated_mb:.0f}MB)")
            
            # Verificar shapes
            shapes = [p.shape for p in patches]
            unique_shapes = set(shapes)
            
            if len(unique_shapes) == 1:
                # CASO IDEAL: todos mismo shape
                patch_shape = shapes[0]
                self.logger.info(f"   │  ├─ Shape: {patch_shape} (homogeneous ✓)")
                
                # Stack todos los patches
                self.logger.info(f"   │  ├─ Stacking {num_patches:,} patches...")
                stacked = np.stack(patches)
                
                # Calcular chunking óptimo
                chunks = (1, *stacked.shape[1:])  # 1 patch por chunk
                
                self.logger.info(f"   │  ├─ Compressing with {self.compressor.codec_id}...")
                partition_group.create_dataset(
                    split,
                    data=stacked,
                    compressor=self.compressor,
                    chunks=chunks,
                    dtype=stacked.dtype
                )
                partition_group.attrs[f'{split}_is_stacked'] = True
                
                # Calcular tamaño sin comprimir
                uncompressed_mb = stacked.nbytes / (1024 * 1024)
                self.logger.info(f"   │  └─ ✅ Done (uncompressed: {uncompressed_mb:.0f}MB)")
                
            else:
                # CASO RARO: diferentes shapes (después de tu fix no debería pasar)
                self.logger.warning(f"   │  ⚠️  Heterogeneous shapes detected: {unique_shapes}")
                self.logger.warning(f"   │     This will result in slower loading!")
                
                split_group = partition_group.create_group(split)
                split_group.attrs['is_ragged'] = True
                split_group.attrs['count'] = num_patches
                
                # Guardar individualmente con progreso
                milestone = max(1, num_patches // 20)  # Log cada 5%
                for idx, patch in enumerate(patches):
                    split_group.create_dataset(
                        f'{idx:06d}',
                        data=patch,
                        compressor=self.compressor,
                        chunks=True
                    )
                    
                    if (idx + 1) % milestone == 0:
                        pct = (idx + 1) / num_patches * 100
                        self.logger.info(f"   │  ├─ Progress: {pct:.0f}% ({idx+1:,}/{num_patches:,})")
                
                self.logger.info(f"   │  └─ ✅ Done")
        
        self.logger.info("   └─ Partition saved successfully")
    
    def _save_nested_dict(self, group: zarr.Group, data: dict, indent: str = "   ") -> None:
        """Guarda dicts anidados con logging."""
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(str(key))
                self._save_nested_dict(subgroup, value, indent + "  ")
            elif isinstance(value, np.ndarray):
                group.create_dataset(str(key), data=value, compressor=self.compressor)
            else:
                group.attrs[str(key)] = value
    
    def _is_json_serializable(self, obj: Any) -> bool:
        """Check if object can be stored as JSON."""
        import json
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False
    
    def _log_final_stats(self, file_path: Path) -> None:
        """Log estadísticas finales de compresión."""
        total_size = 0
        file_count = 0
        
        for path in file_path.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
                file_count += 1
        
        size_mb = total_size / (1024 * 1024)
        size_gb = size_mb / 1024
        
        self.logger.info("")
        self.logger.info("📊 FINAL STATISTICS:")
        self.logger.info(f"   ├─ Total size: {size_mb:.0f}MB ({size_gb:.2f}GB)")
        self.logger.info(f"   ├─ Files created: {file_count:,}")
        self.logger.info(f"   └─ Location: {file_path}")
    
    def load(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Carga dataset con logging del progreso.
        
        Returns:
            Dataset dict con lazy loading para partition
        """
        p = self.fs.ensure_directory(file_path)
        
        self.logger.info("="*70)
        self.logger.info(f"Loading Zarr dataset: {p.name}")
        self.logger.info("="*70)
        
        try:
            if DirectoryStore is None:
                raise DeserializationError(
                    "Zarr DirectoryStore not available. "
                    "Please install zarr>=2.16.0: pip install 'zarr>=2.16.0,<3.0.0'"
                )
            
            store = DirectoryStore(str(p))
            root = zarr.open_group(store=store, mode='r')
            
            dataset = {}
            
            # Cargar partition (lazy!)
            if 'partition' in root.group_keys():
                self.logger.info("📂 Loading partition (lazy mode)...")
                dataset['partition'] = self._load_partition_lazy(root['partition'])
                self.logger.info("   └─ Partition loaded (data not yet in memory)")
            
            # Cargar arrays
            for key in root.array_keys():
                arr = root[key]
                size_mb = arr.nbytes / (1024 * 1024)
                self.logger.info(f"📂 Loading '{key}': {arr.shape}, {size_mb:.1f}MB")
                dataset[key] = arr[:]
            
            # Cargar metadata
            if root.attrs:
                self.logger.info(f"📂 Loading metadata: {len(root.attrs)} items")
                for key, value in root.attrs.items():
                    dataset[key] = value
            
            # Cargar grupos anidados
            for key in root.group_keys():
                if key != 'partition':
                    self.logger.info(f"📂 Loading nested group: '{key}'")
                    dataset[key] = self._load_nested_dict(root[key])
            
            self.logger.info("="*70)
            self.logger.info("✅ Dataset loaded successfully")
            self.logger.info("="*70)
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load Zarr: {e}", exc_info=True)
            raise DeserializationError(f"Failed to load Zarr from {p}: {e}") from e
    
    def _load_partition_lazy(self, partition_group: zarr.Group) -> Dict[str, Any]:
        """
        Carga partition con LAZY LOADING y logging.
        
        Los patches no se cargan en memoria hasta que se hace slicing.
        """
        partition = {}
        
        for split in ['train', 'validation', 'test']:
            # Check si está vacío
            if f'{split}_count' in partition_group.attrs:
                if partition_group.attrs[f'{split}_count'] == 0:
                    self.logger.info(f"   ├─ {split}: EMPTY")
                    partition[split] = []
                    continue
            
            # Check si es stacked (caso normal)
            if split in partition_group.array_keys():
                zarr_array = partition_group[split]
                num_patches = zarr_array.shape[0]
                size_mb = zarr_array.nbytes / (1024 * 1024)
                
                self.logger.info(
                    f"   ├─ {split}: {num_patches:,} patches "
                    f"(~{size_mb:.0f}MB, lazy-loaded)"
                )
                
                # Wrap en lazy loader
                partition[split] = ZarrArrayWrapper(zarr_array)
                
            elif split in partition_group.group_keys():
                # Ragged array (cargar todo - no hay otra opción)
                split_group = partition_group[split]
                count = split_group.attrs['count']
                
                self.logger.warning(
                    f"   ├─ {split}: {count:,} patches (ragged, loading all)"
                )
                
                partition[split] = [
                    split_group[f'{i:06d}'][:] 
                    for i in range(count)
                ]
            else:
                self.logger.info(f"   ├─ {split}: NOT FOUND")
                partition[split] = []
        
        return partition
    
    def _load_nested_dict(self, group: zarr.Group) -> dict:
        """Carga dicts anidados."""
        result = {}
        
        for key in group.array_keys():
            k = int(key) if key.isdigit() else key
            result[k] = group[key][:]
        
        for key in group.group_keys():
            k = int(key) if key.isdigit() else key
            result[k] = self._load_nested_dict(group[key])
        
        for key, value in group.attrs.items():
            if key not in ['is_ragged', 'count']:
                k = int(key) if key.isdigit() else key
                result[k] = value
        
        return result


class ZarrArrayWrapper:
    """
    Wrapper para lazy loading de zarr arrays.
    
    Simula una lista de numpy arrays pero solo carga en memoria
    cuando se hace slicing.
    
    Ejemplo:
        wrapper = ZarrArrayWrapper(zarr_array)  # No carga nada en memoria
        patches = wrapper[100:200]  # Solo carga 100 patches
        all_patches = wrapper[:]  # Carga todo (equivalente a lista)
    """
    
    def __init__(self, zarr_array: zarr.Array):
        self.zarr_array = zarr_array
        self._length = zarr_array.shape[0]
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, key):
        """
        Lazy loading con slicing.
        
        wrapper[0] → carga 1 patch
        wrapper[10:20] → carga 10 patches
        wrapper[:] → carga todos (convierte a lista)
        """
        if isinstance(key, slice):
            # Slicing: cargar y retornar lista
            data = self.zarr_array[key]
            return [data[i] for i in range(data.shape[0])]
        elif isinstance(key, int):
            # Single index: cargar 1 array
            return self.zarr_array[key]
        else:
            # Array de índices
            return [self.zarr_array[i] for i in key]
    
    def extend(self, items):
        """Read-only wrapper."""
        raise NotImplementedError("ZarrArrayWrapper is read-only")
