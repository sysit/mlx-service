"""
Model Manager for MLX Service
Handles model loading, caching, and GPU resource management

P0 + P1 + P2 Implementation:
- P0: Queue + GPU Lock + Synchronization Barrier
- P1: Delayed Unload Strategy
- P2: Event-Driven Architecture
"""
import time
import os
import threading
import queue
from typing import Optional, Dict, List, Tuple, Any, Callable
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass, field


# =============================================================================
# P2: Event-Driven Architecture - Model States and Events
# =============================================================================

class ModelState(Enum):
    """P2: Model lifecycle states for event-driven architecture"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    UNLOADING = "unloading"
    ERROR = "error"


class ModelEvent(Enum):
    """P2: Events that can trigger state transitions"""
    LOAD_REQUEST = "load_request"
    LOAD_COMPLETE = "load_complete"
    LOAD_FAILED = "load_failed"
    UNLOAD_REQUEST = "unload_request"
    UNLOAD_COMPLETE = "unload_complete"


# =============================================================================
# P0: Load Request for Queue-based Processing
# =============================================================================

@dataclass
class LoadRequest:
    """P0: Request object for model loading queue"""
    name: str
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = field(default=None)
    error: Optional[Exception] = field(default=None)
    
    def wait(self, timeout: Optional[float] = None) -> Any:
        """Wait for the request to complete and return result"""
        self.event.wait(timeout)
        if self.error:
            raise self.error
        return self.result
    
    def complete(self, result: Any = None):
        """Mark request as complete with result"""
        self.result = result
        self.event.set()
    
    def fail(self, error: Exception):
        """Mark request as failed with error"""
        self.error = error
        self.event.set()


@dataclass
class ModelInfo:
    """P2: Model information with state tracking"""
    name: str
    path: str
    state: ModelState = ModelState.UNLOADED
    instance: Any = None
    last_used: float = field(default_factory=time.time)
    load_time: Optional[float] = None
    error_count: int = 0


# =============================================================================
# Model Registry (unchanged functionality)
# =============================================================================

class ModelRegistry:
    """Registry for managing model paths and aliases"""
    
    def __init__(self, models_dir: Optional[Path] = None):
        self._models: Dict[str, str] = {}  # alias -> path
        self._full_names: Dict[str, str] = {}  # alias -> full_name
        self._aliases: Dict[str, str] = {}  # full_name -> alias
        self._scan_models(models_dir)
    
    def _scan_models(self, models_dir: Optional[Path] = None):
        """Scan for available models and generate aliases"""
        model_dirs = []
        
        # 使用指定的模型目录
        if models_dir:
            model_dirs.append(Path(models_dir))
        
        # 默认目录
        model_dirs.extend([
            Path.home() / "models" / "mlx-community",
            Path.home() / ".mlx_models",
            Path("/opt/mlx/models"),
            Path("./models"),
        ])
        
        for model_dir in model_dirs:
            if model_dir.exists():
                for model_path in model_dir.iterdir():
                    if model_path.is_dir():
                        # 跳过隐藏目录和缓存目录
                        if model_path.name.startswith('.') or model_path.name in ['cache', '__pycache__']:
                            continue
                        full_name = model_path.name
                        alias = self._generate_alias(full_name)
                        self._models[alias] = str(model_path)
                        self._full_names[alias] = full_name
                        self._aliases[full_name] = alias
    
    def _generate_alias(self, full_name: str) -> str:
        """Generate a short alias from full model name"""
        parts = full_name.replace("_", "-").split("-")
        
        for i, part in enumerate(parts):
            if len(part) > 1 and part[-1].lower() == "b":
                size_part = part[:-1]
                if size_part.isdigit():
                    return "-".join(parts[:i+1]).lower()
        
        if len(parts) >= 2:
            return "-".join(parts[:2]).lower()
        return full_name.lower()
    
    def list_models(self) -> List[Dict]:
        """Return list of model info dicts"""
        result = []
        for alias, full_path in self._models.items():
            result.append({
                "name": alias,
                "full_name": self._full_names.get(alias, alias),
                "path": full_path
            })
        return result
    
    def resolve(self, model_id: str) -> Optional[str]:
        """Resolve a model ID to full path"""
        if model_id in self._models:
            return self._models[model_id]
        if model_id in self._aliases:
            return self._models.get(self._aliases[model_id])
        if os.path.exists(model_id):
            return model_id
        return None
    
    def get_alias(self, full_name: str) -> Optional[str]:
        """Get alias for a full model name"""
        return self._aliases.get(full_name)


# =============================================================================
# P0 + P1 + P2: ModelManager with All Improvements
# =============================================================================

class ModelManager:
    """P0 + P1 + P2: Manages model loading with GPU resource safety"""
    
    def __init__(self, models_dir: Optional[Path] = None):
        self.registry = ModelRegistry(models_dir)
        
        # P0: Queue-based loading
        self._load_queue = queue.Queue()
        
        # P0: GPU-specific lock for GPU operation serialization
        self._gpu_lock = threading.Lock()
        
        # P0: General lock for thread-safe access to internal state
        self._lock = threading.RLock()
        
        # P2: Model state tracking
        self._models = {}
        self._current_model_id = None
        
        # P2: Event callbacks for state changes
        self._event_callbacks = []
        
        # P1: Delayed unload queue
        self._pending_unload = None
        self._unload_delay = 1.0
        
        # P0: Start background queue processor
        self._queue_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._queue_thread.start()
        
        # P1: Start delayed unload processor
        self._unload_thread = threading.Thread(target=self._process_delayed_unloads, daemon=True)
        self._unload_thread.start()
    
    # =====================================================================
    # P2: Event Subscription and State Management
    # =====================================================================
    
    def subscribe(self, callback):
        """P2: Subscribe to model state change events"""
        with self._lock:
            self._event_callbacks.append(callback)
    
    def _emit_event(self, event, model_id, old_state, new_state):
        """P2: Emit event to all subscribers"""
        for callback in self._event_callbacks:
            try:
                callback(event, model_id, old_state, new_state)
            except Exception:
                pass
    
    def _set_model_state(self, model_id, new_state):
        """P2: Set model state and emit event"""
        with self._lock:
            if model_id in self._models:
                old_state = self._models[model_id].state
                self._models[model_id].state = new_state
                
                if new_state == ModelState.LOADING:
                    event = ModelEvent.LOAD_REQUEST
                elif new_state == ModelState.READY:
                    event = ModelEvent.LOAD_COMPLETE
                elif new_state == ModelState.ERROR:
                    event = ModelEvent.LOAD_FAILED
                elif new_state == ModelState.UNLOADING:
                    event = ModelEvent.UNLOAD_REQUEST
                elif new_state == ModelState.UNLOADED:
                    event = ModelEvent.UNLOAD_COMPLETE
                else:
                    return
                
                self._emit_event(event, model_id, old_state, new_state)
    
    def get_model_state(self, model_id):
        """P2: Get current state of a model"""
        with self._lock:
            if model_id in self._models:
                return self._models[model_id].state
            return ModelState.UNLOADED
    
    # =====================================================================
    # P0: Queue Processing (Background Thread)
    # =====================================================================
    
    def _process_queue(self):
        """P0: Background thread that processes load requests serially"""
        while True:
            try:
                request = self._load_queue.get()
                
                with self._gpu_lock:
                    try:
                        result = self._load_model_internal(request.name)
                        request.complete(result)
                    except Exception as e:
                        request.fail(e)
                
                self._load_queue.task_done()
            except Exception:
                pass
    
    # =====================================================================
    # P1: Delayed Unload Processing (Background Thread)
    # =====================================================================
    
    def _process_delayed_unloads(self):
        """P1: Background thread that processes delayed unloads"""
        while True:
            time.sleep(0.1)
            
            with self._lock:
                if self._pending_unload is not None:
                    model_info = self._pending_unload
                    elapsed = time.time() - model_info.last_used
                    if elapsed >= self._unload_delay:
                        self._pending_unload = None
                        self._unload_model_async(model_info)
    
    def _unload_model_async(self, model_info):
        """P1: Actually unload a model (called from background thread)"""
        with self._gpu_lock:
            self._set_model_state(model_info.name, ModelState.UNLOADING)
            
            try:
                try:
                    import mlx.core as mx
                    mx.clear_cache()
                    mx.eval([])
                except ImportError:
                    pass
                
                model_info.instance = None
                self._set_model_state(model_info.name, ModelState.UNLOADED)
            except Exception:
                model_info.error_count += 1
    
    # =====================================================================
    # P0 + P1: Main Public API
    # =====================================================================
    
    def get(self, model_id):
        """Get a model, implementing P0 + P1 + P2 improvements"""
        model_path = self.registry.resolve(model_id)
        if model_path is None:
            return None
        
        with self._lock:
            if model_id in self._models:
                model_info = self._models[model_id]
                if model_info.state == ModelState.READY:
                    model_info.last_used = time.time()
                    return model_info.instance
            
            old_model_id = self._current_model_id
            request = LoadRequest(name=model_id)
            self._load_queue.put(request)
        
        try:
            result = request.wait(timeout=300)
        except Exception:
            return None
        
        if old_model_id and old_model_id != model_id:
            with self._lock:
                if old_model_id in self._models:
                    old_model = self._models[old_model_id]
                    if old_model.state == ModelState.READY:
                        old_model.last_used = time.time()
                        self._pending_unload = old_model
        
        return result
    
    def _load_model_internal(self, model_id):
        """P0: Internal method to load model (called within GPU lock)"""
        model_path = self.registry.resolve(model_id)
        if model_path is None:
            return None
        
        with self._lock:
            if model_id not in self._models:
                self._models[model_id] = ModelInfo(name=model_id, path=model_path)
            
            model_info = self._models[model_id]
            
            if model_info.state == ModelState.LOADING:
                return None
            
            if model_info.state == ModelState.READY and model_info.instance:
                model_info.last_used = time.time()
                self._current_model_id = model_id
                return model_info.instance
        
        self._set_model_state(model_id, ModelState.LOADING)
        
        try:
            try:
                import mlx.core as mx
                mx.clear_cache()
            except ImportError:
                pass
            
            time.sleep(2.0)
            
            instance = self._load_model_from_path(model_path)
            
            if instance is None:
                self._set_model_state(model_id, ModelState.ERROR)
                return None
            
            with self._lock:
                model_info.instance = instance
                model_info.load_time = time.time()
                self._current_model_id = model_id
            
            self._set_model_state(model_id, ModelState.READY)
            
            try:
                import mlx.core as mx
                mx.eval([])
            except ImportError:
                pass
            
            return instance
            
        except Exception:
            self._set_model_state(model_id, ModelState.ERROR)
            with self._lock:
                model_info.error_count += 1
            return None
    
    def _load_model_from_path(self, path):
        """Load a model from the given path"""
        try:
            from mlx_lm import load
            model, processor = load(path)
            return (model, processor)
        except Exception as e:
            return None
    
    # =====================================================================
    # Backward Compatibility
    # =====================================================================
    
    def list_available_models(self):
        """List all available models with their aliases"""
        models = []
        for full_name, alias in self.registry._aliases.items():
            models.append({
                "id": alias,
                "full_name": full_name,
                "path": self.registry._models.get(alias, "")
            })
        return models
    
    def shutdown(self):
        """Shutdown the model manager and release all resources"""
        with self._lock:
            for model_id, model_info in list(self._models.items()):
                if model_info.instance is not None:
                    self._set_model_state(model_id, ModelState.UNLOADING)
                    model_info.instance = None
                    self._set_model_state(model_id, ModelState.UNLOADED)
        
        try:
            import mlx.core as mx
            mx.clear_cache()
        except ImportError:
            pass
    
    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded"""
        with self._lock:
            if model_id in self._models:
                return self._models[model_id].state == ModelState.READY
            return False
    
    def list_loaded(self) -> Dict:
        """List all currently loaded models"""
        result = {"models": [], "total_memory_gb": 0}
        with self._lock:
            for model_id, model_info in self._models.items():
                if model_info.state == ModelState.READY:
                    result["models"].append({
                        "name": model_id,
                        "load_time": model_info.load_time
                    })
        return result


# =============================================================================
# Global Model Manager Instance (Backward Compatibility)
# =============================================================================

model_manager = ModelManager()

# For backward compatibility
_gpu_lock = model_manager._gpu_lock
_gpu_operation_lock = model_manager._lock
