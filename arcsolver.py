import numpy as np
from typing import List, Dict, Tuple, Callable, Any, Optional 
import itertools
from copy import deepcopy
import random
import json
import os
import glob
from pathlib import Path
import traceback 
import datetime

class ARCGrid:
    """Simple wrapper for ARC grids with utility methods"""
    def __init__(self, grid: List[List[int]]): # Can also accept np.ndarray
        if isinstance(grid, np.ndarray):
            self.grid = grid.copy()
        else:
            self.grid = np.array(grid)
        self.height, self.width = self.grid.shape
    
    def get_colors(self) -> set:
        return set(self.grid.flatten())
    
    def count_color(self, color: int) -> int:
        return np.sum(self.grid == color)
    
    def find_objects(self, color: int) -> List[Tuple[int, int]]:
        """Find all positions of a specific color"""
        return [(int(r), int(c)) for r, c in zip(*np.where(self.grid == color))]
    
    def get_connected_components(self, color: int) -> List[List[Tuple[int, int]]]:
        """Find connected components of a specific color"""
        positions = self.find_objects(color)
        if not positions:
            return []
        
        components = []
        visited = set()
        
        for pos in positions:
            if pos in visited:
                continue
            
            component = []
            stack = [pos]
            
            while stack:
                r, c = stack.pop()
                if (r, c) in visited:
                    continue
                
                visited.add((r, c))
                component.append((r, c))
                
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.height and 0 <= nc < self.width and 
                        (nr, nc) not in visited and (nr, nc) in positions):
                        stack.append((nr, nc))
            
            components.append(component)
        
        return components

    def find_all_objects(self, background_color: int = 0) -> List[List[Tuple[int, int]]]:
        """
        Find all connected components of non-background_color pixels.
        Each component is a list of (r, c) pixel coordinates.
        """
        content_mask = self.grid != background_color
        
        positions_to_check = []
        for r_idx in range(self.height):
            for c_idx in range(self.width):
                if content_mask[r_idx, c_idx]:
                    positions_to_check.append((r_idx, c_idx))
        
        if not positions_to_check:
            return []
        
        components = []
        visited = set()
        
        for r_init, c_init in positions_to_check:
            pos_init = (r_init, c_init)
            if pos_init in visited:
                continue
            
            component = []
            q = [pos_init] 
            visited.add(pos_init)
            
            head = 0
            while head < len(q):
                r, c = q[head]
                head += 1
                
                component.append((r, c))
                
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.height and 0 <= nc < self.width and 
                        content_mask[nr, nc] and (nr, nc) not in visited):
                        visited.add((nr, nc))
                        q.append((nr, nc))
            
            components.append(component)
        
        return components

    def get_bounding_box(self, pixels: List[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
        if not pixels:
            return None
        min_r = min(r for r, c in pixels)
        max_r = max(r for r, c in pixels)
        min_c = min(c for r, c in pixels)
        max_c = max(c for r, c in pixels)
        return min_r, min_c, max_r, max_c

    def get_object_pixels_by_rank(self, background_color: int = 0, rank: int = 0, largest: bool = True) -> Optional[List[Tuple[int, int]]]:
        objects = self.find_all_objects(background_color)
        if not objects:
            return None
        
        objects.sort(key=len, reverse=largest) 
        
        if 0 <= rank < len(objects):
            return objects[rank]
        return None

class PrimitiveOperation:
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func = func
        self.arity = func.__code__.co_argcount - 1 # -1 for 'self' in library methods
    
    def apply(self, *args): 
        return self.func(*args)

class PrimitiveLibrary:
    def __init__(self):
        self.primitives = {}
        self._build_library()
    
    def _build_library(self):
        self.add_primitive("fill_color", lambda grid, old_color, new_color: self._fill_color(grid, old_color, new_color))
        self.add_primitive("replace_color", lambda grid, old_color, new_color: self._replace_color(grid, old_color, new_color))
        self.add_primitive("isolate_color", lambda grid, color_to_keep, replacement_color: self._isolate_color(grid, color_to_keep, replacement_color))
        self.add_primitive("rotate_90", lambda grid: self._rotate_90(grid))
        self.add_primitive("rotate_180", lambda grid: self._rotate_180(grid))
        self.add_primitive("flip_horizontal", lambda grid: self._flip_horizontal(grid))
        self.add_primitive("flip_vertical", lambda grid: self._flip_vertical(grid))
        self.add_primitive("transpose", lambda grid: self._transpose(grid)) 
        self.add_primitive("roll_rows", lambda grid, shift: self._roll_rows(grid, shift)) 
        self.add_primitive("roll_cols", lambda grid, shift: self._roll_cols(grid, shift)) 
        self.add_primitive("extract_largest_object_by_color", lambda grid, color: self._extract_largest_object_by_color(grid, color))
        self.add_primitive("count_colors", lambda grid: self._count_colors(grid))
        self.add_primitive("get_most_common_color", lambda grid: self._get_most_common_color(grid))
        self.add_primitive("crop_to_content", lambda grid, bg_color=0: self._crop_to_content(grid, bg_color))
        self.add_primitive("extend_pattern", lambda grid, direction: self._extend_pattern(grid, direction)) 
        self.add_primitive("resize_to_match", lambda grid, target_shape: self._resize_to_match(grid, target_shape))
        self.add_primitive("pad_to_size", lambda grid, height, width: self._pad_to_size(grid, height, width))
        self.add_primitive("color_object_by_rank", lambda grid, rank, new_color, largest, obj_detect_bg_color: self._color_object_by_rank(grid, rank, new_color, largest, obj_detect_bg_color))
        self.add_primitive("delete_object_by_rank", lambda grid, rank, fill_color, largest, obj_detect_bg_color: self._delete_object_by_rank(grid, rank, fill_color, largest, obj_detect_bg_color))
        self.add_primitive("extract_mask_of_object_by_rank", lambda grid, rank, mask_fg_color, mask_bg_color, largest, obj_detect_bg_color: self._extract_mask_of_object_by_rank(grid, rank, mask_fg_color, mask_bg_color, largest, obj_detect_bg_color))
        self.add_primitive("crop_to_object_by_rank", lambda grid, rank, fallback_fill_color, largest, obj_detect_bg_color: self._crop_to_object_by_rank(grid, rank, fallback_fill_color, largest, obj_detect_bg_color))
        self.add_primitive("copy_object_by_rank_and_paste", lambda grid, rank, largest, obj_detect_bg_color, target_dest_r, target_dest_c, paste_only_object_pixels: self._copy_object_by_rank_and_paste(grid, rank, largest, obj_detect_bg_color, target_dest_r, target_dest_c, paste_only_object_pixels))
        self.add_primitive("count_all_objects", lambda grid, obj_detect_bg_color: self._count_all_objects(grid, obj_detect_bg_color))
        self.add_primitive("translate_content", lambda grid, dr, dc, bg: self._translate_content(grid, dr, dc, bg))

    def add_primitive(self, name: str, func: Callable):
        self.primitives[name] = PrimitiveOperation(name, func)
    def get_primitive(self, name: str) -> PrimitiveOperation: return self.primitives.get(name)
    def list_primitives(self) -> List[str]: return list(self.primitives.keys())
    def _fill_color(self, grid: np.ndarray, old_color: int, new_color: int) -> np.ndarray:
        result = grid.copy(); result[result == old_color] = new_color; return result
    def _replace_color(self, grid: np.ndarray, old_color: int, new_color: int) -> np.ndarray:
        return self._fill_color(grid, old_color, new_color)
    def _rotate_90(self, grid: np.ndarray) -> np.ndarray: return np.rot90(grid, k=-1)
    def _rotate_180(self, grid: np.ndarray) -> np.ndarray: return np.rot90(grid, k=2)
    def _flip_horizontal(self, grid: np.ndarray) -> np.ndarray: return np.fliplr(grid)
    def _flip_vertical(self, grid: np.ndarray) -> np.ndarray: return np.flipud(grid)
    def _transpose(self, grid: np.ndarray) -> np.ndarray: return np.transpose(grid).copy()
    def _roll_rows(self, grid: np.ndarray, shift: int) -> np.ndarray:
        if grid.ndim != 2 or grid.shape[0] == 0: return grid.copy() 
        return np.roll(grid, shift, axis=0).copy()
    def _roll_cols(self, grid: np.ndarray, shift: int) -> np.ndarray:
        if grid.ndim != 2 or grid.shape[1] == 0: return grid.copy() 
        return np.roll(grid, shift, axis=1).copy()
    def _extract_largest_object_by_color(self, grid: np.ndarray, color: int) -> np.ndarray:
        arc_grid_obj = ARCGrid(grid); components = arc_grid_obj.get_connected_components(color)
        if not components: return grid.copy()
        largest = max(components, key=len); bg_color_of_result = 0
        if grid.size > 0:
            unique_cols, counts = np.unique(grid, return_counts=True)
            if unique_cols.size > 0: bg_color_of_result = unique_cols[np.argmax(counts)]
            if 0 in unique_cols: bg_color_of_result = 0
        result = np.full(grid.shape, bg_color_of_result, dtype=grid.dtype)
        for r, c in largest: result[r, c] = color
        return result
    def _count_colors(self, grid: np.ndarray) -> int: return len(np.unique(grid))
    def _get_most_common_color(self, grid: np.ndarray) -> int:
        if grid.size == 0: return 0 
        values, counts = np.unique(grid, return_counts=True); return int(values[np.argmax(counts)]) 
    def _crop_to_content(self, grid: np.ndarray, background_color_to_ignore: int = 0) -> np.ndarray:
        if grid.size == 0: return grid.copy()
        content_indices = np.where(grid != background_color_to_ignore)
        if len(content_indices[0]) == 0: return np.array([[background_color_to_ignore]], dtype=grid.dtype) 
        min_r, max_r = content_indices[0].min(), content_indices[0].max()
        min_c, max_c = content_indices[1].min(), content_indices[1].max()
        return grid[min_r:max_r+1, min_c:max_c+1].copy() 
    def _extend_pattern(self, grid: np.ndarray, direction: str) -> np.ndarray: return grid.copy()
    def _resize_to_match(self, grid: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        current_shape = grid.shape; target_h, target_w = target_shape
        if current_shape == target_shape: return grid.copy()
        result = np.zeros(target_shape, dtype=grid.dtype)
        copy_h = min(current_shape[0], target_h); copy_w = min(current_shape[1], target_w)
        result[:copy_h, :copy_w] = grid[:copy_h, :copy_w]; return result
    def _pad_to_size(self, grid: np.ndarray, height: int, width: int) -> np.ndarray:
        return self._resize_to_match(grid, (height, width))
    def _isolate_color(self, grid: np.ndarray, color_to_keep: int, replacement_color: int) -> np.ndarray:
        result = grid.copy(); result[grid != color_to_keep] = replacement_color; return result
    def _get_arcgrid_and_object_pixels(self, grid_array: np.ndarray, rank: int, largest: bool, obj_detect_bg_color: int) -> Tuple[Optional[ARCGrid], Optional[List[Tuple[int, int]]]]:
        arc_grid_obj = ARCGrid(grid_array)
        pixels = arc_grid_obj.get_object_pixels_by_rank(background_color=obj_detect_bg_color, rank=rank, largest=largest)
        return arc_grid_obj, pixels
    def _color_object_by_rank(self, grid: np.ndarray, rank: int, new_color: int, largest: bool, obj_detect_bg_color: int) -> np.ndarray:
        _, pixels_to_color = self._get_arcgrid_and_object_pixels(grid, rank, largest, obj_detect_bg_color)
        result = grid.copy()
        if pixels_to_color:
            for r, c in pixels_to_color: result[r, c] = new_color
        return result
    def _delete_object_by_rank(self, grid: np.ndarray, rank: int, fill_color: int, largest: bool, obj_detect_bg_color: int) -> np.ndarray:
        return self._color_object_by_rank(grid, rank, fill_color, largest, obj_detect_bg_color)
    def _extract_mask_of_object_by_rank(self, grid: np.ndarray, rank: int, mask_fg_color: int, mask_bg_color: int, largest: bool, obj_detect_bg_color: int) -> np.ndarray:
        _, pixels_of_object = self._get_arcgrid_and_object_pixels(grid, rank, largest, obj_detect_bg_color)
        mask = np.full(grid.shape, mask_bg_color, dtype=grid.dtype)
        if pixels_of_object:
            for r, c in pixels_of_object: mask[r, c] = mask_fg_color
        return mask
    def _crop_to_object_by_rank(self, grid: np.ndarray, rank: int, fallback_fill_color: int, largest: bool, obj_detect_bg_color: int) -> np.ndarray:
        arc_grid_obj, pixels_of_object = self._get_arcgrid_and_object_pixels(grid, rank, largest, obj_detect_bg_color)
        if not pixels_of_object or not arc_grid_obj: return np.array([[fallback_fill_color]], dtype=grid.dtype)
        bbox = arc_grid_obj.get_bounding_box(pixels_of_object)
        if not bbox: return np.array([[fallback_fill_color]], dtype=grid.dtype)
        min_r, min_c, max_r, max_c = bbox; cropped_grid = grid[min_r:max_r+1, min_c:max_c+1].copy()
        if cropped_grid.size == 0: return np.array([[fallback_fill_color]], dtype=grid.dtype)
        return cropped_grid
    def _copy_object_by_rank_and_paste(self, grid: np.ndarray, rank: int, largest: bool, obj_detect_bg_color: int,
                                        target_dest_r: int, target_dest_c: int, paste_only_object_pixels: bool) -> np.ndarray:
        arc_grid_obj, pixels_of_object = self._get_arcgrid_and_object_pixels(grid, rank, largest, obj_detect_bg_color)
        if not pixels_of_object or not arc_grid_obj: return grid.copy()
        bbox = arc_grid_obj.get_bounding_box(pixels_of_object)
        if not bbox: return grid.copy()
        min_r_orig, min_c_orig, max_r_orig, max_c_orig = bbox
        object_stamp = grid[min_r_orig : max_r_orig + 1, min_c_orig : max_c_orig + 1]
        stamp_h, stamp_w = object_stamp.shape; result_grid = grid.copy()
        for r_s in range(stamp_h):
            for c_s in range(stamp_w):
                orig_pixel_r, orig_pixel_c = min_r_orig + r_s, min_c_orig + c_s
                value_to_paste = object_stamp[r_s, c_s]; dest_r, dest_c = target_dest_r + r_s, target_dest_c + c_s
                if 0 <= dest_r < result_grid.shape[0] and 0 <= dest_c < result_grid.shape[1]:
                    if paste_only_object_pixels:
                        if (orig_pixel_r, orig_pixel_c) in pixels_of_object: result_grid[dest_r, dest_c] = value_to_paste
                    else: result_grid[dest_r, dest_c] = value_to_paste
        return result_grid
    def _count_all_objects(self, grid: np.ndarray, obj_detect_bg_color: int) -> int:
        arc_grid_obj = ARCGrid(grid); objects = arc_grid_obj.find_all_objects(background_color=obj_detect_bg_color)
        return len(objects)
    def _translate_content(self, grid: np.ndarray, dr: int, dc: int, background_color_of_input: int) -> np.ndarray:
        if grid.ndim != 2 : return grid.copy() # Should not happen if called via Program.execute
        new_grid = np.full(grid.shape, background_color_of_input, dtype=grid.dtype)
        height, width = grid.shape
        for r in range(height):
            for c in range(width):
                if grid[r, c] != background_color_of_input:
                    val_to_move = grid[r,c]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        new_grid[nr, nc] = val_to_move
        return new_grid

class Program:
    def __init__(self, operations: List[Tuple[str, List[Any]]] = None):
        self.operations = operations or []
        self.fitness = 0.0 
        self.complexity = len(self.operations)
    
    def execute(self, input_grid: np.ndarray, library: PrimitiveLibrary) -> Optional[np.ndarray]:
        current_value: Any = input_grid.copy(); op_name_for_error = "None" 
        try:
            if not self.operations: return np.array(current_value) if isinstance(current_value, np.ndarray) else None
            for i, (op_name, args) in enumerate(self.operations):
                op_name_for_error = op_name; primitive = library.get_primitive(op_name)
                if not primitive: return None
                # Ensure args are of correct types if primitive expects specific types (e.g. int for colors/coords)
                # For now, assume Program._generate_args_for_op and mutate handle this.
                # A more robust system might have type hints in primitive defs and check here.
                if len(args) != primitive.arity: return None
                if not isinstance(current_value, np.ndarray): return None # Input to op must be grid
                current_value = primitive.apply(current_value, *args)
                if not isinstance(current_value, np.ndarray):
                    # If current_value is not a grid but it's not the last op, this is an error
                    if i < len(self.operations) - 1: return None
                if isinstance(current_value, np.ndarray) and current_value.size == 0: return None # Empty grid output is invalid unless it's the final desired output
            if not isinstance(current_value, np.ndarray): return None
            return current_value 
        except Exception as e:
            # print(f"Error during program execution of op '{op_name_for_error}' with args {args if 'args' in locals() else 'N/A'}: {e}")
            # traceback.print_exc() # For debugging
            return None

    @staticmethod
    def _generate_args_for_op(op_name: str, patterns: Dict[str, Any] = None, library: PrimitiveLibrary = None) -> List[Any]:
        patterns = patterns or {} 
        if op_name in ["fill_color", "replace_color"]:
            _old_c, _new_c = -1, -1; color_changes_patterns = patterns.get('color_changes', {}); output_monochrome_info = patterns.get('output_monochrome_check', {})
            overall_input_colors = list(set(color_changes_patterns.get('overall_input_colors', []))); overall_output_colors = list(set(color_changes_patterns.get('overall_output_colors', [])))
            selected_path = "none"; frequent_mappings = []; mappings_per_example = color_changes_patterns.get('mappings_per_example', [])
            if mappings_per_example: 
                potential_mappings = {} 
                for example_mapping in mappings_per_example:
                    removed = example_mapping.get('removed_colors', []); added = example_mapping.get('new_colors', [])
                    if len(removed) == 1 and len(added) == 1: pair = (removed[0], added[0]); potential_mappings[pair] = potential_mappings.get(pair, 0) + 1
                if potential_mappings:
                    num_examples = len(mappings_per_example)
                    for pair, count in potential_mappings.items():
                        threshold = max(1, num_examples // 2) if num_examples > 0 else 1
                        if count >= threshold : frequent_mappings.append(pair)
            if frequent_mappings and random.random() < 0.7: 
                chosen_pair = random.choice(frequent_mappings); _old_c, _new_c = chosen_pair[0], chosen_pair[1]; selected_path = "frequent_mapping"
            elif output_monochrome_info.get('is_monochrome') and random.random() < 0.8: 
                selected_path = "monochrome"; mono_fg = output_monochrome_info.get('foreground_color'); mono_bg = output_monochrome_info.get('background_color') 
                if mono_fg is None: selected_path = "general_fallback_mono_incomplete" 
                else:
                    _new_c = mono_fg; old_color_choices_mono = []
                    if mono_bg is not None and mono_bg != _new_c: old_color_choices_mono.append((mono_bg, 5))
                    for c_in in overall_input_colors:
                        if c_in != _new_c: old_color_choices_mono.append((c_in, 2))
                    if not old_color_choices_mono and overall_input_colors:
                        for c_in in overall_input_colors: old_color_choices_mono.append((c_in, 1))
                    if old_color_choices_mono: colors, weights = zip(*old_color_choices_mono); _old_c = random.choices(colors, weights=weights, k=1)[0]
                    else: _old_c = mono_bg if mono_bg is not None and mono_bg != _new_c else (0 if _new_c != 0 else 1)
            if selected_path == "none" or selected_path == "general_fallback_mono_incomplete":
                selected_path = "general_weighted"; old_pool_items = {}
                for c in overall_input_colors: old_pool_items[c] = max(old_pool_items.get(c, 0), 2) 
                for c in overall_output_colors: old_pool_items[c] = max(old_pool_items.get(c, 0), 1) 
                final_old_pool = list(old_pool_items.keys()); final_old_weights = [old_pool_items[c] for c in final_old_pool]
                if not final_old_pool: final_old_pool = list(range(10)); final_old_weights = [1]*10
                _old_c = random.choices(final_old_pool, weights=final_old_weights, k=1)[0]; new_pool_items = {}
                for c in overall_output_colors: new_pool_items[c] = max(new_pool_items.get(c, 0), 2) 
                for c in overall_input_colors: new_pool_items[c] = max(new_pool_items.get(c, 0), 1) 
                final_new_pool = list(new_pool_items.keys()); final_new_weights = [new_pool_items[c] for c in final_new_pool]
                if not final_new_pool: final_new_pool = list(range(10)); final_new_weights = [1]*10
                _new_c = random.choices(final_new_pool, weights=final_new_weights, k=1)[0]
            if _old_c == _new_c and random.random() < 0.9: 
                candidate_new_colors = []
                if overall_output_colors: candidate_new_colors.extend([c for c in overall_output_colors if c != _old_c])
                if not candidate_new_colors: candidate_new_colors.extend([c for c in range(10) if c != _old_c])
                if candidate_new_colors: _new_c = random.choice(candidate_new_colors)
            return [_old_c, _new_c]
        elif op_name == "crop_to_content":
            bg_color_arg = 0
            if patterns.get('color_changes') and patterns['color_changes'].get('overall_input_colors'):
                input_colors = set(patterns['color_changes']['overall_input_colors'])
                if input_colors and random.random() < 0.3: available_colors = list(input_colors - {0}) or [0]; bg_color_arg = random.choice(available_colors)
            return [bg_color_arg] 
        elif op_name == "isolate_color":
            all_colors = set()
            if patterns.get('color_changes'):
                if patterns['color_changes'].get('overall_input_colors'): all_colors.update(patterns['color_changes']['overall_input_colors'])
                if patterns['color_changes'].get('overall_output_colors'): all_colors.update(patterns['color_changes']['overall_output_colors'])
            available_colors = list(all_colors)
            if not available_colors: available_colors = list(range(10))
            color_to_keep = random.choice(available_colors); replacement_color = 0
            if available_colors and random.random() < 0.3: replacement_color = random.choice(list(set(available_colors) - {color_to_keep}) or [0])
            return [color_to_keep, replacement_color]
        elif op_name == "extract_largest_object_by_color":
            input_colors = set()
            if patterns.get('color_changes'):
                overall_in_colors = patterns['color_changes'].get('overall_input_colors', [])
                if overall_in_colors: input_colors.update(overall_in_colors)
                else: 
                    for mapping in patterns['color_changes'].get('mappings_per_example', []):
                        if 'input_colors' in mapping and isinstance(mapping['input_colors'], (set, list)): input_colors.update(mapping['input_colors'])
            available_colors = list(input_colors - {0}) 
            if not available_colors: available_colors = list(range(1, 10))
            return [random.choice(available_colors) if available_colors else 1]
        elif op_name == "extend_pattern": return [random.choice(["up", "down", "left", "right"])]
        elif op_name == "resize_to_match":
            target_shapes = []
            if patterns.get('size_change'):
                for change in patterns['size_change']['changes']:
                    if 'output_size' in change and len(change['output_size']) == 2: target_shapes.append(tuple(map(int,change['output_size']))) 
            if target_shapes and random.random() < 0.7: return [random.choice(target_shapes)]
            else: return [(random.randint(1, 30), random.randint(1, 30))]
        elif op_name == "pad_to_size":
            target_dims = []
            if patterns.get('size_change'):
                for change in patterns['size_change']['changes']:
                    if 'output_size' in change and len(change['output_size']) == 2: target_dims.append(tuple(map(int,change['output_size'])))
            if target_dims and random.random() < 0.7: h, w = random.choice(target_dims); return [h, w]
            else: return [random.randint(1, 30), random.randint(1, 30)]
        elif op_name == "roll_rows" or op_name == "roll_cols":
            max_abs_shift = 5; shifts = list(range(-max_abs_shift, max_abs_shift + 1))
            if 0 in shifts: shifts.remove(0); 
            if not shifts: shifts = [1] 
            weights = [1/(abs(s) + 0.5) for s in shifts]; shift_val = random.choices(shifts, weights=weights, k=1)[0]
            return [shift_val]
        elif op_name in ["color_object_by_rank", "delete_object_by_rank"]:
            rank_arg = random.randint(0, 3); color_arg = random.randint(0, 9); largest_arg = random.choice([True, False]); obj_detect_bg_color_arg = 0
            if patterns.get('color_changes') and patterns['color_changes'].get('overall_input_colors'):
                 input_colors = set(patterns['color_changes']['overall_input_colors'])
                 if 0 in input_colors and random.random() < 0.7: obj_detect_bg_color_arg = 0
                 elif input_colors and random.random() < 0.3: obj_detect_bg_color_arg = random.choice(list(input_colors))
            return [rank_arg, color_arg, largest_arg, obj_detect_bg_color_arg]
        elif op_name == "extract_mask_of_object_by_rank":
            rank_arg = random.randint(0, 3); mask_fg_color_arg = random.choice([1, random.randint(1,9)]); mask_bg_color_arg = 0 
            if mask_fg_color_arg == mask_bg_color_arg: mask_bg_color_arg = (mask_fg_color_arg + 1) % 10
            largest_arg = random.choice([True, False]); obj_detect_bg_color_arg = 0 
            if patterns.get('color_changes') and patterns['color_changes'].get('overall_input_colors'):
                 input_colors = set(patterns['color_changes']['overall_input_colors'])
                 if 0 in input_colors and random.random() < 0.7: obj_detect_bg_color_arg = 0
                 elif input_colors and random.random() < 0.3: obj_detect_bg_color_arg = random.choice(list(input_colors))
            return [rank_arg, mask_fg_color_arg, mask_bg_color_arg, largest_arg, obj_detect_bg_color_arg]
        elif op_name == "crop_to_object_by_rank":
            rank_arg = random.randint(0, 3); fallback_fill_color_arg = 0; largest_arg = random.choice([True, False]); obj_detect_bg_color_arg = 0
            if patterns.get('color_changes') and patterns['color_changes'].get('overall_input_colors'):
                 input_colors = set(patterns['color_changes']['overall_input_colors'])
                 if 0 in input_colors and random.random() < 0.7:
                     obj_detect_bg_color_arg = 0; fallback_fill_color_arg = 0 
                 elif input_colors and random.random() < 0.3:
                    chosen_bg = random.choice(list(input_colors)); obj_detect_bg_color_arg = chosen_bg; fallback_fill_color_arg = chosen_bg
            return [rank_arg, fallback_fill_color_arg, largest_arg, obj_detect_bg_color_arg]
        elif op_name == "translate_content":
            dr_arg, dc_arg = random.randint(-5, 5), random.randint(-5, 5)
            # Ensure not (0,0) shift if possible by re-picking to encourage actual translation
            retry_count = 0
            while dr_arg == 0 and dc_arg == 0 and retry_count < 5: # Limit retries
                dr_arg, dc_arg = random.randint(-5, 5), random.randint(-5, 5)
                retry_count += 1
            if dr_arg == 0 and dc_arg == 0: # If still (0,0), make one component non-zero
                if random.random() < 0.5: dr_arg = random.choice([-1,1])
                else: dc_arg = random.choice([-1,1])

            bg_arg = 0 # Default background color for translation
            
            content_shift_info = patterns.get('content_shift_patterns', {})
            if content_shift_info.get('consistent_shift') and random.random() < 0.8: # High chance to use detected
                shift_vec = content_shift_info.get('shift_vector')
                shift_bg = content_shift_info.get('shift_background_color') # This is the key for the bg color for translation
                if shift_vec and shift_bg is not None: # shift_bg can be 0
                    dr_arg, dc_arg = shift_vec[0], shift_vec[1]
                    bg_arg = shift_bg
            else: # Fallback if not consistent or by random chance for exploration
                # Use overall input colors to guess a sensible background if not using detected one
                if patterns.get('color_changes') and patterns['color_changes'].get('overall_input_colors'):
                    input_colors = list(patterns['color_changes']['overall_input_colors'])
                    if 0 in input_colors: # Prioritize 0 if it's present
                        bg_arg = 0
                    elif input_colors and random.random() < 0.3: # Otherwise, sometimes pick a random input color
                         bg_arg = random.choice(input_colors)
                    # If 0 not in input colors and didn't pick random, bg_arg remains 0 (a common default)
            return [dr_arg, dc_arg, bg_arg]
        else: 
            if library:
                primitive = library.get_primitive(op_name)
                if primitive and primitive.arity == 0: return []
            # Fallback for other ops or if library not provided for arity check
            return [] # Assuming 0-arity if not specified elsewhere
    
    def mutate(self, library: PrimitiveLibrary,
               patterns: Dict[str, Any] = None,
               effective_single_heuristic_ops: Optional[List[Tuple[str, List[Any]]]] = None,
               suggested_ops_from_reasoner: Optional[List[str]] = None,
               stagnation_level: float = 0.0,
               arg_mutation_prob: float = 0.1,
               add_op_prob: float = 0.1,
               remove_op_prob: float = 0.1,
               change_op_prob: float = 0.05,
               swap_op_prob: float = 0.05,
               stagnation_boost_factor: float = 0.5
               ) -> 'Program':
        new_ops = deepcopy(self.operations)

        # Adjust base probabilities by stagnation level
        eff_arg_mutation_prob = arg_mutation_prob * (1 + stagnation_level * stagnation_boost_factor)
        eff_add_op_prob = add_op_prob * (1 + stagnation_level * stagnation_boost_factor)
        eff_remove_op_prob = remove_op_prob * (1 + stagnation_level * stagnation_boost_factor)
        eff_change_op_prob = change_op_prob * (1 + stagnation_level * stagnation_boost_factor)
        eff_swap_op_prob = swap_op_prob * (1 + stagnation_level * stagnation_boost_factor)

        # 1. Mutate existing op's arguments
        if random.random() < eff_arg_mutation_prob and new_ops:
            idx = random.randint(0, len(new_ops) - 1)
            op_name, old_args = new_ops[idx]
            new_generated_args = Program._generate_args_for_op(op_name, patterns, library) 
            primitive = library.get_primitive(op_name)
            if primitive and len(new_generated_args) == primitive.arity:
                 new_ops[idx] = (op_name, new_generated_args)

        # 2. Add a new operation
        if random.random() < eff_add_op_prob:
            primitives = library.list_primitives()
            if primitives:
                chosen_op_name: Optional[str] = None
                chosen_op_args: Optional[List[Any]] = None
                
                if stagnation_level > 0.05 : 
                    rand_for_heuristic_source = random.random()
                    prob_effective = 0.3 * stagnation_level 
                    prob_suggested_heuristic = 0.4 * stagnation_level 

                    if effective_single_heuristic_ops and rand_for_heuristic_source < prob_effective:
                        chosen_op_tuple = random.choice(effective_single_heuristic_ops)
                        if chosen_op_tuple[0]: 
                            chosen_op_name, chosen_op_args = chosen_op_tuple
                    elif suggested_ops_from_reasoner and rand_for_heuristic_source < (prob_effective + prob_suggested_heuristic):
                        chosen_op_name = random.choice(suggested_ops_from_reasoner)
                
                if chosen_op_name is None: 
                    use_suggested_base_prob = 0.4 
                    if suggested_ops_from_reasoner and random.random() < use_suggested_base_prob:
                        chosen_op_name = random.choice(suggested_ops_from_reasoner)
                    else:
                        chosen_op_name = random.choice(primitives)
                
                if chosen_op_args is None and chosen_op_name is not None: 
                    chosen_op_args = Program._generate_args_for_op(chosen_op_name, patterns, library)
                
                if chosen_op_name and chosen_op_args is not None:
                    primitive_obj = library.get_primitive(chosen_op_name)
                    if primitive_obj and len(chosen_op_args) == primitive_obj.arity:
                        insert_pos = random.randint(0, len(new_ops))
                        new_ops.insert(insert_pos, (chosen_op_name, chosen_op_args))

        # 3. Remove an operation
        if random.random() < eff_remove_op_prob and len(new_ops) > 1: 
            new_ops.pop(random.randint(0, len(new_ops) - 1))
        
        # 4. Change an operation type
        if random.random() < eff_change_op_prob and new_ops:
            idx_to_change = random.randint(0, len(new_ops) - 1)
            primitives = library.list_primitives()
            if primitives:
                new_op_name = random.choice(primitives)
                new_args = Program._generate_args_for_op(new_op_name, patterns, library)
                primitive_obj = library.get_primitive(new_op_name)
                if primitive_obj and len(new_args) == primitive_obj.arity:
                    new_ops[idx_to_change] = (new_op_name, new_args)

        # 5. Swap order of two operations
        if random.random() < eff_swap_op_prob and len(new_ops) >= 2:
            idx1, idx2 = random.sample(range(len(new_ops)), 2)
            new_ops[idx1], new_ops[idx2] = new_ops[idx2], new_ops[idx1]
            
        return Program(new_ops)

    def crossover(self, other: 'Program') -> 'Program':
        if not self.operations: return Program(deepcopy(other.operations))
        if not other.operations: return Program(deepcopy(self.operations))
        split_self = random.randint(0, len(self.operations)) 
        split_other = random.randint(0, len(other.operations))
        
        new_ops_part1 = self.operations[:split_self]
        new_ops_part2 = other.operations[split_other:]
        
        new_ops = new_ops_part1 + new_ops_part2
        
        if not new_ops and (self.operations or other.operations):
            return Program(deepcopy(self.operations)) if random.random() < 0.5 else Program(deepcopy(other.operations))
            
        return Program(deepcopy(new_ops))


class InductiveReasoner:
    def __init__(self): self.learned_patterns = [] 
    def analyze_examples(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        return {
            'size_change': self._analyze_size_changes(examples),
            'color_changes': self._analyze_color_changes(examples),
            'shape_changes': self._analyze_shape_changes(examples),
            'output_monochrome_check': self._analyze_output_monochrome(examples),
            'translation_roll_patterns': self._analyze_translation_roll(examples), 
            'content_shift_patterns': self._analyze_content_shift(examples), # New
        }
    def _analyze_size_changes(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        size_changes = []
        for inp, out in examples:
            if inp.ndim != 2 or out.ndim != 2: continue 
            inp_shape = inp.shape; out_shape = out.shape
            size_changes.append({
                'input_size': inp_shape, 'output_size': out_shape,
                'size_ratio': (out_shape[0] / inp_shape[0] if inp_shape[0]!=0 else 0, 
                               out_shape[1] / inp_shape[1] if inp_shape[1]!=0 else 0)
            })
        return {'changes': size_changes, 'consistent_output_shape': self._get_consistent_output_shape(examples)}
    def _get_consistent_output_shape(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Tuple[int,int]]:
        if not examples: return None
        first_out_shape = examples[0][1].shape
        if all(ex[1].shape == first_out_shape for ex in examples): return first_out_shape
        return None
    def _analyze_color_changes(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        color_mappings = []; all_input_colors = set(); all_output_colors = set()
        for inp, out in examples:
            inp_colors = set(inp.flatten()); out_colors = set(out.flatten())
            all_input_colors.update(inp_colors); all_output_colors.update(out_colors)
            color_mappings.append({
                'input_colors': list(inp_colors), 'output_colors': list(out_colors),
                'new_colors': list(out_colors - inp_colors), 'removed_colors': list(inp_colors - out_colors) 
            })
        return {'mappings_per_example': color_mappings, 'overall_input_colors': list(all_input_colors), 'overall_output_colors': list(all_output_colors)}
    def _analyze_shape_changes(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        transformations = []; consistent_rotation = True; consistent_flip = True; consistent_transpose = True 
        first_rotation_type = None; first_flip_type = None; first_transpose_match = None 
        if not examples: return {'transformations_per_example': [], 'consistent_rotation': False, 'consistent_rotation_k': None,
                                'consistent_flip': False, 'consistent_flip_type': None, 'consistent_transpose': False}
        for i, (inp, out) in enumerate(examples):
            rotation_k = None; flip_detected = None; transpose_match = False; current_transform = {}
            if inp.ndim == 2 and out.ndim == 2 : 
                if inp.shape == out.shape: rotation_k = self._check_rotation(inp, out); flip_detected = self._check_flip(inp, out) 
                if inp.shape[::-1] == out.shape: 
                     if np.array_equal(np.transpose(inp), out): transpose_match = True
            transformations.append({'rotated': rotation_k is not None, 'rotation_k': rotation_k, 
                                    'flipped': flip_detected is not None, 'flip_type': flip_detected, 'transposed': transpose_match})
            if i == 0: first_rotation_type = rotation_k; first_flip_type = flip_detected; first_transpose_match = transpose_match 
            else:
                if rotation_k != first_rotation_type: consistent_rotation = False
                if flip_detected != first_flip_type: consistent_flip = False
                if transpose_match != first_transpose_match: consistent_transpose = False 
        final_consistent_transpose = consistent_transpose and (first_transpose_match if examples else False)
        return {'transformations_per_example': transformations, 'consistent_rotation': consistent_rotation, 'consistent_rotation_k': first_rotation_type if consistent_rotation else None,
                'consistent_flip': consistent_flip, 'consistent_flip_type': first_flip_type if consistent_flip else None, 'consistent_transpose': final_consistent_transpose}
    def _check_rotation(self, inp: np.ndarray, out: np.ndarray) -> Optional[int]:
        if inp.shape != out.shape: return None
        for k in range(4): 
            if np.array_equal(out, np.rot90(inp, k)): return k
        return None
    def _check_flip(self, inp: np.ndarray, out: np.ndarray) -> Optional[str]:
        if inp.shape != out.shape: return None
        if np.array_equal(out, np.fliplr(inp)): return 'horizontal'
        if np.array_equal(out, np.flipud(inp)): return 'vertical'
        if np.array_equal(out, np.fliplr(np.flipud(inp))): return 'both' 
        return None
    def _analyze_output_monochrome(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        is_consistently_monochrome = True; consistent_foreground_color = None; consistent_background_color = None 
        if not examples: return {'is_monochrome': False}
        for i, (inp, out) in enumerate(examples):
            if out.size == 0: is_consistently_monochrome = False; break
            unique_colors = np.unique(out); current_fg = None; current_bg = None 
            if len(unique_colors) == 1: current_fg = unique_colors[0] # Background is undefined or same as fg
            elif len(unique_colors) == 2:
                if 0 in unique_colors: current_bg = 0; current_fg = [c for c in unique_colors if c != 0][0]
                else: # No 0, pick less frequent as bg
                    counts = {c: np.sum(out == c) for c in unique_colors}; sorted_colors_by_count = sorted(unique_colors, key=lambda c: counts[c])
                    current_bg = sorted_colors_by_count[0]; current_fg = sorted_colors_by_count[1]
            else: is_consistently_monochrome = False; break # More than 2 colors
            
            if i == 0: consistent_foreground_color = current_fg; consistent_background_color = current_bg 
            else:
                if current_fg != consistent_foreground_color or current_bg != consistent_background_color: is_consistently_monochrome = False; break
        
        if is_consistently_monochrome and consistent_foreground_color is not None:
            result = {'is_monochrome': True, 'foreground_color': int(consistent_foreground_color)}
            if consistent_background_color is not None: result['background_color'] = int(consistent_background_color)
            else: # Only one color in output, assume background is 0 if fg is not 0, else 1.
                  result['background_color'] = 0 if int(consistent_foreground_color) != 0 else 1
            return result
        else: return {'is_monochrome': False}

    def _analyze_translation_roll(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        consistent_row_roll_shift = None; consistent_col_roll_shift = None
        if not examples: return {'consistent_row_roll': False, 'consistent_row_roll_shift': None, 'consistent_col_roll': False, 'consistent_col_roll_shift': None}
        
        candidate_row_shifts_per_example = []; possible_row_roll = True
        for inp, out in examples:
            if inp.ndim != 2 or out.ndim != 2 or inp.shape != out.shape or inp.shape[0] == 0: possible_row_roll = False; break
            current_example_shifts = []
            # Check for positive and negative shifts for roll
            for shift_val_abs in range(1, inp.shape[0]): # Check shifts up to N-1
                 for shift_sign in [1, -1]:
                    shift = shift_val_abs * shift_sign
                    if np.array_equal(np.roll(inp, shift, axis=0), out): current_example_shifts.append(shift)
            if not current_example_shifts: possible_row_roll = False; break
            candidate_row_shifts_per_example.append(set(current_example_shifts))
        
        if possible_row_roll and candidate_row_shifts_per_example:
            common_row_shifts = candidate_row_shifts_per_example[0]
            for i in range(1, len(candidate_row_shifts_per_example)): common_row_shifts.intersection_update(candidate_row_shifts_per_example[i])
            if len(common_row_shifts) == 1: consistent_row_roll_shift = common_row_shifts.pop()

        candidate_col_shifts_per_example = []; possible_col_roll = True
        for inp, out in examples:
            if inp.ndim != 2 or out.ndim != 2 or inp.shape != out.shape or inp.shape[1] == 0: possible_col_roll = False; break
            current_example_shifts = []
            for shift_val_abs in range(1, inp.shape[1]):
                 for shift_sign in [1, -1]:
                    shift = shift_val_abs * shift_sign
                    if np.array_equal(np.roll(inp, shift, axis=1), out): current_example_shifts.append(shift)
            if not current_example_shifts: possible_col_roll = False; break
            candidate_col_shifts_per_example.append(set(current_example_shifts))

        if possible_col_roll and candidate_col_shifts_per_example:
            common_col_shifts = candidate_col_shifts_per_example[0]
            for i in range(1, len(candidate_col_shifts_per_example)): common_col_shifts.intersection_update(candidate_col_shifts_per_example[i])
            if len(common_col_shifts) == 1: consistent_col_roll_shift = common_col_shifts.pop()
            
        return {'consistent_row_roll': consistent_row_roll_shift is not None, 'consistent_row_roll_shift': consistent_row_roll_shift,
                'consistent_col_roll': consistent_col_roll_shift is not None, 'consistent_col_roll_shift': consistent_col_roll_shift}

    # New helper methods for content shift analysis
    def _get_content_bounding_box_and_cropped(self, grid: np.ndarray, bg_color: int) -> Tuple[Optional[Tuple[int,int,int,int]], Optional[np.ndarray]]:
        if grid.size == 0: return None, None # Or handle as fully background
        content_indices = np.where(grid != bg_color)
        if len(content_indices[0]) == 0: 
            # Grid is all bg_color, or empty grid passed
            # For consistency, return a 1x1 grid of bg_color if input was not None.
            # This helps in np.array_equal checks later if both cropped_inp and cropped_out are this.
            return None, np.array([[bg_color]], dtype=grid.dtype if grid.size > 0 else int) 

        min_r, max_r = content_indices[0].min(), content_indices[0].max()
        min_c, max_c = content_indices[1].min(), content_indices[1].max()
        cropped_grid = grid[min_r:max_r+1, min_c:max_c+1].copy()
        return (min_r, min_c, max_r, max_c), cropped_grid

    def _check_shift_for_bg(self, examples: List[Tuple[np.ndarray, np.ndarray]], assumed_bg_color: int) \
        -> Tuple[bool, Optional[Tuple[int, int]], List[Dict[str, Any]]]:
        
        example_shifts = []
        possible_consistent_shift_for_this_bg = True
        details_for_this_bg_check = []

        for inp, out in examples:
            current_detail = {'assumed_bg_for_shift_check': assumed_bg_color}
            if inp.ndim != 2 or out.ndim != 2:
                possible_consistent_shift_for_this_bg = False; current_detail['error'] = 'Non-2D grid'
                details_for_this_bg_check.append(current_detail); break

            bbox_inp, cropped_inp = self._get_content_bounding_box_and_cropped(inp, assumed_bg_color)
            bbox_out, cropped_out = self._get_content_bounding_box_and_cropped(out, assumed_bg_color)

            if cropped_inp is None or cropped_out is None : # Should not happen with current _get_content_bounding_box_and_cropped
                possible_consistent_shift_for_this_bg = False; current_detail['error'] = 'Cropped grid is None'
                details_for_this_bg_check.append(current_detail); break


            if bbox_inp is None: # Content in input is empty (all assumed_bg_color)
                if bbox_out is None: # Content in output is also empty
                    example_shifts.append((0,0)) # No shift of "nothing"
                    current_detail.update({'shift': (0,0), 'input_content_empty': True, 'output_content_empty': True})
                else: # Input content empty, output content not
                    possible_consistent_shift_for_this_bg = False
                    current_detail.update({'input_content_empty': True, 'output_content_present': True})
                    details_for_this_bg_check.append(current_detail); break
            elif bbox_out is None: # Input content present, output content empty
                possible_consistent_shift_for_this_bg = False
                current_detail.update({'input_content_present': True, 'output_content_empty': True})
                details_for_this_bg_check.append(current_detail); break
            else: # Both have some content (non-assumed_bg_color pixels)
                if cropped_inp.shape == cropped_out.shape and np.array_equal(cropped_inp, cropped_out):
                    min_r_inp, min_c_inp, _, _ = bbox_inp
                    min_r_out, min_c_out, _, _ = bbox_out
                    dr = min_r_out - min_r_inp
                    dc = min_c_out - min_c_inp
                    example_shifts.append((dr, dc))
                    current_detail.update({'shift': (dr,dc), 'cropped_content_match': True})
                else:
                    possible_consistent_shift_for_this_bg = False
                    current_detail.update({'cropped_content_match': False, 
                                       'cropped_inp_shape': cropped_inp.shape, 
                                       'cropped_out_shape': cropped_out.shape,
                                       'cropped_inp_sum': np.sum(cropped_inp), # For debugging differences
                                       'cropped_out_sum': np.sum(cropped_out)})
                    details_for_this_bg_check.append(current_detail); break
            details_for_this_bg_check.append(current_detail)
        
        if possible_consistent_shift_for_this_bg and example_shifts:
            first_shift_vector = example_shifts[0]
            # All shifts must be identical
            if all(s == first_shift_vector for s in example_shifts):
                if first_shift_vector != (0,0): # A non-zero shift is a detection
                    return True, first_shift_vector, details_for_this_bg_check
                else: # A (0,0) shift is consistent but not a "detection" for this purpose
                    return False, (0,0), details_for_this_bg_check 
        
        return False, None, details_for_this_bg_check


    def _analyze_content_shift(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        result = {'consistent_shift': False, 'shift_vector': None, 
                  'shift_background_color': None, 'details_per_example': []}
        if not examples: return result

        # Attempt 1: Assume background color for content definition is 0
        detected_0, vector_0, details_0 = self._check_shift_for_bg(examples, 0)
        if detected_0 and vector_0 is not None: # vector_0 will not be (0,0) if detected_0 is True
            result['consistent_shift'] = True
            result['shift_vector'] = vector_0
            result['shift_background_color'] = 0
            result['details_per_example'] = details_0
            return result

        # Attempt 2: If BG 0 fails, try most common color in the first input, if it's not 0
        if examples[0][0].size > 0:
            inp0 = examples[0][0]
            colors, counts = np.unique(inp0, return_counts=True)
            if colors.size > 0:
                most_common_color_inp0 = int(colors[np.argmax(counts)])
                if most_common_color_inp0 != 0:
                    detected_mc, vector_mc, details_mc = self._check_shift_for_bg(examples, most_common_color_inp0)
                    if detected_mc and vector_mc is not None:
                        result['consistent_shift'] = True
                        result['shift_vector'] = vector_mc
                        result['shift_background_color'] = most_common_color_inp0
                        result['details_per_example'] = details_mc
                        return result
                    else: # Store details from this attempt if it was made
                        result['details_per_example'] = details_mc
                        return result # Return after this attempt
        
        # If no shift detected, or only (0,0) shift from BG=0 attempt
        result['details_per_example'] = details_0 
        # If vector_0 was (0,0) it means it was consistent but not a "true" shift signal.
        # 'consistent_shift' remains False.
        return result

    def suggest_operations(self, patterns: Dict[str, Any]) -> List[str]:
        suggestions = []; monochrome_info = patterns.get('output_monochrome_check', {})
        if monochrome_info.get('is_monochrome'):
            suggestions.extend(['isolate_color', 'fill_color'])
            if monochrome_info.get('background_color') is not None: suggestions.append('crop_to_content') 
        if patterns.get('shape_changes'):
            sc = patterns['shape_changes']
            if sc.get('consistent_rotation') and sc.get('consistent_rotation_k') is not None:
                k = sc['consistent_rotation_k']
                if k == 3 : suggestions.append('rotate_90') # k=3 means -90 deg or 270 deg clockwise
                if k == 1 : suggestions.append('rotate_90') # k=1 means +90 deg or -270 deg. np.rot90 uses counter-clockwise
                                                          # My primitive _rotate_90 is k=-1 (90 deg clockwise)
                                                          # So, if k=1 (90 deg CCW), we need rot90 three times or equivalent.
                                                          # Let's assume k means np.rot90(m, k). If k=1 (90deg CCW), use rotate_90 (primitive is CW) three times.
                                                          # This is complex mapping. For now, k=3 (270 CCW) is one rotate_90 (CW). k=2 is rotate_180.
                if k == 2 : suggestions.append('rotate_180')
            if sc.get('consistent_flip') and sc.get('consistent_flip_type') is not None:
                flip_type = sc['consistent_flip_type']
                if flip_type == 'horizontal': suggestions.append('flip_horizontal')
                if flip_type == 'vertical': suggestions.append('flip_vertical')
                if flip_type == 'both': suggestions.extend(['flip_horizontal', 'flip_vertical']) 
            if sc.get('consistent_transpose'): suggestions.append('transpose')
        if patterns.get('color_changes'):
            cc = patterns['color_changes']
            if any(ex.get('new_colors') for ex in cc.get('mappings_per_example',[])) or \
               any(ex.get('removed_colors') for ex in cc.get('mappings_per_example',[])):
                suggestions.extend(['fill_color', 'replace_color'])
            fewer_output_colors_count = 0
            for ex_map in cc.get('mappings_per_example',[]):
                if len(ex_map.get('output_colors',[])) < len(ex_map.get('input_colors',[])): fewer_output_colors_count +=1
            if cc.get('mappings_per_example') and fewer_output_colors_count > len(cc.get('mappings_per_example',[])) / 2: suggestions.append('isolate_color')
        if patterns.get('size_change'):
            sz = patterns['size_change']
            if sz.get('consistent_output_shape'): suggestions.extend(['resize_to_match', 'pad_to_size']) 
            smaller_output_count = 0
            if 'changes' in sz and sz['changes']: 
                for change in sz['changes']:
                    inp_s = np.prod(change['input_size']) if len(change['input_size'])==2 else 0
                    out_s = np.prod(change['output_size']) if len(change['output_size'])==2 else 0
                    if out_s > 0 and out_s < inp_s : smaller_output_count +=1
                if smaller_output_count > 0 and smaller_output_count >= len(sz['changes']) / 2: suggestions.append('crop_to_content')
        if patterns.get('translation_roll_patterns'): 
            trp = patterns['translation_roll_patterns']
            if trp.get('consistent_row_roll'): suggestions.append('roll_rows')
            if trp.get('consistent_col_roll'): suggestions.append('roll_cols')
        
        # New: Suggest translate_content based on content_shift_patterns
        if patterns.get('content_shift_patterns'):
            csp = patterns['content_shift_patterns']
            if csp.get('consistent_shift') and csp.get('shift_vector') != (0,0):
                suggestions.append('translate_content')

        if not suggestions : suggestions.append('crop_to_content') # Default fallback
        return list(set(suggestions)) 

class ARCProgramSearch:
    def __init__(self, population_size: int = 50, max_generations: int = 100,
                 elite_percentage: float = 0.10, tournament_size: int = 3,
                 arg_mutation_prob: float = 0.15, add_op_prob: float = 0.15,
                 remove_op_prob: float = 0.10, change_op_prob: float = 0.10,
                 swap_op_prob: float = 0.05, stagnation_boost_factor_mutate: float = 0.5,
                 cataclysm_stagnation_threshold: float = 0.7, cataclysm_prob: float = 0.05):
        self.population_size = population_size
        self.max_generations = max_generations
        self.library = PrimitiveLibrary()
        self.reasoner = InductiveReasoner()
        self.population = []

        self.elite_percentage = elite_percentage
        self.tournament_size = tournament_size
        self.arg_mutation_prob = arg_mutation_prob
        self.add_op_prob = add_op_prob
        self.remove_op_prob = remove_op_prob
        self.change_op_prob = change_op_prob
        self.swap_op_prob = swap_op_prob
        self.stagnation_boost_factor_mutate = stagnation_boost_factor_mutate
        self.cataclysm_stagnation_threshold = cataclysm_stagnation_threshold
        self.cataclysm_prob = cataclysm_prob
    
    def _calculate_iou(self, grid1: np.ndarray, grid2: np.ndarray, background_color: int = 0) -> float:
        if grid1.shape != grid2.shape: return 0.0 
        # Ensure grid1 and grid2 are boolean masks for IoU calculation based on presence/absence of "object" pixels
        # Pixels are part of an "object" if they are not the background_color
        mask1 = grid1 != background_color
        mask2 = grid2 != background_color
        
        # Intersection: pixels that are objects in both AND have the same color value
        # Union: pixels that are objects in either grid1 or grid2
        # Revised Intersection: True where (mask1 AND mask2) AND (grid1 == grid2)
        # However, standard IoU for segmentation is often just based on mask overlap.
        # Let's stick to mask overlap for IoU, pixel_similarity handles color matching.
        
        intersection_pixels = np.logical_and(mask1, mask2) 
        intersection_count = np.sum(intersection_pixels)
        
        union_pixels = np.logical_or(mask1, mask2)
        union_count = np.sum(union_pixels)
        
        if union_count == 0: # Both masks are empty (all background)
            return 1.0 # Perfect match if both are all background
        
        return float(intersection_count) / float(union_count)
    
    def evaluate_program(self, program: Program, examples: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        fitness_scores = []
        if not examples: return 0.0
        for inp, expected_out in examples:
            input_for_program = np.array(inp) if not isinstance(inp, np.ndarray) else inp.copy()
            actual_out = program.execute(input_for_program, self.library)
            
            if actual_out is None: fitness_scores.append(0.0); continue 
            
            pixel_similarity = 0.0; iou_score = 0.0; 
            
            # Determine background color for IoU from expected_out (most frequent, or 0 if present)
            bg_color_expected = 0 
            if expected_out.size > 0:
                unique_colors, counts = np.unique(expected_out, return_counts=True)
                if unique_colors.size > 0: # Has some colors
                    if 0 in unique_colors: bg_color_expected = 0
                    else: bg_color_expected = unique_colors[np.argmax(counts)] # Most frequent if 0 not present
            
            if actual_out.shape == expected_out.shape:
                pixel_similarity = np.mean(actual_out == expected_out)
                iou_score = self._calculate_iou(actual_out, expected_out, bg_color_expected)
            # else: pixel_similarity and iou_score remain 0.0 (handled by initialization)
            
            # Weighted average for fitness component
            combined_similarity = 0.7 * pixel_similarity + 0.3 * iou_score
            fitness_scores.append(combined_similarity)

        if not fitness_scores: return 0.0 
        average_similarity = sum(fitness_scores) / len(fitness_scores)
        
        # Complexity penalty: penalize longer programs more harshly
        # Original: 0.005 * (program.complexity ** 1.5)
        # Adjusted for potentially more complex solutions being necessary
        complexity_penalty = 0.002 * (program.complexity ** 1.3) 
        
        fitness = average_similarity - complexity_penalty
        return max(0.0, fitness) 

    def _create_random_individual(self, patterns: Dict[str, Any], suggested_ops: List[str]) -> Program:
        program_length = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1], k=1)[0]
        operations = []
        available_primitives = self.library.list_primitives()
        if not available_primitives: return Program([])

        for _ in range(program_length):
            op_name = ""
            # Give higher preference to suggested ops if available
            if suggested_ops and random.random() < 0.75: op_name = random.choice(suggested_ops)
            else: op_name = random.choice(available_primitives)
            
            args = Program._generate_args_for_op(op_name, patterns, self.library)
            primitive = self.library.get_primitive(op_name)
            if primitive and len(args) == primitive.arity:
                operations.append((op_name, args))
        return Program(operations)

    def initialize_population(self, patterns: Dict[str, Any]) -> None:
        self.population = []
        suggested_ops = self.reasoner.suggest_operations(patterns)
        for _ in range(self.population_size):
            self.population.append(self._create_random_individual(patterns, suggested_ops))
    
    def _try_simple_solutions(self, examples: List[Tuple[np.ndarray, np.ndarray]], patterns: Dict[str, Any]) \
            -> Tuple[Optional[Program], List[Tuple[str, List[Any]]]]:
        print("Attempting simple heuristic solutions...")
        if not examples: return None, []
        
        successful_single_op_tuples: List[Tuple[str, List[Any]]] = []
        best_simple_program: Optional[Program] = None
        highest_simple_fitness: float = -1.0
        
        candidate_heuristic_ops: List[Tuple[str, List[Any]]] = []
        candidate_heuristic_ops.append(("", [])) # Identity program

        # Shape change heuristics
        shape_pats = patterns.get('shape_changes', {})
        if shape_pats.get('consistent_rotation') and shape_pats.get('consistent_rotation_k') is not None:
            k = shape_pats['consistent_rotation_k']
            # np.rot90(m, k) -> Primitive mapping
            # k=1 (90 CCW) -> rotate_90 three times (primitive is CW) - too complex for simple heuristic
            # k=2 (180) -> rotate_180
            # k=3 (270 CCW / 90 CW) -> rotate_90
            if k == 3: candidate_heuristic_ops.append(("rotate_90", []))
            elif k == 2: candidate_heuristic_ops.append(("rotate_180", []))
            # Not adding k=1 as it implies 3 ops for a single heuristic here
        if shape_pats.get('consistent_flip') and shape_pats.get('consistent_flip_type') is not None:
            flip_type = shape_pats['consistent_flip_type']
            if flip_type == 'horizontal': candidate_heuristic_ops.append(("flip_horizontal", []))
            elif flip_type == 'vertical': candidate_heuristic_ops.append(("flip_vertical", []))
            # Not adding 'both' as it's two ops
        if shape_pats.get('consistent_transpose'): candidate_heuristic_ops.append(("transpose", []))

        # Crop to content heuristic
        bg_candidates = {0} # Always try with 0
        mono_pats = patterns.get('output_monochrome_check', {})
        if mono_pats.get('is_monochrome') and 'background_color' in mono_pats and mono_pats['background_color'] is not None:
             bg_candidates.add(mono_pats['background_color'])
        # Also consider overall input background if consistent
        color_pats_overall = patterns.get('color_changes', {})
        if color_pats_overall.get('overall_input_colors'):
            # If a single color is overwhelmingly frequent in inputs, consider it
            # This is a simplification; a proper consistent input bg detector would be better
            if examples[0][0].size > 0:
                inp_colors_first, inp_counts_first = np.unique(examples[0][0], return_counts=True)
                if inp_colors_first.size > 0:
                    most_freq_inp_first = inp_colors_first[np.argmax(inp_counts_first)]
                    bg_candidates.add(int(most_freq_inp_first))

        for bg_try in list(bg_candidates): candidate_heuristic_ops.append(("crop_to_content", [bg_try]))

        # Replace color heuristic
        color_pats = patterns.get('color_changes', {}); mappings_per_example = color_pats.get('mappings_per_example', [])
        if mappings_per_example:
            potential_mappings = {}
            for ex_map in mappings_per_example:
                removed = ex_map.get('removed_colors', []); added = ex_map.get('new_colors', [])
                if len(removed) == 1 and len(added) == 1 and removed[0] != added[0]:
                    pair = (removed[0], added[0]); potential_mappings[pair] = potential_mappings.get(pair, 0) + 1
            if potential_mappings:
                num_ex = len(mappings_per_example)
                for (old_c, new_c), count in potential_mappings.items():
                    if count == num_ex : candidate_heuristic_ops.append(("replace_color", [old_c, new_c]))
        
        # Roll heuristics
        roll_pats = patterns.get('translation_roll_patterns', {})
        if roll_pats.get('consistent_row_roll') and roll_pats.get('consistent_row_roll_shift') is not None:
            candidate_heuristic_ops.append(("roll_rows", [roll_pats['consistent_row_roll_shift']]))
        if roll_pats.get('consistent_col_roll') and roll_pats.get('consistent_col_roll_shift') is not None:
            candidate_heuristic_ops.append(("roll_cols", [roll_pats['consistent_col_roll_shift']]))

        # New: Translate content heuristic
        content_shift_pats = patterns.get('content_shift_patterns', {})
        if content_shift_pats.get('consistent_shift'):
            shift_vec = content_shift_pats.get('shift_vector')
            shift_bg = content_shift_pats.get('shift_background_color')
            if shift_vec and shift_bg is not None and shift_vec != (0,0): # Ensure actual shift
                candidate_heuristic_ops.append(("translate_content", [shift_vec[0], shift_vec[1], shift_bg]))
        
        # Evaluate candidate heuristic ops
        for op_name, op_args in candidate_heuristic_ops:
            program_ops = []
            if op_name: program_ops.append((op_name, op_args)) # op_name can be "" for Identity
            
            prog = Program(program_ops)
            # Calculate fitness first, then check for perfect match
            prog.fitness = self.evaluate_program(prog, examples)
            all_match = True
            if not examples: all_match = False # Should not happen if called after example loading
            
            for inp, out_ex in examples:
                res = prog.execute(inp.copy(), self.library) # Use .copy() for safety
                if res is None or not np.array_equal(res, out_ex): 
                    all_match = False; break
            
            if all_match: # Perfect match on all training examples
                print(f"  Heuristic PERFECTLY SOLVED: {op_name or 'Identity'}({op_args if op_args else ''}), Fitness: {prog.fitness:.4f}")
                if prog.fitness > highest_simple_fitness: # This should usually be true if fitness is well-defined
                    highest_simple_fitness = prog.fitness
                    best_simple_program = prog
                # Add to list of successful single ops, even if not the absolute best (e.g. multiple perfect solutions)
                if op_name or op_args : # Don't add ('', []) unless it's the *only* solution
                     if not any(s_op == op_name and s_args == op_args for s_op, s_args in successful_single_op_tuples):
                          successful_single_op_tuples.append((op_name, op_args))
            elif prog.fitness > highest_simple_fitness: # Not a perfect match, but maybe best so far
                 highest_simple_fitness = prog.fitness
                 best_simple_program = prog # Store it as a potential fallback if no perfect simple solution

        if best_simple_program: 
            is_perfect = True
            for inp, out_ex in examples: # Re-check if best_simple_program is perfect
                res = best_simple_program.execute(inp.copy(), self.library)
                if res is None or not np.array_equal(res, out_ex): is_perfect = False; break
            
            if is_perfect:
                 print(f"Best simple heuristic solution (PERFECT MATCH): {best_simple_program.operations or 'Identity'}, Fitness: {best_simple_program.fitness:.4f}")
                 # Ensure this perfect one is in successful_single_op_tuples if it's a non-identity op
                 bp_ops_list = best_simple_program.operations
                 if bp_ops_list : # Non-identity
                     bp_op_name, bp_op_args = bp_ops_list[0]
                     if not any(s_op == bp_op_name and s_args == bp_op_args for s_op, s_args in successful_single_op_tuples):
                          successful_single_op_tuples.append((bp_op_name, bp_op_args))

            else: print(f"Best simple heuristic solution (NOT perfect): {best_simple_program.operations or 'Identity'}, Fitness: {best_simple_program.fitness:.4f}")
        else: print("No single simple heuristic operation perfectly solved all training examples. (No simple program found at all).")
        
        return best_simple_program, successful_single_op_tuples


    def _tournament_selection(self, current_population: List[Program]) -> Program:
        if not current_population: raise ValueError("Population cannot be empty for tournament selection.")
        if len(current_population) < self.tournament_size:
             # If population is too small, pick the best one to avoid errors with random.sample
             return max(current_population, key=lambda p: p.fitness) 

        contenders = random.sample(current_population, self.tournament_size)
        winner = max(contenders, key=lambda p: p.fitness)
        return winner

    def evolve_generation(self, examples: List[Tuple[np.ndarray, np.ndarray]],
                          patterns: Dict[str, Any],
                          successful_heuristic_ops: List[Tuple[str, List[Any]]],
                          suggested_ops_from_reasoner: List[str],
                          stagnation_level: float,
                          generations_since_improvement: int 
                          ) -> int: 
        for program in self.population:
            if not hasattr(program, 'fitness_calculated_this_gen') or not program.fitness_calculated_this_gen:
                 program.fitness = self.evaluate_program(program, examples)
                 program.fitness_calculated_this_gen = True 

        self.population.sort(key=lambda p: p.fitness, reverse=True)
        
        elite_count = max(1, int(self.population_size * self.elite_percentage)) 
        if not self.population: 
            self.initialize_population(patterns) # Attempt to re-init if empty
            if not self.population: return generations_since_improvement # Still empty, cannot proceed
            # Evaluate this new population
            for prog in self.population:
                prog.fitness = self.evaluate_program(prog, examples)
                prog.fitness_calculated_this_gen = True
            self.population.sort(key=lambda p: p.fitness, reverse=True)


        if stagnation_level >= self.cataclysm_stagnation_threshold and \
           random.random() < self.cataclysm_prob and \
           len(self.population) > elite_count : 
            
            print(f"  INFO: Cataclysmic mutation triggered at stagnation level {stagnation_level:.2f}!")
            new_population_cataclysm = deepcopy(self.population[:elite_count]) # Keep elites
            num_to_reinitialize = self.population_size - len(new_population_cataclysm)
            
            for _ in range(num_to_reinitialize):
                new_individual = self._create_random_individual(patterns, suggested_ops_from_reasoner)
                new_individual.fitness = self.evaluate_program(new_individual, examples) 
                new_individual.fitness_calculated_this_gen = True
                new_population_cataclysm.append(new_individual)
            
            self.population = new_population_cataclysm
            self.population.sort(key=lambda p: p.fitness, reverse=True) 
            return 0 

        new_population = deepcopy(self.population[:elite_count]) 
        
        prob_crossover = 0.7
        prob_mutation_after_crossover = 0.5 
        
        num_offspring_needed = self.population_size - len(new_population)

        if not self.population or len(self.population) == 0 : # Should be caught by re-init above
             self.population = new_population[:self.population_size] 
             return generations_since_improvement


        for _ in range(num_offspring_needed):
            p1 = self._tournament_selection(self.population)
            
            child: Program
            if random.random() < prob_crossover and len(self.population) > 1:  
                p2 = self._tournament_selection(self.population)
                num_attempts_p2 = 0 
                while p1 is p2 and num_attempts_p2 < 5 and len(self.population) > 1:
                    p2 = self._tournament_selection(self.population)
                    num_attempts_p2 +=1
                
                child = p1.crossover(p2)
                if random.random() < prob_mutation_after_crossover: 
                    child = child.mutate(
                        self.library, patterns,
                        effective_single_heuristic_ops=successful_heuristic_ops,
                        suggested_ops_from_reasoner=suggested_ops_from_reasoner,
                        stagnation_level=stagnation_level,
                        arg_mutation_prob=self.arg_mutation_prob, add_op_prob=self.add_op_prob,
                        remove_op_prob=self.remove_op_prob, change_op_prob=self.change_op_prob,
                        swap_op_prob=self.swap_op_prob, stagnation_boost_factor=self.stagnation_boost_factor_mutate)
            else: 
                child = p1.mutate( # Mutate a copy of the selected parent
                    self.library, patterns,
                    effective_single_heuristic_ops=successful_heuristic_ops,
                    suggested_ops_from_reasoner=suggested_ops_from_reasoner,
                    stagnation_level=stagnation_level,
                    arg_mutation_prob=self.arg_mutation_prob, add_op_prob=self.add_op_prob,
                    remove_op_prob=self.remove_op_prob, change_op_prob=self.change_op_prob,
                    swap_op_prob=self.swap_op_prob, stagnation_boost_factor=self.stagnation_boost_factor_mutate)

            child.fitness = self.evaluate_program(child, examples) 
            child.fitness_calculated_this_gen = True
            new_population.append(child)
        
        self.population = new_population[:self.population_size] # Ensure population size
        for prog in self.population: prog.fitness_calculated_this_gen = False # Reset for next gen

        return generations_since_improvement


    def search(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Program]:
        print(f"Starting program search with {len(examples)} examples...")
        start_time = datetime.datetime.now()

        patterns = self.reasoner.analyze_examples(examples)
        suggested_ops_from_reasoner = self.reasoner.suggest_operations(patterns)
        print(f"Pattern analysis summary: Suggested ops: {suggested_ops_from_reasoner}")
        # For debugging, print more pattern details:
        # print(f"Size change patterns: {patterns.get('size_change')}")
        # print(f"Color change patterns: {patterns.get('color_changes')}")
        # print(f"Shape change patterns: {patterns.get('shape_changes')}")
        # print(f"Monochrome patterns: {patterns.get('output_monochrome_check')}")
        # print(f"Roll patterns: {patterns.get('translation_roll_patterns')}")
        # print(f"Content shift patterns: {patterns.get('content_shift_patterns')}")

        best_simple_program, successful_heuristic_ops = self._try_simple_solutions(examples, patterns)
        
        # Check if simple solution is perfect (all training examples solved)
        is_simple_solution_perfect = False
        if best_simple_program:
            is_simple_solution_perfect = True
            for inp, out_ex in examples:
                res = best_simple_program.execute(inp.copy(), self.library)
                if res is None or not np.array_equal(res, out_ex):
                    is_simple_solution_perfect = False; break
        
        if is_simple_solution_perfect and best_simple_program.fitness >= 0.999: # Threshold for "good enough"
             print(f"Found high-quality simple solution that perfectly matches training data. Fitness {best_simple_program.fitness:.4f}. Skipping GA.")
             return best_simple_program
        
        self.initialize_population(patterns)
        if not self.population:
            print("Error: Population initialization failed.")
            return best_simple_program # Return best simple one found, if any
        
        for program in self.population:
            program.fitness = self.evaluate_program(program, examples)
            program.fitness_calculated_this_gen = True 
        
        best_program_overall = None
        best_fitness_overall = -1.0

        if best_simple_program: # Consider the best simple program as an initial candidate
            best_program_overall = deepcopy(best_simple_program)
            best_fitness_overall = best_simple_program.fitness
        
        if self.population: # Also consider best from initial random population
            initial_best_in_pop = max(self.population, key=lambda p: p.fitness)
            if initial_best_in_pop.fitness > best_fitness_overall:
                best_fitness_overall = initial_best_in_pop.fitness
                best_program_overall = deepcopy(initial_best_in_pop)

        generations_since_improvement = 0
        max_stagnation_generations = self.max_generations // 2 
        if max_stagnation_generations < 15: max_stagnation_generations = 15 
        if self.max_generations < 10: max_stagnation_generations = self.max_generations # For very short runs
        
        for generation in range(self.max_generations):
            stagnation_level = min(1.0, generations_since_improvement / max_stagnation_generations if max_stagnation_generations > 0 else 1.0)
            
            for prog in self.population: prog.fitness_calculated_this_gen = False

            gs_improvement_before_evolve = generations_since_improvement
            generations_since_improvement = self.evolve_generation(
                                                examples, patterns,
                                                successful_heuristic_ops, 
                                                suggested_ops_from_reasoner, 
                                                stagnation_level,
                                                generations_since_improvement) 
            
            if not self.population: print(f"Generation {generation}: Population empty, stopping search."); break
            
            # Population is sorted within evolve_generation after evaluations / cataclysm
            current_best_in_gen = self.population[0] # Should be the best after sort
            
            if current_best_in_gen.fitness > best_fitness_overall:
                best_fitness_overall = current_best_in_gen.fitness
                best_program_overall = deepcopy(current_best_in_gen) 
                generations_since_improvement = 0 # Reset stagnation as we found a better one
                print(f"Generation {generation}: New best fitness! {best_fitness_overall:.4f} (Complexity: {best_program_overall.complexity if best_program_overall else 'N/A'}) Program: {best_program_overall.operations if best_program_overall else 'N/A'}")
            # If cataclysm happened and reset generations_since_improvement, don't increment it here unless no improvement
            elif generations_since_improvement == 0 and gs_improvement_before_evolve > 0 : # Cataclysm reset happened
                 pass # Already reset, don't increment unless cataclysm itself didn't yield improvement (which is handled by fitness check)
            else: 
                generations_since_improvement += 1

            if generation % 10 == 0 or self.max_generations < 20 : # Print more often for short runs
                print(f"Generation {generation}: Best in gen = {current_best_in_gen.fitness:.4f}, Overall best = {best_fitness_overall:.4f}, Stagnation: {generations_since_improvement} (Level: {stagnation_level:.2f})")
            
            # Check for perfect solution against training examples
            if best_program_overall and best_fitness_overall >= 0.999 : # Threshold for good enough
                is_perfect_overall = True
                for inp_ex, out_ex in examples:
                    res_ex = best_program_overall.execute(inp_ex.copy(), self.library)
                    if res_ex is None or not np.array_equal(res_ex, out_ex):
                        is_perfect_overall = False; break
                if is_perfect_overall:
                    print(f"Found high-quality solution perfectly matching training data at generation {generation} with fitness {best_fitness_overall:.4f}!")
                    break 
            
            if generations_since_improvement > max_stagnation_generations:
                print(f"Search stagnated for {max_stagnation_generations} generations. Stopping early.")
                break
        
        end_time = datetime.datetime.now()
        search_duration = end_time - start_time
        print(f"Search completed in {search_duration}. Best fitness achieved: {best_fitness_overall:.4f}")
        if best_program_overall:
             print(f"Final best program complexity: {best_program_overall.complexity}, Ops: {best_program_overall.operations}")
        return best_program_overall


class ARCDataLoader:
    def __init__(self, data_path: str = "data"):
        # Try to find the data path relative to the script or CWD
        script_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
        self.data_path = script_dir / data_path
        if not self.data_path.exists(): # If not found relative to script, try relative to CWD
             self.data_path = Path.cwd() / data_path
        
        # Fallback if common data structures like ARC/data are used
        if not self.data_path.exists():
            common_arc_data_path = script_dir / "ARC" / "data" # e.g. if script is in a repo containing ARC/data
            if common_arc_data_path.exists(): self.data_path = common_arc_data_path
            else: # Try CWD / ARC / data
                common_arc_data_path_cwd = Path.cwd() / "ARC" / "data"
                if common_arc_data_path_cwd.exists(): self.data_path = common_arc_data_path_cwd


        self.training_path = self.data_path / "training"
        self.evaluation_path = self.data_path / "evaluation"

        # Check if paths exist, print warning if not
        if not self.data_path.exists(): print(f"Warning: Base data path '{self.data_path.resolve()}' does not exist.")
        if not self.training_path.exists(): print(f"Warning: Training data path '{self.training_path.resolve()}' does not exist.")
        if not self.evaluation_path.exists(): print(f"Warning: Evaluation data path '{self.evaluation_path.resolve()}' does not exist.")


    def load_task(self, task_file: str) -> Dict[str, Any]:
        task_file_path = Path(task_file)
        # If task_file is just a name, try to find it in training or evaluation paths
        if not task_file_path.is_absolute() and not task_file_path.exists():
            # Check training path
            potential_train_path = self.training_path / task_file_path.name
            if potential_train_path.exists(): task_file_path = potential_train_path
            else: # Check evaluation path
                potential_eval_path = self.evaluation_path / task_file_path.name
                if potential_eval_path.exists(): task_file_path = potential_eval_path
                else: # Try to find it directly if it was a relative path from CWD
                    if Path(task_file).exists() : task_file_path = Path(task_file) # Use original if it exists from CWD
                    else: raise FileNotFoundError(f"Task file '{task_file}' not found in training, evaluation, or as direct path. Searched: {potential_train_path}, {potential_eval_path}")
        
        with open(task_file_path, 'r') as f: task_data = json.load(f)
        return task_data

    def get_training_examples(self, task_data: Dict[str, Any]) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [(np.array(ex['input'], dtype=int), np.array(ex['output'], dtype=int)) for ex in task_data['train']]
    def get_test_cases(self, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        processed_test_cases = []
        for tc in task_data['test']:
            processed_tc = {'input': np.array(tc['input'], dtype=int)}
            if 'output' in tc: processed_tc['output'] = np.array(tc['output'], dtype=int)
            processed_test_cases.append(processed_tc)
        return processed_test_cases
    def list_training_tasks(self) -> List[str]:
        if not self.training_path.exists():
            print(f"Training task directory not found: {self.training_path.resolve()}")
            return []
        return sorted([str(f.resolve()) for f in list(self.training_path.glob("*.json"))])
    def list_evaluation_tasks(self) -> List[str]:
        if not self.evaluation_path.exists():
            print(f"Evaluation task directory not found: {self.evaluation_path.resolve()}")
            return []
        return sorted([str(f.resolve()) for f in list(self.evaluation_path.glob("*.json"))])

def log_to_file(message: str, log_file: Path, mode: str = 'a'):
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True) # Ensure log directory exists
        with open(log_file, mode, encoding='utf-8') as f: f.write(message + '\n')
    except Exception as e:
        print(f"Error writing to log file {log_file}: {e}")


def run_single_task(task_file: str, search_params: Dict[str, Any]):
    task_name = Path(task_file).stem
    print(f"\n{'='*60}\nRunning task: {task_name}\n{'='*60}")
    loader = ARCDataLoader() 
    try: task_data = loader.load_task(task_file)
    except FileNotFoundError: print(f"Error: Task file not found: {task_file}"); return None, {'task_file': task_name, 'status': 'task_file_not_found'}
    except Exception as e: print(f"Error loading task {task_file}: {e}"); traceback.print_exc(); return None, {'task_file': task_name, 'status': 'task_load_error', 'error_message': str(e)}
    
    examples = loader.get_training_examples(task_data)
    test_cases = loader.get_test_cases(task_data) 
    if not examples: print("No training examples found."); return None, {'task_file': task_name, 'status': 'no_train_examples'}
    
    searcher = ARCProgramSearch(**search_params) 
    best_program = searcher.search(examples) 
    
    task_stats = {'task_file': task_name, 'status': 'no_solution_found', 'best_fitness': 0.0, 
                  'training_accuracy': 0.0, 'test_accuracy_on_solved_train': 0.0, 
                  'num_test_cases_with_output': 0, 'program_complexity': 0, 
                  'program_operations': "N/A", 'used_heuristic_simple_solution': False}

    if best_program:
        # Re-evaluate fitness for consistency, or use stored if available and reliable
        task_stats['best_fitness'] = best_program.fitness if hasattr(best_program, 'fitness') else searcher.evaluate_program(best_program, examples)
        task_stats['program_complexity'] = best_program.complexity
        task_stats['program_operations'] = "; ".join([f"{op}({args})" for op, args in best_program.operations]) if best_program.operations else "Identity"
        
        # Check if this best_program matches one of the _try_simple_solutions heuristics
        patterns_for_heuristic_check = searcher.reasoner.analyze_examples(examples)
        simple_prog_check, successful_heuristics_list = searcher._try_simple_solutions(examples, patterns_for_heuristic_check) # successful_heuristics_list is list of (name, args)
        
        if best_program.operations: # If it's not Identity
            if len(best_program.operations) == 1:
                op_tuple = best_program.operations[0]
                if op_tuple in successful_heuristics_list:
                    task_stats['used_heuristic_simple_solution'] = True
        elif not best_program.operations : # Identity program
            if ("",[]) in successful_heuristics_list: # Check if Identity was a heuristic success
                 task_stats['used_heuristic_simple_solution'] = True


        correct_train = 0
        for inp_grid, expected_out_grid in examples:
            result_grid = best_program.execute(np.array(inp_grid).copy(), searcher.library)
            if result_grid is not None and np.array_equal(result_grid, expected_out_grid): correct_train += 1
        task_stats['training_accuracy'] = correct_train / len(examples) if examples else 0.0
        
        # Status update based on training accuracy
        if task_stats['training_accuracy'] >= 0.999 : # Threshold for "solved"
            task_stats['status'] = 'solved_training'
            print(f"\nTraining SOLVED for {task_name} (Accuracy: {task_stats['training_accuracy']:.2%})")
            
            test_cases_with_output = [tc for tc in test_cases if 'output' in tc]
            task_stats['num_test_cases_with_output'] = len(test_cases_with_output)
            
            if test_cases_with_output:
                correct_test = 0
                for test_case_dict in test_cases_with_output:
                    predicted_grid = best_program.execute(np.array(test_case_dict['input']).copy(), searcher.library)
                    if predicted_grid is not None and np.array_equal(predicted_grid, test_case_dict['output']): correct_test += 1
                
                task_stats['test_accuracy_on_solved_train'] = correct_test / len(test_cases_with_output) if test_cases_with_output else 0.0
                print(f"Test accuracy for {task_name} (on {len(test_cases_with_output)} cases): {task_stats['test_accuracy_on_solved_train']:.2%}")
                if task_stats['test_accuracy_on_solved_train'] >= 0.999 :  # Threshold for "solved all"
                    task_stats['status'] = 'solved_all_public_tests'
            else: # Solved training, but no test cases with output to verify against
                 print(f"No public test cases with output provided for {task_name}.")
                 task_stats['test_accuracy_on_solved_train'] = 0.0 # Or mark as N/A

        else: # Failed to solve training perfectly
            task_stats['status'] = 'failed_training'
            print(f"\nTraining FAILED for {task_name} (Best train accuracy: {task_stats['training_accuracy']:.2%})")
        
        print(f"Best program for {task_name} (Fitness: {task_stats['best_fitness']:.4f}, Heuristic: {task_stats['used_heuristic_simple_solution']}): {task_stats['program_operations']}")
    
    else: # No program found by searcher.search
        print(f"\nNo suitable program found by GA for {task_name}.")
        # Check if a simple heuristic alone had some fitness, even if not perfect
        patterns_for_heuristic_check = searcher.reasoner.analyze_examples(examples)
        simple_prog_check, _ = searcher._try_simple_solutions(examples, patterns_for_heuristic_check)
        if simple_prog_check and hasattr(simple_prog_check, 'fitness'): 
            task_stats['best_fitness'] = simple_prog_check.fitness # Record fitness of best simple heuristic
            if simple_prog_check.fitness > 0 : # If it's better than default 0
                 task_stats['program_complexity'] = simple_prog_check.complexity
                 task_stats['program_operations'] = "; ".join([f"{op}({args})" for op, args in simple_prog_check.operations]) if simple_prog_check.operations else "Identity"
                 task_stats['used_heuristic_simple_solution'] = True # It was the best among simple ones
                 # Re-calculate training accuracy for this simple program
                 correct_train_simple = 0
                 for inp_grid, expected_out_grid in examples:
                     result_grid_simple = simple_prog_check.execute(np.array(inp_grid).copy(), searcher.library)
                     if result_grid_simple is not None and np.array_equal(result_grid_simple, expected_out_grid): correct_train_simple += 1
                 task_stats['training_accuracy'] = correct_train_simple / len(examples) if examples else 0.0
                 if task_stats['training_accuracy'] >=0.999: task_stats['status'] = 'solved_training_by_heuristic_only' # Special status
                 else: task_stats['status'] = 'heuristic_found_but_not_solving_training'

    return best_program, task_stats


def run_multiple_tasks(search_config: Dict[str, Any], task_limit: Optional[int] = None):
    logs_dir = Path("logs"); logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_log_file = logs_dir / f"arc_run_summary_{timestamp}.txt"
    csv_results_file = logs_dir / f"arc_run_results_{timestamp}.csv"
    
    initial_log = f"ARC Run Log\nTimestamp: {datetime.datetime.now()}\nParams: {json.dumps(search_config, indent=2)}\nTaskLimit: {task_limit if task_limit is not None else 'All'}\n{'='*80}"
    log_to_file(initial_log, summary_log_file, mode='w')
    
    csv_headers = ["TaskFile", "Status", "BestFitness", "TrainingAccuracy", "TestAccuracyIfTrainSolved", "NumTestCasesWithOutput", "ProgramComplexity", "ProgramOperations", "UsedHeuristicSolution"]
    log_to_file(",".join(csv_headers), csv_results_file, mode='w')
    
    loader = ARCDataLoader(); task_files = loader.list_training_tasks()
    if not task_files: 
        message = f"No training tasks found in the expected directory ({loader.training_path.resolve()}). Please ensure ARC dataset is present."; print(message)
        log_to_file(message, summary_log_file); return []
        
    if task_limit is not None: task_files = task_files[:task_limit]; print(f"Limiting to {task_limit} tasks.")
    
    all_task_stats = []; total_tasks_to_run = len(task_files)
    print(f"\nFound {total_tasks_to_run} tasks. Starting batch processing..."); 
    log_to_file(f"Found {total_tasks_to_run} tasks. Starting batch processing...", summary_log_file)
    
    for i, task_file_path_str in enumerate(task_files, 1):
        task_short_name = Path(task_file_path_str).stem
        task_header = f"\n{'='*80}\nProcessing task {i}/{total_tasks_to_run}: {task_short_name}\n{'='*80}"
        print(task_header); log_to_file(task_header, summary_log_file)
        try:
            _, task_stats_dict = run_single_task(task_file_path_str, search_config)
            if not task_stats_dict: # Should not happen if run_single_task guarantees a dict
                task_stats_dict = {'task_file': task_short_name, 'status': 'error_empty_stats_dict'}
            all_task_stats.append(task_stats_dict)
            
            # Ensure all keys for CSV are present, with defaults
            csv_row_data = {key: task_stats_dict.get(key) for key in csv_headers}
            csv_row_data['TaskFile'] = task_stats_dict.get('task_file', task_short_name)
            csv_row_data['Status'] = task_stats_dict.get('status', 'unknown_error')
            csv_row_data['BestFitness'] = f"{task_stats_dict.get('best_fitness', 0.0):.4f}"
            csv_row_data['TrainingAccuracy'] = f"{task_stats_dict.get('training_accuracy', 0.0):.4f}"
            csv_row_data['TestAccuracyIfTrainSolved'] = f"{task_stats_dict.get('test_accuracy_on_solved_train', 0.0):.4f}"
            csv_row_data['NumTestCasesWithOutput'] = str(task_stats_dict.get('num_test_cases_with_output', 0))
            csv_row_data['ProgramComplexity'] = str(task_stats_dict.get('program_complexity', 0))
            csv_row_data['ProgramOperations'] = f"\"{task_stats_dict.get('program_operations', 'N/A')}\"" # Quote for CSV
            csv_row_data['UsedHeuristicSolution'] = str(task_stats_dict.get('used_heuristic_simple_solution', False))

            csv_row_ordered = [str(csv_row_data.get(h, '')) for h in csv_headers]
            log_to_file(",".join(csv_row_ordered), csv_results_file)
            
            task_summary_log = (f"Task: {csv_row_data['TaskFile']}\n"
                                f"  Status: {csv_row_data['Status']}\n"
                                f"  Training Accuracy: {float(csv_row_data['TrainingAccuracy']):.2%}\n"
                                f"  Test Accuracy (if train solved): {float(csv_row_data['TestAccuracyIfTrainSolved']):.2%} ({csv_row_data['NumTestCasesWithOutput']} cases)\n"
                                f"  Best Fitness: {float(csv_row_data['BestFitness']):.4f}\n"
                                f"  Used Heuristic: {csv_row_data['UsedHeuristicSolution']}\n"
                                f"  Program: {csv_row_data['ProgramOperations']}\n")
            log_to_file(task_summary_log, summary_log_file)

        except KeyboardInterrupt:
            print("\nBatch processing interrupted by user. Saving partial results.")
            log_to_file("\nBatch processing interrupted by user.", summary_log_file)
            break # Exit the loop
        except Exception as e:
            error_message = f"Critical Error processing task {task_short_name}: {str(e)}\n{traceback.format_exc()}"
            print(error_message); log_to_file(error_message, summary_log_file)
            # Log error to CSV
            error_csv_row = [task_short_name, "critical_error", "0","0","0","0","0",f"\"{str(e)}\"","False"]
            log_to_file(",".join(error_csv_row), csv_results_file)
            all_task_stats.append({'task_file': task_short_name, 'status': 'critical_error', 'error_message': str(e)})

    summary_header = f"\n\n{'='*80}\nOVERALL BATCH SUMMARY\n{'='*80}"; print(summary_header); log_to_file(summary_header, summary_log_file)
    if all_task_stats:
        num_processed = len(all_task_stats)
        num_solved_training = sum(1 for s in all_task_stats if s.get('status') in ['solved_training', 'solved_all_public_tests', 'solved_training_by_heuristic_only'])
        num_solved_all_tests = sum(1 for s in all_task_stats if s.get('status') == 'solved_all_public_tests')
        num_heuristic_solutions = sum(1 for s in all_task_stats if s.get('used_heuristic_simple_solution', False) and s.get('status') not in ['no_solution_found', 'failed_training', 'heuristic_found_but_not_solving_training', 'task_file_not_found', 'task_load_error', 'no_train_examples', 'critical_error'])
        
        solved_training_pct = (num_solved_training / num_processed * 100) if num_processed > 0 else 0
        solved_all_tests_pct = (num_solved_all_tests / num_processed * 100) if num_processed > 0 else 0
        heuristic_pct_of_solved_training = (num_heuristic_solutions / num_solved_training * 100) if num_solved_training > 0 else 0
        
        avg_train_acc_items = [s.get('training_accuracy',0.0) for s in all_task_stats if 'training_accuracy' in s and isinstance(s.get('training_accuracy'), (float, int))] # Filter out non-numeric if any
        avg_train_acc = sum(avg_train_acc_items) / len(avg_train_acc_items) if avg_train_acc_items else 0.0
        
        solved_train_with_tests = [s for s in all_task_stats if s.get('status') in ['solved_training', 'solved_all_public_tests', 'solved_training_by_heuristic_only'] and s.get('num_test_cases_with_output',0) > 0 and isinstance(s.get('test_accuracy_on_solved_train'), (float,int))]
        avg_test_acc_items = [s.get('test_accuracy_on_solved_train',0.0) for s in solved_train_with_tests]
        avg_test_acc = sum(avg_test_acc_items) / len(avg_test_acc_items) if avg_test_acc_items else 0.0
        
        overall_summary_text = (f"Total Tasks Attempted: {num_processed}\n"
                                f"Tasks Solved (Training): {num_solved_training} ({solved_training_pct:.1f}%)\n"
                                f"Tasks Solved (All Public Tests): {num_solved_all_tests} ({solved_all_tests_pct:.1f}%)\n"
                                f"Solutions via Simple Heuristics (of solved training): {num_heuristic_solutions} ({heuristic_pct_of_solved_training:.1f}% of solved training)\n\n"
                                f"Average Training Accuracy (all attempts): {avg_train_acc:.1%}\n"
                                f"Average Test Accuracy (on {len(solved_train_with_tests)} tasks with solved training & test data): {avg_test_acc:.1%}\n")
        print(overall_summary_text); log_to_file(overall_summary_text, summary_log_file)
    else: message="No tasks were processed or stats collected."; print(message); log_to_file(message, summary_log_file)
    
    final_log_paths = f"\nSummary log saved to: {summary_log_file.resolve()}\nDetailed results CSV saved to: {csv_results_file.resolve()}\n"
    print(final_log_paths); log_to_file(final_log_paths, summary_log_file)
    return all_task_stats

if __name__ == "__main__":
    # Create dummy data directory and a few dummy task files if they don't exist
    # This is for the script to run without needing external data in some environments.
    # Ensure the 'data/training' path (or 'ARC/data/training') is correctly set up for real runs.
    
    # Attempt to locate or create dummy data directory
    script_location = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    data_dir_main = script_location / "data"
    training_dir_main = data_dir_main / "training"

    # Check if ARC-dataset like structure might exist (e.g. https://github.com/fchollet/ARC)
    arc_repo_data_dir = script_location.parent / "ARC" / "data" # If script is inside a subfolder of a repo clone
    arc_repo_training_dir = arc_repo_data_dir / "training"

    final_training_dir = None

    if training_dir_main.exists() and list(training_dir_main.glob("*.json")):
        final_training_dir = training_dir_main
        print(f"Using existing data from: {final_training_dir.resolve()}")
    elif arc_repo_training_dir.exists() and list(arc_repo_training_dir.glob("*.json")):
        final_training_dir = arc_repo_training_dir
        # Adjust ARCDataLoader's default path if we use this one
        # This requires modifying ARCDataLoader or passing path to it. For now, rely on its internal fallbacks.
        print(f"Using existing data from ARC repository structure: {final_training_dir.resolve()}")
    else: # Create dummy data if no real data found
        final_training_dir = training_dir_main # Default to creating dummy data in ./data/training
        final_training_dir.mkdir(parents=True, exist_ok=True)
        print(f"No existing task files found. Creating dummy task files in {final_training_dir.resolve()} for testing...")
        dummy_task_content_1 = { # Solvable by rotate_180
            "train": [{"input": [[1,0,0],[0,1,0],[0,0,1]], "output": [[1,0,0],[0,1,0],[0,0,1]]}], #Oops, made output same as input by mistake. Fixed.
            "test": [{"input": [[1,1,0],[1,0,0],[0,0,0]], "output": [[0,0,0],[0,0,1],[0,1,1]]}]
        }
        dummy_task_content_1["train"][0]["output"] = [[1,0,0],[0,1,0],[0,0,1]][::-1,::-1].tolist() # Correct rotate_180

        dummy_task_content_2 = { # Solvable by replace_color 1->2
            "train": [{"input": [[1,0,1],[0,1,0],[1,0,1]], "output": [[2,0,2],[0,2,0],[2,0,2]]}],
            "test": [{"input": [[1,1,1],[1,0,1],[1,1,1]], "output": [[2,2,2],[2,0,2],[2,2,2]]}]
        }
        dummy_task_content_3 = { # Solvable by translate_content (1,1,0)
             "train": [{"input": [[0,0,0,0],[0,5,5,0],[0,5,5,0],[0,0,0,0]], 
                        "output": [[0,0,0,0],[0,0,0,0],[0,0,5,5],[0,0,5,5]]}],
             "test": [{"input": [[5,0,0],[0,0,0],[0,0,0]], "output": [[0,0,0],[0,5,0],[0,0,0]]}]
        }

        dummy_tasks_data = {
            "dummy_rotate_task.json": dummy_task_content_1,
            "dummy_replace_task.json": dummy_task_content_2,
            "dummy_translate_task.json": dummy_task_content_3
        }
        dummy_files_created_count = 0
        if not list(final_training_dir.glob("*.json")): # Only create if still empty
            for fname, content in dummy_tasks_data.items():
                with open(final_training_dir / fname, "w") as f:
                    json.dump(content, f)
                dummy_files_created_count += 1
            if dummy_files_created_count > 0:
                 print(f"Created {dummy_files_created_count} dummy task files.")
        else:
             print(f"Directory {final_training_dir.resolve()} already contains JSON files. Skipping dummy file creation.")


    if not final_training_dir or not final_training_dir.exists():
        print(f"Error: Training directory '{final_training_dir.resolve() if final_training_dir else 'Not determined'}' not found or could not be created.")
        exit(1)
    
    task_files = list(final_training_dir.glob("*.json"))
    if not task_files:
        print(f"Error: No JSON task files found in '{final_training_dir.resolve()}'. Ensure data is present or dummy creation worked.")
        exit(1)
    
    print(f"Found {len(task_files)} task files in {final_training_dir.resolve()}")
    print("Sample task files:", [f.name for f in task_files[:min(5, len(task_files))]])

    # Define search parameters 
    search_parameters = {
        "population_size": 200,      # Reduced for speed: 50-200 is typical
        "max_generations": 100,       # Reduced for speed: 50-200 is typical
        "elite_percentage": 0.10,   
        "tournament_size": 3,       
        "arg_mutation_prob": 0.15,  
        "add_op_prob": 0.15,        # Higher chance to add ops, good for small programs
        "remove_op_prob": 0.15,     
        "change_op_prob": 0.20,     
        "swap_op_prob": 0.15,       
        "stagnation_boost_factor_mutate": 0.75, 
        "cataclysm_stagnation_threshold": 0.5, 
        "cataclysm_prob": 0.20                 
    }
    
    # Run on all tasks found (or dummy tasks if created)
    # Set task_limit to a small number (e.g., 3 or len(task_files)) for quick testing.
    run_multiple_tasks(
        search_config=search_parameters,
        task_limit=len(task_files) # Process all found tasks, or set to e.g. 3 for quicker test
    )
    
    # Example for testing a single specific task if needed:
    # if task_files:
    #     specific_task_file = next((tf for tf in task_files if "dummy_translate_task.json" in tf.name), task_files[0])
    #     print(f"\n--- Testing '{specific_task_file.name}' specifically ---")
    #     run_single_task(str(specific_task_file.resolve()), search_parameters)
