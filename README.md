# ARC Task Solver using Genetic Programming

This project implements a system to solve tasks from the Abstraction and Reasoning Corpus (ARC) using a Genetic Programming (GP) approach. It attempts to find a sequence of primitive grid operations that transform input grids into their corresponding output grids based on a set of training examples.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Core Components](#core-components)
- [How it Works](#how-it-works)
- [Primitives Library](#primitives-library)
- [Directory Structure and Setup](#directory-structure-and-setup)
- [Usage](#usage)
- [Output](#output)
- [Future Work](#future-work)

## Overview

The Abstraction and Reasoning Corpus (ARC) is a challenging benchmark for artificial intelligence. Tasks require identifying abstract patterns and applying them to new inputs. This project uses a genetic programming framework where:
-   **Individuals** are programs, represented as sequences of primitive operations.
-   **Fitness** is determined by how well a program transforms training input grids into the expected output grids, considering pixel similarity, Intersection over Union (IoU) for object shapes, and program complexity.
-   **Evolution** involves selection, crossover, and mutation to generate new, potentially better programs over generations.
-   An **Inductive Reasoner** analyzes training examples to extract patterns and guide the GP search by suggesting relevant operations and informing argument generation for these operations.

## Features

-   **Genetic Programming:** Evolves programs (sequences of operations) to solve ARC tasks.
-   **Rich Primitive Library:** A comprehensive set of over 20 elementary grid operations (e.g., color manipulation, geometric transformations, object extraction, cropping, padding, translation).
-   **Inductive Reasoner:** Analyzes I/O examples to detect patterns like:
    -   Size and shape changes (rotation, flip, transpose).
    -   Color transformations and mappings.
    -   Output monochrome properties.
    -   Grid content translation, rolling, and shifting.
    -   Suggests relevant operations based on detected patterns.
-   **Intelligent Argument Generation:** Primitives' arguments are often generated based on patterns observed in the task data (e.g., using colors present in input/output, target sizes from examples).
-   **Heuristic Solver:** Attempts to find simple, single-operation solutions based on strong heuristic cues before initiating the full GP search.
-   **Stagnation Handling:** Implements strategies like mutation rate boosting and "cataclysmic mutation" (partial re-initialization of the population) to overcome search stagnation.
-   **Batch Processing:** Can run on multiple ARC tasks sequentially.
-   **Logging:** Generates detailed logs, including a summary of the run and a CSV file with results for each task.
-   **Configurable:** GP parameters (population size, generation count, mutation rates, etc.) can be easily adjusted.

## Core Components

-   **`ARCGrid`**: A wrapper for ARC grids (NumPy arrays) providing utility methods for common grid analyses (e.g., finding objects, connected components, colors, bounding boxes).
-   **`PrimitiveOperation` & `PrimitiveLibrary`**: Defines and manages the set of basic operations that can be applied to grids.
-   **`Program`**: Represents a candidate solution as a list of primitive operations. Includes methods for execution, mutation, and crossover.
-   **`InductiveReasoner`**: Analyzes training examples to extract abstract patterns and suggest relevant operations to the GP.
-   **`ARCProgramSearch`**: Implements the genetic algorithm, including population initialization, fitness evaluation, selection, and evolutionary operators.
-   **`ARCDataLoader`**: Handles loading of ARC task data from JSON files.

## How it Works

1.  **Load Task:** An ARC task (JSON file) is loaded, providing training examples and test cases.
2.  **Inductive Reasoning:** The `InductiveReasoner` analyzes the `(input, output)` pairs from the training examples to identify patterns related to size, color, shape, and position. It generates a summary of these patterns and suggests potentially useful primitive operations.
3.  **Heuristic Solution Attempt:** The system first tries to solve the task using simple, single-operation programs based on strong heuristics derived from the reasoner's analysis (e.g., if all examples show a consistent rotation, try the rotation primitive).
4.  **Genetic Programming (if no simple solution):**
    *   **Initialization:** A population of random programs is created. Programs are sequences of operations from the `PrimitiveLibrary`. Operation arguments are generated using guidance from the `InductiveReasoner`.
    *   **Evaluation:** Each program in the population is executed on all training examples. Its fitness is calculated based on:
        *   Pixel-wise similarity between the program's output and the expected output.
        *   Intersection over Union (IoU) of non-background content.
        *   A penalty for program complexity (length).
    *   **Selection:** Programs are selected to be parents for the next generation, typically using tournament selection (fitter programs have a higher chance).
    *   **Evolution:**
        *   **Crossover:** Selected parent programs are combined to create offspring.
        *   **Mutation:** Offspring (and sometimes other individuals) undergo random changes:
            *   Modifying arguments of an operation.
            *   Adding a new operation.
            *   Removing an operation.
            *   Changing an operation type.
            *   Swapping the order of operations.
        *   Mutation probabilities can be dynamically adjusted based on search stagnation.
    *   **Replacement:** The new generation replaces the old, often keeping a percentage of the best individuals (elitism).
5.  **Termination:** The GP loop continues for a maximum number of generations or until a program is found that perfectly solves all training examples with a high fitness score.
6.  **Logging:** Results, including the best program found, its fitness, and performance on training/test sets, are logged.

## Primitives Library

The system uses a variety of primitive operations, including but not limited to:

-   **Color Manipulation:** `fill_color`, `replace_color`, `isolate_color`
-   **Geometric Transformations:** `rotate_90`, `rotate_180`, `flip_horizontal`, `flip_vertical`, `transpose`
-   **Positional Adjustments:** `roll_rows`, `roll_cols`, `translate_content`
-   **Object-based Operations:**
    -   `extract_largest_object_by_color`
    -   `color_object_by_rank`, `delete_object_by_rank`
    -   `extract_mask_of_object_by_rank`
    -   `crop_to_object_by_rank`
    -   `copy_object_by_rank_and_paste`
-   **Structural Operations:** `crop_to_content`, `pad_to_size`, `resize_to_match`
-   **Counting/Querying:** `count_colors`, `get_most_common_color`, `count_all_objects`

## Directory Structure and Setup

1.  **ARC Data:**
    The script expects ARC task files (JSON format) to be located in a specific directory structure. It will search in the following order:
    *   `./data/training/` and `./data/evaluation/` (relative to the script's location)
    *   `./data/training/` and `./data/evaluation/` (relative to the current working directory)
    *   `./ARC/data/training/` and `./ARC/data/evaluation/` (relative to the script's location, common for ARC repository clones)
    *   `./ARC/data/training/` and `./ARC/data/evaluation/` (relative to the current working directory)

    If no task files are found, the script will automatically create a few dummy task files in `./data/training/` for demonstration and testing purposes.

2.  **Dependencies:**
    -   Python 3.x
    -   NumPy (`import numpy as np`)

    Other standard Python libraries like `json`, `os`, `glob`, `random`, `datetime`, `copy`, `itertools`, `traceback`, `pathlib` are also used.

## Usage

1.  **Ensure Data:** Place your ARC task JSON files in one of the directories mentioned above (e.g., `data/training/`).
2.  **Run Script:** Execute the Python script from your terminal:
    ```bash
    python arcsolver.py
    ```

3.  **Configuration:**
    The main execution logic is within the `if __name__ == "__main__":` block. Key GP parameters can be adjusted directly in the `search_parameters` dictionary in this block:
    ```python
    search_parameters = {
        "population_size": 200,
        "max_generations": 100,
        "elite_percentage": 0.13,
        # ... and other parameters
    }
    ```
    You can also control the maximum number of tasks to process using the `task_limit` argument in `run_multiple_tasks`.

## Output

-   **Console Output:** The script prints progress information, including:
    -   Task being processed.
    -   Summary of pattern analysis.
    -   Results of heuristic solution attempts.
    -   Per-generation best fitness and overall best fitness during GP.
    -   Final best program found for each task.
    -   Overall batch summary statistics.
-   **Log Files:** Generated in a `logs/` directory (created if it doesn't exist):
    -   `arc_run_summary_[timestamp].txt`: A human-readable summary of the entire run, including parameters, per-task results, and overall statistics.
    -   `arc_run_results_[timestamp].csv`: A CSV file containing detailed results for each task, suitable for programmatic analysis. Columns include task file, status, fitness, accuracies, program complexity, and the program itself.

## Future Work

-   **Enhanced Primitives:** Introduce more complex or domain-specific primitives.
-   **Sophisticated Reasoning:** Improve the `InductiveReasoner` to detect more subtle patterns or relationships between examples.
-   **Hierarchical Solutions:** Explore ways to evolve programs that can call other evolved sub-programs or macros.
-   **Adaptive Parameter Control:** Dynamically adjust GP parameters during the run based on search progress.
-   **Type System for Primitives/Arguments:** Implement a more formal type system for operation arguments to ensure validity and potentially guide generation.
-   **Visualization Tools:** Add tools to visualize grid transformations performed by programs.
