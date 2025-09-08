# Arkanoid Atom

Arkanoid Atom analyzes the behavior of graphical elements in a simple Arkanoid game using heuristics and logic-based methods. It avoids deep learning, relying on interpretable and rule-based mechanisms. This project provides insights into object interactions, properties, and changes over time.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Glossary](#glossary)

## Features

- Classic brick-breaking gameplay with event-driven analysis.
- Logic-based detection of object appearance, disappearance, and changes.
- Modular design for extensibility and experimentation.

## Installation

### Prerequisites

- Python 3.8+
- Pygame library

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/BlankTo/arkanoid_atom.git
   ```

2. Navigate to the project directory:

   ```bash
   cd arkanoid_atom
   ```

3. Install the required dependencies:

   ```bash
   pip install pygame
   ```

4. Run the game and analysis:

   ```bash
   python arkanoid.py  # Play the game and generate logs
   python main.py       # Analyze the generated logs
   ```

## Usage

The project consists of two main components:

1. **Gameplay**: Run `arkanoid.py` to play the game and generate logs of object interactions.
2. **Analysis**: Use `main.py` to analyze the generated logs and gain insights into object behaviors.

### Commands in Gameplay

- **Left Arrow Key**: Move the paddle to the left.
- **Right Arrow Key**: Move the paddle to the right.
- **Up Arrow Key**: Speed up the game.
- **Down Arrow Key**: Slow down the game.
- **q**: Exit the game (logs are still recorded).

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Glossary

### Property
A **Property** is a characteristic of a graphical element, such as its position (e.g., `Pos_x`, `Pos_y`) or shape.

### Patch
A **Patch** represents a graphical element in the game. Patches are tracked across frames but are treated as anonymous graphical elements during analysis. Identifying which real-world elements they belong to is a central part of the project.

### Object
An **Object** is an inferred reconstruction of a "real" game entity. For example, if there is a ball moving in the game, the project attempts to determine which patches correspond to the ball and reconstruct its expected behaviors and interactions.

### Individual
An **Individual** is a collection of objects grouped together as a possible interpretation of the game state. If multiple valid sequences of patches could correspond to an object, the individual is split into separate possibilities, each representing a distinct hypothesis of object behavior.

---
