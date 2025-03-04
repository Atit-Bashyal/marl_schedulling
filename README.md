# marl_schedulling

This repository contains a framework for modeling and simulating industrial processes using XML-based configuration files. The system supports equipment modeling, process simulation, and reinforcement learning for multi-agent scheduling.

## 🏭 Industrial Process Simulation

- **XML Configuration**: The industrial process is defined in an XML file (`config.xml`).
- **Equipment Modeling (`equipments.py`)**: Defines different types of industrial equipment as Python classes.
- **Factory Simulation (`factory_*.py`)**: 
  - Reads the XML file.
  - Instantiates equipment based on the XML definition.
  - Simulates the process through various functions.

## 🤖 Multi-Agent Reinforcement Learning (MARL)

- **MARL Environment (`marl_*.py`)**: 
  - Uses `factory_*.py` to simulate the factory.
  - Implements a multi-agent RL environment.
- **Training & Testing**:
  - `training_*.py`: Trains MARL agents using different RL algorithms.
  - `testing_*.py`: Evaluates trained models.

## 🚀 Getting Started

### 1️⃣ Install Dependencies  
```sh
pip install -r requirements.txt
```
### 2️⃣ Train 

```sh
python training_*.py
```
`training_*.py` file depends on the model type

### 3️⃣ Test

```sh
python test_*.py
```
