# PyTorch Template

A comprehensive PyTorch project template with Docker support, configuration management, and extensible training framework.

## Features

- 🐳 Docker support with CPU/GPU configurations
- ⚡ Fast dependency management with uv
- 🔧 Hydra-based configuration system
- 📊 Built-in training extensions (model saving, history tracking, plotting)
- 📝 Jupyter notebook integration
- 🧪 Comprehensive testing framework

## Environment Setup

### Prerequisites

- Docker and Docker Compose (recommended)
- Python 3.8+ and uv (for native installation)
- NVIDIA Docker runtime (for GPU support)

### Docker Setup (Recommended)

This project supports multiple Docker configurations:

#### 1. CPU Version (Default)
```bash
# Clone the repository
git clone <repository-url>
cd pytorch-template

# Build and start container
docker compose build
docker compose up -d

# Enter the container
docker compose exec core /bin/bash

# Train a model
uv run python scripts/train.py
```

#### 2. GPU Version
```bash
# Enable GPU support
echo "COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml" > ./docker/.env

# Build and start container
docker compose build
docker compose up -d

# Enter the container
docker compose exec core /bin/bash

# Verify GPU access
nvidia-smi
```

#### 3. Rootful Version
```bash
# Copy environment configuration
cp docker/env.example docker/.env

# Start container with root privileges
docker compose up -d
docker compose exec core /bin/bash
```

### Native Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies based on your system
uv sync --extra cpu     # CPU-only PyTorch
uv sync --extra cu124   # CUDA 12.4
uv sync --extra cu128   # CUDA 12.8

# Verify installation
uv run python scripts/installation_verification.py
```

## Quick Start

Once your environment is set up:

```bash
# Train with default configuration
uv run python scripts/train.py

# Train with specific config
uv run python scripts/train.py --config-path=conf --config-name=config.yaml

# Test trained models
uv run python scripts/test.py

# Test with specific history directory
uv run python scripts/test.py --history_dir outputs/train/history

# Clean up untrained models
uv run python scripts/test.py --remove_untrained_id
```

## Jupyter Development

### Docker Environment
```bash
# Start Jupyter server (inside Docker container)
docker compose exec core ./docker/run_jupyter.sh
```

### Native Environment
```bash
# Start Jupyter with uv
uv run --with jupyter jupyter notebook
```

When using Jupyter, select the kernel named after your project name.

## Project Structure

```
├── src/                    # Source code
│   ├── data_pipeline.py    # Data loading and caching
│   ├── model.py           # Model definitions
│   ├── trainer.py         # Training framework
│   ├── extension.py       # Training extensions/callbacks
│   ├── train_id.py        # Unique ID generation
│   └── util.py           # Utilities
├── scripts/               # Executable scripts
│   ├── train.py          # Main training script
│   ├── test.py           # Model testing script
│   └── installation_verification.py
├── conf/                  # Configuration files
│   ├── config.yaml       # Main config
│   ├── dataset/          # Dataset configs
│   ├── model/            # Model configs
│   ├── optimizer/        # Optimizer configs
│   └── lr_scheduler/     # Scheduler configs
├── notebooks/            # Jupyter notebooks
│   └── local/           # Local development notebooks
├── outputs/              # Training outputs
│   └── train/history/   # Training history by ID
├── docker/               # Docker configuration
│   ├── rootless/        # Rootless Docker setup
│   ├── rootful/         # Rootful Docker setup
│   └── run_jupyter.sh   # Jupyter startup script
└── tests/               # Test files
```

## Configuration System

This template uses Hydra for configuration management:

- **Hierarchical configs**: Organize settings by component (dataset, model, optimizer, etc.)
- **Config composition**: Easily combine different configurations
- **Unique training IDs**: Generated from config hashes for reproducibility
- **Override support**: Command-line parameter overrides

## Core Components

### Data Pipeline
- Custom data loading with caching capabilities
- Supports both static and dynamic transforms
- Efficient tensor-numpy conversion for caching

### Models
- `SimpleCNN`: Basic CNN architecture with lazy layers
- `ResNet`: Imported ResNet implementations
- Extensible model registry

### Training Framework
- Abstract base class for consistent trainer implementation
- Built-in validation and extension support
- Flexible evaluator system

### Extensions
- `ModelSaver`: Save best models based on triggers
- `HistorySaver`: Track training history
- `LearningCurvePlotter`: Generate learning curves
- Trigger system for flexible callbacks

## Development Workflow

1. **Prototyping**: Create notebooks in `notebooks/local/`
2. **Development**: Add source files to `src/`
3. **Testing**: Add test codes to `tests/`
4. **Evaluation**: Add script files to `scripts/`

## Customization

### Project Configuration
1. Update `pyproject.toml` with your project details (name, authors, etc.)
2. Modify `docker-compose.yml` variables as needed
3. Customize Docker files in `docker/rootless/` or `docker/rootful/`

### Dependencies
```bash
# Add new dependencies
uv add <library-name>

# Add development dependencies
uv add --dev <library-name>

# Update dependencies
uv sync
```

## Environment Variables

- `PROJECT_NAME`: Used in paths throughout the codebase
- `JUPYTER_IP`: Jupyter server IP (default: 0.0.0.0)
- `JUPYTER_PORT`: Jupyter server port (default: 8888)
- `JUPYTER_NOTEBOOK_DIR`: Notebook directory path

## Troubleshooting

### Docker Issues
- Ensure Docker daemon is running
- For GPU support, verify NVIDIA Docker runtime installation
- Check port availability (8888 for Jupyter)

### Permission Issues
- Use rootful Docker configuration if needed
- Ensure proper file permissions in mounted volumes

### Dependencies
- Run `uv run python scripts/installation_verification.py` to verify setup
- Check CUDA compatibility for GPU installations