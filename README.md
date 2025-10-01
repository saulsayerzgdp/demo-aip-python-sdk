# CV Reader Agent Demo

A Python-based demonstration of using the AIP (AI Platform) Python SDK to create a CV/Resume reader agent. This project showcases how to process PDF documents and extract information using AI-powered tools.

## Features

- Extract text from PDF documents
- Process CV/Resume content using AI
- Simple command-line interface
- Batch processing of multiple queries
- Evaluation of results using GEval

## Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)
- AIP API credentials

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/saulsayerzgdp/demo-aip-python-sdk.git
   cd demo-aip-python-sdk
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install --no-root
   ```

3. Set up your environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` with your AIP API credentials.

## Usage

### Simple Demo

Run the basic CV reader demo:
```bash
poetry run python hello_world.py
```

### Full Pipeline

Run the end-to-end pipeline with evaluation:
```bash
poetry run python demo.py
```

## Project Structure

- `hello_world.py` - Simple demo script
- `demo.py` - Full pipeline with evaluation
- `utils.py` - Utility functions
- `cv_reader_tool.py` - Custom PDF reader tool
- `sample_cv.pdf` - Example CV for testing
- `cv_agent_results.csv` - Output results file

## Configuration

Edit `.env` to configure:
- `AIP_API_KEY` - Your AIP API key
- `AIP_API_URL` - AIP API endpoint

## Dependencies

- `glaip-sdk` - AIP Python SDK
- `gllm-evals` - For evaluation metrics
- `python-dotenv` - Environment variable management

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Saul Sayers - [@saulsayerzgdp](https://github.com/saulsayerzgdp)

## Acknowledgments

- GDP Labs AI Team
- AIP Platform
