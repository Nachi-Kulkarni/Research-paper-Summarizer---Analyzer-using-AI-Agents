# Multi-Agent Research Paper Analysis System

A comprehensive system that processes research papers using multi-agent architecture, generates visualizations, creates flowcharts, and provides an intelligent RAG-enabled chatbot.

## 🌟 Features

- **Advanced PDF Processing**: Automated extraction and parsing with intelligent content analysis
- **Multi-Agent Architecture**: Specialized agents for summarization, pseudocode, code generation, visualization
- **RAG-Enabled Chatbot**: Interactive Q&A with document-aware responses using local Ollama models
- **Intelligent Visualization**: Domain-specific charts, graphs, and visual representations
- **Automated Flowcharts**: Mermaid-based diagram generation for algorithms and system architectures
- **Domain-Adaptive Processing**: Qwen3 1.5B model with templates for AI/ML, CS, ECE, ETE domains

## 🤖 RAG Chatbot & Visualizations

### Interactive Chatbot
- **Logical reasoning** with retrieval-augmented generation
- **Streaming responses** with contextual accuracy
- **FAISS vector store** integration for efficient retrieval

### Advanced Visualizations (`visualization_agent.py`)
- **Domain-specific charts**: AI/ML plots, network diagrams, system architectures
- **Multiple visualization types**: scatter plots, heatmaps, line plots, 3D visualizations
- **Data accuracy**: Only uses actual paper data, never fabricates statistics
- **Publication-quality** graphics with proper labels and annotations

### Flowchart Generation (`generate_diagram.py`)
- **Mermaid-based diagrams**: flowcharts, sequence, class, state diagrams
- **Algorithm workflows**: Converts pseudocode into visual flowcharts
- **System architecture**: Comprehensive design visualizations

## 📄 Output Files

Generated research analysis reports in root directory:

- [`visualizations and flowchart generated.pdf`](./1112_combined_report.pdf)
- [`attention is all you need's visualizations.pdf`](./attention_combined_report.pdf) 
- [`explanation of the attention's paper.pdf`](./research_report_1742590704.pdf) 

Reports include executive summaries, extracted algorithms, implementation code, visual diagrams, and comprehensive analysis.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+, Ollama, required packages

### Installation
```bash
git clone <repository-url>
cd donefinal
pip install -r requirements.txt

# Setup Ollama models
ollama pull qwen2.5:1.5b
ollama pull llama3.2:3b
```

### Usage
```bash
cd multiagenticfinal

# Process papers
python main.py research_papers/paper.pdf

# Start chatbot
python agents/qna_chatbot.py research_papers/paper.pdf

# Generate visualizations
python tryneway/pdf_to_visualization.py paper.pdf
```

## 📁 Project Structure

```
donefinal/
├── multiagenticfinal/           # Main multi-agent system
│   ├── agents/                  # Specialized processing agents
│   └── main.py                  # Orchestrator workflow
├── tryneway/                    # Visualization & diagram generation
│   ├── visualization_agent.py  # Advanced visualization engine
│   └── generate_diagram.py     # Flowchart generation
└── requirements.txt             # Dependencies
```

## 🛠️ Technology Stack

- **Models**: Ollama (Qwen2.5, Llama3.2), Qwen3 1.5B parameter model
- **Vector DB**: FAISS
- **Framework**: Python, LangChain, PyMuPDF
- **Visualization**: Matplotlib, Mermaid diagrams

## 🎯 Use Cases

- Academic research analysis and summarization
- Algorithm implementation from papers
- Interactive learning with document Q&A
- Literature review with visual summaries
- Automated code generation from algorithms

---

**Made with ❤️ for the research community**