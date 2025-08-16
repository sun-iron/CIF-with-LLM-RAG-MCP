# CIF(Conversational Integration Framework) with LLM, RAG, MCP

## Project Overview

This project originates from the growing need for specialized technical knowledge within enterprise systems, especially in the fast-evolving world of technology. While generative AI models provide convenient access to information, the differences in types and versions between external AI sources and internal enterprise systems often make it difficult to obtain immediately practical results.

<img width="1028" height="459" alt="image" src="https://github.com/user-attachments/assets/0c2b04ac-f577-4c8d-81d5-8758398ff22e" />

To address this, we propose an interactive framework that integrates the following components:
- An on-premises LLM (Large Language Model) tailored for enterprise environments
- A RAG (Retrieval Augmented Generation) system powered by a vector database specialized for enterprise data
- An MCP (Modular Control Platform) that connects seamlessly with internal systems
## Key Features

- **Enterprise-Specific RAG**: Stores and retrieves company technical documents and data using a vector database, enabling accurate and context-aware answers.
- **Internal System Integration (MCP)**: Flexible integration with various enterprise systems for automation and support.
- **On-Premises LLM Support**: Ensures data privacy and compliance with internal security policies by running AI models locally.
- **Conversational Interface**: Users can interact using natural language to receive real-time, system-integrated responses.

## Tested Environment

<img width="1703" height="799" alt="image" src="https://github.com/user-attachments/assets/6434bd9a-82a9-4530-9077-9123d3fb0c7b" />

This project has been tested on a Linux environment using the following technologies:
- Python-based uv
- Mistral 7B (LLM)
- FastAPI
- FastMCP

## Installation & Usage

1. Install dependencies and configure the environment.
2. Connect to internal systems and load enterprise data.
3. Set up the LLM and MCP modules.
4. Launch the conversational interface.

## Target Users

- Enterprise IT administrators and developers
- Teams responsible for automation, knowledge management, or technical support within companies

## License

This research is prepared for my doctoral dissertation.

---

For questions or feedback, please open an Issue or contact me via email: sunpark@soongsil.ac.kr
