# Use vLLM for Ragas synthetic dataset creation

Ragas synthetic dataset creation is part of the evaluation workflow, and in this project it works reliably only when the generation backend is served through vLLM. We will use vLLM for synthetic dataset creation so benchmark data can be generated reproducibly, while keeping the rest of the LLM integration behind replaceable adapters where possible.
