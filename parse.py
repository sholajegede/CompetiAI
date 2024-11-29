from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = (
    "You are tasked with extracting or generating specific information based on the following text content: {dom_content}. "
    "Please follow these instructions carefully:\n\n"
    "1. **Follow Instructions:** Perform the task as described here: {parse_description}.\n"
    "2. **Precise Output:** Provide the most concise and accurate response possible.\n"
    "3. **No Additional Text:** Do not include extra comments, explanations, or unrelated information in your response."
)

model = OllamaLLM(model="llama3.2")

def parse_with_ollama(dom_chunks, parse_description):
    """
    Handles a variety of tasks based on the provided parse_description.
    Parameters:
        - dom_chunks (list of str): Chunks of text content to process.
        - parse_description (str): Instruction for the task (e.g., "Find competitors", "Summarize product").
    Returns:
        - str: Combined result of all tasks across chunks.
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    parsed_results = []
    for i, chunk in enumerate(dom_chunks, start=1):
        response = chain.invoke(
            {"dom_content": chunk, "parse_description": parse_description}
        )
        print(f"Processed batch {i}: {response}")
        parsed_results.append(response)

    return "\n".join(parsed_results)