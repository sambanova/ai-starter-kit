import os


def format_rag_prompt(question, candidate_contexts):
    prompt = 'Here is some relevant context that might assist in answering the SambaNova-related question.\n\n'
    for candidate_context, filename in candidate_contexts:
        prompt += f"Content: {candidate_context}\n\n"
    prompt += f'Answer the following SambaNova-related question: {question}\n\nAnswer:'
    return prompt


def chunk_docs(doc_dir, chunk_size=4096):
    docs = []
    for filename in os.listdir(f"{doc_dir}"):
        if filename.startswith("document_store") or filename.startswith("faiss_document_store.db"):
            continue
        with open(os.path.join(doc_dir, filename), 'r') as f:
            while True:
                chunk = f.read(chunk_size)
                lines = chunk.split('\n')
                content = "".join(lines)
                if not content:  # end of file
                    break
                docs.append(content)
        # with open(f"{doc_dir}/{filename}") as f:
        #     lines = f.readlines()
        #     content = "".join(lines)
        # content_length = len(content)
        # if content_length > 4000:
        #     print(f"Skip file {filename} [{content_length} tokens]")
        #     continue

        # docs.append(content)
    return docs