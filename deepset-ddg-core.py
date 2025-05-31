from duckduckgo_search import DDGS
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import sys

model_name = "deepset/roberta-base-squad2"

def retrieve_context(query, max_results=5):
    """
    Retrieve search result snippets from DuckDuckGo for a given query.
    Combines the 'body' fields of the search results into a single context.
    """
    context_parts = []
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        for result in results:
            snippet = result.get("body")
            if snippet:
                context_parts.append(snippet)
    # Combine snippets into one context string.
    context = " ".join(context_parts)
    return context

def answer_question(question):
    """
    Retrieves search context from DuckDuckGo and uses it as input for the QA pipeline.
    """
    # Retrieve context using the question as the search query.
    context = retrieve_context(question)
    if not context:
        print("No context found. Please try a different query or check your connection.")
        return None
    print("Retrieved Context:\n", context, "\n" + "-"*80 + "\n")
    
    # Initialize the question-answering pipeline.
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    
    # Prepare the input for the pipeline.
    QA_input = {
        'question': question,
        'context': context
    }
    
    # Get the answer from the QA model.
    answer = nlp(QA_input)
    return answer

if __name__ == "__main__":
    question = " ".join(sys.argv[1:])
    answer = answer_question(question)
    if answer:
        print("Answer:")
        print(answer)
