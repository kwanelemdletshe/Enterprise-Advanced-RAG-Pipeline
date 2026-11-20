import os
from typing import List, Dict, Any
from azure.identity import DefaultAzureCredential
from azure.core.credentials import TokenCredential
from inference import chat_completion


class RAGGenerationOrchestrator:
    """
    Orchestration layer utilizing Azure AI Foundry Project Client to check
    retrieved context bounds, apply guardrails, and stream responses.
    """

    def __init__(
        self,
        model_deployment: str,
        credential: TokenCredential | None = None,
    ):
        self.credential = credential or DefaultAzureCredential()
        self.model_deployment = model_deployment

    def create_grounded_prompt(
        self, query: str, context_segments: List[Dict[str, Any]]
    ) -> str:
        """
        Structures an enterprise context payload injecting strict validation parameters.
        """
        context_str = ""
        for i, doc in enumerate(context_segments, 1):
            context_str += f"\n[{i}] Source: {doc['source_file']} | Page: {doc['page_number']} | Header: {doc['heading_hierarchy']}\n"
            context_str += f"Content: {doc['content']}\n"
            context_str += "-" * 40 + "\n"

        system_prompt = (
            "You are an enterprise AI assistant retrieving structural data parameters.\n"
            "Analyze the verified context blocks provided below to answer the user query.\n\n"
            "CRITICAL GUARDRAILS:\n"
            "1. Answer the query using ONLY the facts explicitly mentioned in the context blocks below.\n"
            "2. If the context does not contain the answer, reply precisely with: 'I am unable to answer this question based on the provided technical documentation.'\n"
            "3. For every claim you make, inject an inline markdown bracketed index citation linking back to its original document source (e.g., [1], [2]).\n\n"
            f"--- VERIFIED CONTEXT BLOCK ---\n{context_str}\n"
            f"--- USER QUERY ---\n{query}"
        )
        return system_prompt

    def generate_response(
        self, query: str, context_segments: List[Dict[str, Any]]
    ) -> str:
        """
        Evaluates context thickness and executes inference via Azure AI Project foundations.
        """
        # FR-4.2 (Agentic Guardrails): Catch execution before LLM processing if context is empty
        if not context_segments or len(context_segments) == 0:
            return "I am unable to answer this question based on the provided technical documentation."

        formatted_prompt = self.create_grounded_prompt(query, context_segments)

        # Utilize unified inference clients mapping directly to workspace endpoints
        return chat_completion(
            credential=self.credential,
            model=self.model_deployment,
            prompt=formatted_prompt,
            temperature=0.0,
            max_tokens=800,
        )
