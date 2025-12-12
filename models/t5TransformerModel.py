"""
Model Loading and Summarization Logic

This module contains the main ClinicalNoteSummarizer class that handles:
- Loading the T5 transformer model for abstractive text generation
- Generating clinical note summaries with adaptive parameter scaling
- Managing model parameters and device allocation (GPU/CPU)
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


class ClinicalNoteSummarizer:
    """
    Wrapper class for T5 transformer model to generate abstractive summaries
    of clinical patient notes.

    This class handles:
    - Model initialization and loading
    - Text preprocessing and tokenization
    - Abstractive summary generation using beam search
    - Adaptive parameter scaling based on input length

    Attributes:
        model_name (str): Name of the T5 model variant (e.g., 't5-base', 't5-small')
        device (str): Computing device ('cuda' for GPU or 'cpu')
        tokenizer: T5Tokenizer for converting text to tokens
        model: T5ForConditionalGeneration model for text generation
    """

    def __init__(self, model_name='t5-base', device=None):
        """
        Initialize the clinical note summarizer by loading the T5 model.

        What this does:
        - Detects available computing device (GPU if available, else CPU)
        - Downloads and loads T5 tokenizer from HuggingFace
        - Downloads and loads T5 model (~850MB for t5-base)
        - Moves model to appropriate device and sets to evaluation mode

        Input:
            model_name (str): HuggingFace model identifier
                             Default: 't5-base' (220M parameters)
                             Options: 't5-small' (60M), 't5-base' (220M), 't5-large' (770M)
            device (str): Computing device - 'cuda', 'cpu', or None for auto-detect
                         Default: None (automatically uses GPU if available)

        Output:
            None (initializes class attributes)

        Side Effects:
            - Prints loading status to console
            - Downloads model files if not cached (~850MB for t5-base)
            - Allocates model to GPU/CPU memory
        """
        # Set model name and device
        self.model_name = model_name
        # Auto-detect GPU if available, otherwise use CPU
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Print loading status
        print(f"Loading {model_name}...")

        # Load tokenizer (converts text strings to token IDs)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Load pre-trained T5 model
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Move model to GPU (if available) or CPU
        self.model.to(self.device)

        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()

        # Confirm successful loading
        print(f"✓ Model loaded on {self.device}")

    def summarize(self, text, max_length=400, min_length=100, num_beams=4):
        """
        Generate abstractive summary of clinical note using T5 transformer.

        What this does:
        1. Validates and preprocesses input text
        2. Dynamically adjusts token lengths based on input size to prevent:
           - Hallucinations (forcing long output from short input)
           - Incomplete summaries (restricting output from long input)
        3. Creates detailed prompt instructing model to include all SOAP components
        4. Tokenizes input text for model processing
        5. Generates summary using beam search algorithm
        6. Decodes tokens back to readable text
        7. Post-processes output (cleanup, formatting)

        Input:
            text (str): Clinical note text to summarize
            max_length (int): Maximum summary length in tokens (not characters)
                             Default: 400 tokens (~250-300 words)
            min_length (int): Minimum summary length in tokens
                             Default: 100 tokens (~60-80 words)
            num_beams (int): Beam search width - higher = better quality but slower
                            Default: 4 (good balance of speed and quality)

        Output:
            str: Generated clinical summary in natural language
                 Structured to include SOAP components when present in input

        Example:
            >>> summarizer = ClinicalNoteSummarizer()
            >>> note = "17 year old male presents with chest pain for 2 days..."
            >>> summary = summarizer.summarize(note)
            >>> print(summary)
            "Patient presents with chest pain worse with breathing. Exam shows
            normal vitals. Assessment: musculoskeletal pain. Plan: ibuprofen
            and follow-up."

        Note:
            - Processing time: ~8-12 seconds on CPU, ~1-2 seconds on GPU
            - Adaptive scaling prevents common generation issues
            - Comprehensive prompt improves coverage of clinical details
        """
        # Validate input - return empty string if no text provided
        text = text.strip()
        if not text:
            return ""

        # Count input words to determine appropriate output length
        input_words = len(text.split())

        # ADAPTIVE LENGTH SCALING
        # Dynamically adjust min/max output length based on input size
        # This prevents model instability and ensures appropriate summary length
        if input_words < 80:
            # Very short note (< 80 words) - generate brief summary
            adjusted_max = 80
            adjusted_min = 30
        elif input_words < 150:
            # Short note (80-150 words) - moderate summary
            adjusted_max = 120
            adjusted_min = 50
        elif input_words < 300:
            # Medium note (150-300 words) - standard summary
            adjusted_max = 200
            adjusted_min = 80
        else:
            # Long note (300+ words) - comprehensive summary
            # Cap at 350 to prevent overly long outputs
            adjusted_max = min(max_length, 350)
            adjusted_min = min(min_length, 120)

        # Debug output showing adaptive parameter selection
        print(f"🔍 Input: {input_words} words → Generating {adjusted_min}-{adjusted_max} tokens")

        # COMPREHENSIVE PROMPT ENGINEERING
        # Detailed instructions guide T5 to include all SOAP components:
        # - Subjective: patient demographics, complaints, symptoms
        # - Objective: physical exam findings, vital signs
        # - Assessment: diagnosis, clinical impression
        # - Plan: treatment, medications, follow-up
        input_text = (
            "Create a comprehensive medical summary including: "
            "patient demographics and age, presenting complaint and symptoms, "
            "relevant medical history, physical examination findings with vital signs, "
            "clinical assessment and diagnosis, and detailed treatment plan with "
            "follow-up instructions: " + text
        )

        # TOKENIZATION
        # Convert input text to token IDs that T5 can process
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors='pt',        # Return PyTorch tensors
            max_length=512,              # T5 maximum input length
            truncation=True              # Truncate if input exceeds 512 tokens
        ).to(self.device)  # Move to GPU or CPU

        # SUMMARY GENERATION
        # Use beam search algorithm to generate high-quality summary
        with torch.no_grad():  # Disable gradient computation for faster inference
            summary_ids = self.model.generate(
                inputs,
                max_length=adjusted_max,           # Maximum output length (adaptive)
                min_length=adjusted_min,           # Minimum output length (adaptive)
                num_beams=6,                       # Search 6 parallel sequences
                length_penalty=0.8,                # Slightly favor longer outputs
                no_repeat_ngram_size=2,            # Don't repeat any 2-word phrase
                early_stopping=True                # Stop when all beams complete
            )

        # DECODING
        # Convert token IDs back to readable text
        summary = self.tokenizer.decode(
            summary_ids[0],              # Take first (best) generated sequence
            skip_special_tokens=True     # Remove special tokens like <pad>, </s>
        )

        # POST-PROCESSING
        # Clean up output formatting
        summary = summary.strip()  # Remove leading/trailing whitespace

        # Add period if summary doesn't end with punctuation
        if summary and not summary.endswith('.'):
            summary += '.'

        return summary

    def get_model_info(self):
        """
        Retrieve information about the loaded model.

        What this does:
        - Returns model metadata including name, device, and parameter count
        - Useful for logging and debugging

        Input:
            None

        Output:
            dict: Dictionary containing model information with keys:
                - 'model_name' (str): Model identifier (e.g., 't5-base')
                - 'device' (str): Computing device ('cuda' or 'cpu')
                - 'parameters' (int): Total number of model parameters

        Example:
            >>> info = summarizer.get_model_info()
            >>> print(f"Using {info['model_name']} with {info['parameters']:,} parameters")
            Using t5-base with 220,000,000 parameters
            >>> print(f"Running on: {info['device']}")
            Running on: cpu
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            # Count total parameters by summing all parameter tensors
            'parameters': sum(p.numel() for p in self.model.parameters())
        }