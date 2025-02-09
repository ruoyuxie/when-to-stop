# chunking_manager.py

from typing import List, Tuple, Dict, Optional
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk

class ChunkingManager:
    def __init__(self, tokenizer, activation_manager):
        """Initialize ChunkingManager with tokenizer and activation manager."""
        self.tokenizer = tokenizer
        self.activation_manager = activation_manager
        
        # Initialize NLTK for sentence tokenization
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def create_chunks(
        self,
        context_activations: Optional['ContextActivations'],
        strategy: str,
        task_name: str = "multihop_qa",
        num_chunks: int = 10,
        context: Optional[str] = None  
    ) -> List[int]:
        """Create chunks based on specified strategy."""
        if strategy == "percentage":
            return self._create_percentage_chunks(context_activations, num_chunks, context)
        elif strategy == "sentence":
            return self._create_sentence_chunks(context_activations, task_name, context)
        elif strategy == "token":
            return self._create_token_chunks(context_activations, context)
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

    def _get_tokens(
        self,
        context_activations: Optional['ContextActivations'],
        context: Optional[str]
    ) -> int:
        """Get the total number of tokens, either from ContextActivations or raw text."""
        if context_activations is not None:
            return len(context_activations.model_token_ids[0])
        elif context is not None:
            # Tokenize the context directly
            tokens = self.tokenizer.encode(
                context,
                add_special_tokens=False,
                return_tensors="pt"
            )
            return tokens
        else:
            raise ValueError("Either context_activations or context must be provided")


    def _create_percentage_chunks(
        self,
        context_activations: Optional['ContextActivations'],
        num_chunks: int,
        context: Optional[str] = None
    ) -> List[int]:
        """Create evenly-sized percentage-based chunks."""
        if context_activations is not None:
            tokens = None
            total_tokens = len(context_activations.model_token_ids[0])
        elif context is not None:
            tokens = self._get_tokens(context_activations, context)
            total_tokens = tokens.size(1)
            
        # Calculate chunk boundaries to ensure even distribution
        base_size = total_tokens // num_chunks
        remainder = total_tokens % num_chunks
        
        # Calculate all chunk end points
        chunk_boundaries = []
        current_end = 0
        for i in range(num_chunks):
            # Add one extra token to some chunks if we have remainder
            chunk_size = base_size + (1 if i < remainder else 0)
            current_end += chunk_size
            chunk_boundaries.append(current_end)
        
        return chunk_boundaries, tokens

    def _create_sentence_chunks(
        self,
        context_activations: Optional['ContextActivations'],
        task_name: str,
        context: Optional[str] = None
    ) -> List[int]:
        """Create sentence-based chunks."""
        if context_activations is not None:
            tokens = None
            total_tokens = len(context_activations.model_token_ids[0])
        elif context is not None:
            tokens = self._get_tokens(context_activations, context)
            total_tokens = tokens.size(1)
           
        # Get full text either from context_activations or direct context
        if context_activations is not None:
            full_text = self.tokenizer.decode(
                context_activations.model_token_ids[0],
                skip_special_tokens=True
            )
        else:
            full_text = context

        if not full_text:
            raise ValueError("No text provided for sentence chunking")

        if task_name == "code":
            sentences = self._split_code_into_chunks(full_text)
        elif task_name == "multihop_kv":
            sentences = self._split_kv_into_chunks(full_text)
        else:  # multihop_qa and singlehop_qa
            sentences = self._split_text_into_sentences(full_text)

        # Calculate token boundaries for each sentence
        chunk_boundaries = []
        current_tokens = 0
        
        for sentence in sentences:
            sent_tokens = self.tokenizer.encode(
                sentence,
                add_special_tokens=False
            )
            current_tokens += len(sent_tokens)
            chunk_boundaries.append(current_tokens)
        
        return chunk_boundaries, tokens

    def _create_token_chunks(
        self,
        context_activations: Optional['ContextActivations'],
        context: Optional[str] = None
    ) -> List[int]:
        """Create token-based chunks."""
        if context_activations is not None:
            tokens = None
        elif context is not None:
            tokens = self._get_tokens(context_activations, context)
           
        total_tokens = len(context_activations.model_token_ids[0])
        return list(range(1, total_tokens + 1)), tokens


    def _process_chunks(
        self,
        context_activations: 'ContextActivations',
        chunk_boundaries: List[int]
    ) -> List[Tuple[int, int, Dict[str, np.ndarray], str]]:
        """Process chunks with boundaries and extract features."""
        if context_activations is None:
            raise ValueError("Cannot process chunks without context activations")
            
        # Get features for all chunks in one pass
        all_chunk_features = self.activation_manager.get_chunk_features(
            context_activations,
            chunk_boundaries
        )
        
        # Create chunks with their features
        chunks = []
        for i, end_idx in enumerate(chunk_boundaries):
            # Get cumulative tokens from start to current end
            chunk_tokens = context_activations.model_token_ids[0][0:end_idx]
            
            # Get text for the cumulative chunk
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Store chunk
            chunks.append((
                0,  # start_idx always 0 for progressive chunks
                end_idx,
                all_chunk_features[i],
                chunk_text,
            ))
        
        return chunks


    def _split_text_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving whitespace."""
        sentence_spans = []
        last_end = 0
        current_text = ""
        sentences = []
        
        # Get sentence boundaries from NLTK
        for sent in sent_tokenize(text):
            # Find the actual start of this sentence in the original text
            start = text.find(sent, last_end)
            end = start + len(sent)
            
            # Get any whitespace before this sentence
            whitespace = text[last_end:start]
            
            # Combine whitespace with the sentence
            current_text += whitespace + sent if last_end > 0 else sent
            sentences.append(current_text)
            
            last_end = end
            
        return sentences

    def _split_code_into_chunks(self, text: str) -> List[str]:
        """Split code context into logical chunks."""
        # Split by newlines and remove empty lines
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_text = ""
        
        for line in lines:
            if not line.strip():
                continue
            
            current_chunk.append(line)
            current_text += line + '\n'
            
            # Start new chunk if we find function definition or class
            if line.strip().startswith(('def ', 'class ')):
                chunks.append(current_text)
        
        # Add remaining lines
        if current_chunk:
            chunks.append(current_text)
        
        return chunks

    def _split_kv_into_chunks(self, text: str) -> List[str]:
        """Split key-value pairs into chunks."""
        # Handle both semicolon and newline separators
        text = text.replace('\n', ';')
        # Split by semicolon and filter empty chunks
        pairs = [chunk.strip() for chunk in text.split(';') if chunk.strip()]
        
        # Create progressive chunks
        chunks = []
        current_text = ""
        for pair in pairs:
            current_text += pair + "; "
            chunks.append(current_text)
        
        return chunks