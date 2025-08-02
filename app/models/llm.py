import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import PersistentClient
import numpy as np
from typing import List, Dict, Optional
import json
from glob import glob


class LLM:
    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
        peft_dir: str = "./qlora_adapter",
        vectorstore_dir: str = "./chroma_db",
        embedder_model: str = "all-MiniLM-L6-v2",
        load_existing_adapter: bool = False,
        device: Optional[str] = None,
    ):
        """
        Initialize the LLM class with QLoRA support and vector store integration.
        Args:
            base_model: HuggingFace model identifier
            peft_dir: Directory to save/load LoRA adapter weights
            vectorstore_dir: Directory for ChromaDB vector store
            embedder_model: Sentence transformer model for embeddings
            load_existing_adapter: Whether to load existing LoRA weights
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.base_model = base_model
        self.peft_dir = peft_dir
        self.vectorstore_dir = vectorstore_dir
        self.load_existing_adapter = load_existing_adapter
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.embedder = SentenceTransformer(embedder_model)

        # Initialize vector store
        self._setup_vector_store()

        # Load model and tokenizer
        self._load_model_and_tokenizer()

    def _setup_vector_store(self):
        """Initialize ChromaDB client and collection."""
        self.client = PersistentClient(path=self.vectorstore_dir)
        self.collection = self.client.get_or_create_collection(name="style_memory")

    def _check_model_files_exist(self, path: str) -> bool:
        """Check if required model files exist in the given path."""
        required_files = ["config.json"]  # Minimum required file
        optional_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model-00001-of-00001.bin",
        ]

        # Check if config.json exists (required)
        if not os.path.exists(os.path.join(path, "config.json")):
            return False

        # Check if at least one model weight file exists
        model_file_exists = any(
            os.path.exists(os.path.join(path, f)) for f in optional_files
        )

        # Also check for sharded models (multiple files)
        if not model_file_exists:
            # Look for any pytorch_model files or safetensors files
            for file in os.listdir(path):
                if file.startswith("pytorch_model") or file.endswith(".safetensors"):
                    model_file_exists = True
                    break

        return model_file_exists

    def _find_cached_model(self, local_cache_dir: str) -> Optional[str]:
        """Find the model in local cache directory if it exists."""
        if not os.path.exists(local_cache_dir):
            return None

        # Try different potential paths
        potential_paths = [
            os.path.join(
                local_cache_dir, self.base_model.split("/")[-1]
            ),  # Just model name
            os.path.join(
                local_cache_dir, self.base_model.replace("/", "--")
            ),  # HF cache format
            os.path.join(local_cache_dir, self.base_model),  # Full path
        ]

        for path in potential_paths:
            if os.path.exists(path) and os.path.isdir(path):
                if self._check_model_files_exist(path):
                    print(f"Found cached model at: {path}")
                    return path
                else:
                    print(f"Found directory {path} but missing required model files")

        return None

    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with QLoRA configuration from local cache or download."""
        print(f"Loading model on device: {self.device}")

        # Login to HuggingFace if token is available
        if os.environ.get("HF_TOKEN"):
            from huggingface_hub import login

            login(token=os.environ.get("HF_TOKEN"))

        # Define local cache directory
        local_cache_dir = "/app/models/model_storage/hf_cache/"

        # Check if model exists in local cache
        cached_model_path = self._find_cached_model(local_cache_dir)

        if cached_model_path:
            model_path = cached_model_path
            use_local_files = True
            cache_dir = None  # Don't need cache_dir when using local files
            print(f"Using cached model from: {model_path}")
        else:
            model_path = self.base_model
            use_local_files = False
            cache_dir = (
                local_cache_dir
                if os.path.exists(os.path.dirname(local_cache_dir))
                else None
            )
            print(
                f"Model not found in cache, downloading from HuggingFace: {self.base_model}"
            )

            # Create cache directory if it doesn't exist
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
                print(f"Created cache directory: {cache_dir}")

        # Initialize tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            cache_dir=cache_dir,
            local_files_only=use_local_files,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4-bit quantization config for QLoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # Load base model with quantization
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=use_local_files,
        )

        # Setup LoRA adapter
        self._setup_lora_adapter()

    def _setup_lora_adapter(self):
        """Setup LoRA adapter - either load existing or prepare for training."""
        # Load existing adapter or prepare for training
        if self.load_existing_adapter and os.path.exists(self.peft_dir):
            print(f"Loading existing LoRA adapter from {self.peft_dir}")
            self.model = PeftModel.from_pretrained(self.model, self.peft_dir)
        else:
            print("Preparing model for training")
            self.model = prepare_model_for_kbit_training(self.model)

            # LoRA configuration
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

        self.model.to(self.device)

        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()

    def load_data_to_vector_store(self, data_dir: str, pattern: str = "*.json"):
        """
        Load training data into the vector store.
        Args:
            data_dir: Directory containing JSON data files
            pattern: File pattern to match (default: "*.json")
        """
        data_files = glob(os.path.join(data_dir, pattern))
        if not data_files:
            print(f"No files found matching pattern {pattern} in {data_dir}")
            return

        for data_file in data_files:
            print(f"Loading data from: {data_file}")
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "text" in data:
                texts = (
                    data["text"] if isinstance(data["text"], list) else [data["text"]]
                )
                embeddings = self.embedder.encode(texts).tolist()
                ids = [
                    f"doc_{os.path.basename(data_file)}_{i}" for i in range(len(texts))
                ]

                self.collection.add(documents=texts, embeddings=embeddings, ids=ids)
                print(f"Added {len(texts)} documents to vector store")

    def prepare_training_dataset(
        self, data_dir: str, pattern: str = "*.json", max_length: int = 128
    ):
        """
        Prepare dataset for training from JSON files.
        Args:
            data_dir: Directory containing training data
            pattern: File pattern to match
            max_length: Maximum sequence length for tokenization
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        data_files = glob(os.path.join(data_dir, pattern))
        all_texts = []

        for data_file in data_files:
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "text" in data:
                texts = (
                    data["text"] if isinstance(data["text"], list) else [data["text"]]
                )
                all_texts.extend(texts)

        dataset = Dataset.from_dict({"text": all_texts})

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        def group_texts(examples):
            labels = []
            for input_ids in examples["input_ids"]:
                label = input_ids.copy()
                label = [
                    -100 if token == self.tokenizer.pad_token_id else token
                    for token in label
                ]
                labels.append(label)
            examples["labels"] = labels
            return examples

        tokenized_dataset = tokenized_dataset.map(group_texts, batched=True)
        split = tokenized_dataset.train_test_split(test_size=0.1)
        return split["train"], split["test"]

    def train(
        self,
        train_dataset,
        eval_dataset,
        output_dir: Optional[str] = None,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        learning_rate: float = 2e-4,
        num_train_epochs: int = 3,
        save_steps: int = 100,
        eval_steps: int = 100,
        logging_steps: int = 10,
    ):
        """
        Train the model with QLoRA.
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Directory to save model checkpoints
            **kwargs: Additional training arguments
        """
        output_dir = output_dir or self.peft_dir

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            eval_steps=eval_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            save_total_limit=2,
            bf16=True if torch.cuda.is_bf16_supported() else False,
            gradient_checkpointing=True,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        print("Starting training...")
        trainer.train()
        print("Training completed!")

    def generate_with_context(
        self,
        user_input: str,
        top_k: int = 10,
        max_new_tokens: int = 50,
        context_length: int = 1000,
        do_sample: bool = False,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate response using retrieved context from vector store.
        Args:
            user_input: User's question/prompt
            top_k: Number of similar documents to retrieve
            max_new_tokens: Maximum tokens to generate
            context_length: Maximum context length in characters
            do_sample: Whether to use sampling for generation
            temperature: Temperature for sampling (if enabled)
        Returns:
            Generated response string
        """
        # Retrieve relevant context
        query_vec = self.embedder.encode(user_input).tolist()
        results = self.collection.query(query_embeddings=[query_vec], n_results=top_k)

        docs = results.get("documents", [[]])[0]
        if not docs:
            context = "No relevant context found."
        else:
            context = "\n".join(docs)[:context_length]

        # Create prompt
        prompt = f"""You are the person in the contextual statements. Use the context to answer the question briefly and only once.

Context:
{context}

Q: {user_input}
A:"""

        # Tokenize and generate
        inputs = self.tokenizer(
            prompt, return_tensors="pt", return_attention_mask=True
        ).to(self.device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = decoded.split("A:")[-1].strip()
        return answer

    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> str:
        """
        Generate response without context retrieval.
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
        Returns:
            Generated response string
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt", return_attention_mask=True
        ).to(self.device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **kwargs,
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from the output
        response = decoded[len(prompt) :].strip()
        return response

    def save_adapter(self, save_path: Optional[str] = None):
        """Save the LoRA adapter weights."""
        save_path = save_path or self.peft_dir
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_path)
            print(f"Adapter saved to {save_path}")
        else:
            print("Model does not support saving adapters")

    def get_vector_store_stats(self) -> Dict:
        """Get statistics about the vector store."""
        try:
            count = self.collection.count()
            return {"total_documents": count}
        except Exception as e:
            return {"error": str(e)}


# Example usage
# if __name__ == "__main__":
# Initialize the LLM
# llm = LLM(load_existing_adapter=False)

# Load data into vector store
# data_dir = "../data/prompt_response/"
# llm.load_data_to_vector_store(data_dir)

# Example inference
# user_question = "What is your favorite color?"
# response = llm.generate_with_context(user_question)
# print(f"Q: {user_question}")
# print(f"A: {response}")

# Example training (uncomment to use)
# train_dataset, eval_dataset = llm.prepare_training_dataset(data_dir)
# llm.train(train_dataset, eval_dataset)
