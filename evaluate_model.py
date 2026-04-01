"""
Evaluation script for the Quantum Physics Text Generator.

Computes perplexity on held-out quantum physics text, compares base GPT-2
with fine-tuned model, and generates multiple samples for qualitative evaluation.

Author: Thiago Girao
"""

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import math
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_perplexity(model, dataloader, device):
    """
    Compute perplexity of model on evaluation dataset.

    Perplexity = exp(average cross-entropy loss)
    Lower perplexity indicates better model performance.

    Args:
        model: GPT-2 model
        dataloader: DataLoader with evaluation data
        device: torch device

    Returns:
        float: Perplexity score
    """
    model.eval()
    total_loss = 0
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            num_samples += input_ids.size(0)

    avg_loss = total_loss / num_samples
    perplexity = math.exp(avg_loss)

    return perplexity, avg_loss


def generate_samples(model, tokenizer, prompts, num_samples=3, device='cpu'):
    """
    Generate multiple samples from provided prompts.

    Args:
        model: GPT-2 model
        tokenizer: Tokenizer
        prompts (list): List of prompt strings
        num_samples (int): Number of samples per prompt
        device: torch device

    Returns:
        dict: Dictionary mapping prompts to generated samples
    """
    model.eval()
    results = {}

    for prompt in prompts:
        logger.info(f"\nGenerating {num_samples} samples for: '{prompt}'")

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=150,
                num_return_sequences=num_samples,
                temperature=0.85,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results[prompt] = generated_texts

    return results


def evaluate_model_comparison(base_model, fine_tuned_model, test_texts, tokenizer, device):
    """
    Compare base GPT-2 vs fine-tuned model on test texts.

    Args:
        base_model: Pre-trained GPT-2 model
        fine_tuned_model: Fine-tuned GPT-2 model
        test_texts (list): List of test text snippets
        tokenizer: Tokenizer
        device: torch device

    Returns:
        dict: Comparison metrics
    """
    logger.info("Comparing base GPT-2 vs fine-tuned model...")

    # Prepare test data
    from quantum_text_generator import QuantumPhysicsDataset

    dataset = QuantumPhysicsDataset(test_texts, tokenizer, max_length=256)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Compute perplexities
    base_perplexity, base_loss = compute_perplexity(base_model, dataloader, device)
    tuned_perplexity, tuned_loss = compute_perplexity(fine_tuned_model, dataloader, device)

    improvement = ((base_perplexity - tuned_perplexity) / base_perplexity) * 100

    results = {
        'base_gpt2_perplexity': base_perplexity,
        'base_gpt2_loss': base_loss,
        'fine_tuned_perplexity': tuned_perplexity,
        'fine_tuned_loss': tuned_loss,
        'improvement_percent': improvement,
    }

    return results


def print_evaluation_report(comparison_results, generation_results):
    """
    Print formatted evaluation report.

    Args:
        comparison_results (dict): Results from model comparison
        generation_results (dict): Results from text generation
    """
    logger.info("\n" + "="*80)
    logger.info("EVALUATION REPORT: QUANTUM PHYSICS TEXT GENERATOR")
    logger.info("="*80 + "\n")

    # Perplexity comparison
    logger.info("PERPLEXITY COMPARISON:")
    logger.info(f"  Base GPT-2:")
    logger.info(f"    Perplexity: {comparison_results['base_gpt2_perplexity']:.4f}")
    logger.info(f"    Loss: {comparison_results['base_gpt2_loss']:.4f}")
    logger.info(f"\n  Fine-tuned Model:")
    logger.info(f"    Perplexity: {comparison_results['fine_tuned_perplexity']:.4f}")
    logger.info(f"    Loss: {comparison_results['fine_tuned_loss']:.4f}")
    logger.info(f"\n  Improvement: {comparison_results['improvement_percent']:.2f}%")

    # Generated samples
    logger.info("\n" + "="*80)
    logger.info("GENERATED TEXT SAMPLES:")
    logger.info("="*80 + "\n")

    for prompt, samples in generation_results.items():
        logger.info(f"Prompt: '{prompt}'")
        logger.info("-" * 80)
        for i, sample in enumerate(samples, 1):
            logger.info(f"Sample {i}:")
            logger.info(f"  {sample}\n")


def main():
    """Main evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load pre-trained (base) GPT-2
    logger.info("Loading base GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # Load fine-tuned model (if available)
    logger.info("Loading fine-tuned model...")
    try:
        fine_tuned_model = GPT2LMHeadModel.from_pretrained("./quantum-gpt2").to(device)
    except OSError:
        logger.warning("Fine-tuned model not found at ./quantum-gpt2")
        logger.warning("Please run quantum_text_generator.py first to create the fine-tuned model")
        return

    # Test data (held-out quantum physics snippets)
    test_texts = [
        """Quantum key distribution protocols like BB84 exploit the measurement postulate of quantum mechanics
        to detect eavesdropping, ensuring unconditional security guaranteed by fundamental quantum mechanical principles.""",

        """The Quantum Zeno effect describes how frequent measurements of a quantum system inhibit its time evolution,
        effectively freezing the system in its measured state through repeated projections onto the measured eigenstate.""",

        """Quantum walks provide a quantum analog of classical random walks with enhanced spreading dynamics and interference
        patterns that enable quantum search algorithms with quadratic speedup over classical counterparts.""",
    ]

    # Compare models
    comparison_results = evaluate_model_comparison(
        base_model,
        fine_tuned_model,
        test_texts,
        tokenizer,
        device
    )

    # Generate samples from fine-tuned model
    prompts = [
        "Quantum entanglement is",
        "The variational quantum eigensolver",
        "Many-body localization exhibits",
        "Topological quantum computing enables",
    ]

    generation_results = generate_samples(
        fine_tuned_model,
        tokenizer,
        prompts,
        num_samples=3,
        device=device
    )

    # Print report
    print_evaluation_report(comparison_results, generation_results)

    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
