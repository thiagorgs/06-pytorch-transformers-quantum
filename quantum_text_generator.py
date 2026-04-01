"""
Quantum Physics Text Generator

Fine-tunes GPT-2 (small) on quantum physics abstracts/text to generate
quantum-physics-flavored text. Demonstrates: Hugging Face Transformers,
PyTorch training loop, tokenization, text generation strategies.

Author: Thiago Girao
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantumPhysicsDataset(Dataset):
    """Dataset of quantum physics text snippets for fine-tuning."""

    def __init__(self, texts, tokenizer, max_length=256):
        """
        Initialize the dataset.

        Args:
            texts (list): List of text strings for fine-tuning
            tokenizer: Hugging Face tokenizer
            max_length (int): Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Tokenize all texts
        self.encodings = []
        for text in texts:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            self.encodings.append(enc)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        enc = self.encodings[idx]
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)

        # For language modeling, labels = input_ids (next token prediction)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


def train(model, dataloader, optimizer, scheduler, device, epochs=3, gradient_accumulation_steps=4):
    """
    Fine-tune the model with proper PyTorch training loop.

    Args:
        model: GPT-2 model to fine-tune
        dataloader: DataLoader for training data
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        device: torch device (cuda or cpu)
        epochs (int): Number of training epochs
        gradient_accumulation_steps (int): Steps to accumulate gradients
    """
    model = model.to(device)
    model.train()

    total_steps = len(dataloader) * epochs
    logger.info(f"Starting training for {epochs} epochs ({total_steps} steps)")

    global_step = 0
    accumulation_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # Normalize loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps
            loss.backward()

            accumulation_counter += 1
            total_loss += loss.item()

            # Update weights after accumulating gradients
            if accumulation_counter % gradient_accumulation_steps == 0:
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Log progress
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

        epoch_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss:.4f}")

    logger.info("Training completed!")
    return model


def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=200,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
    device='cpu'
):
    """
    Generate text using nucleus sampling (top-p + top-k).

    Args:
        model: GPT-2 model
        tokenizer: Tokenizer
        prompt (str): Starting prompt for generation
        max_length (int): Maximum length of generated text
        temperature (float): Controls randomness (higher = more random)
        top_k (int): Keep only top-k most likely next tokens
        top_p (float): Nucleus sampling parameter
        num_return_sequences (int): Number of sequences to generate
        device: torch device

    Returns:
        list: Generated text sequences
    """
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_texts = tokenizer.batch_decode(output, skip_special_tokens=True)
    return generated_texts


def get_training_data():
    """
    Return a list of original quantum physics text snippets for fine-tuning.
    These are original passages written to demonstrate domain knowledge in quantum physics.

    Topics covered:
    - Quantum entanglement and Bell inequalities
    - Quantum computing and variational algorithms
    - Many-body localization and thermalization
    - Quantum error correction
    - Quantum simulation and Hubbard models
    - Quantum machine learning applications
    """
    texts = [
        """Quantum entanglement represents one of the most counterintuitive phenomena in quantum mechanics,
        where two or more particles become correlated in such a way that the quantum state of one particle
        cannot be described independently of the others, even when separated by macroscopic distances. This
        phenomenon, which Einstein famously called 'spooky action at a distance,' has been extensively verified
        through Bell test experiments and forms the foundation for quantum information technologies.""",

        """The variational quantum eigensolver is a hybrid classical-quantum algorithm that combines quantum
        circuits with classical optimization to find ground state energies of molecular Hamiltonians. By preparing
        a parameterized ansatz state on the quantum processor and evaluating expectation values of the Hamiltonian,
        the algorithm iteratively refines parameters using classical gradient descent, making it suitable for
        near-term quantum devices with limited coherence time.""",

        """Many-body localization describes a phase of matter where disorder in a quantum system prevents the
        spreading of information and energy, contradicting the expectations of thermalization. In the presence
        of strong disorder and weak interactions, isolated quantum systems can fail to reach thermal equilibrium
        even in the long-time limit, displaying persistent local memory of their initial conditions and localized
        excitations that do not propagate.""",

        """Quantum error correction is essential for building fault-tolerant quantum computers, as quantum states
        are inherently fragile and subject to decoherence from environmental interactions. Surface codes represent
        a promising topological error correction scheme where logical qubits are encoded nonlocally across a two-dimensional
        lattice of physical qubits, enabling error detection and correction through parity measurements without directly
        measuring the encoded information.""",

        """The quantum approximate optimization algorithm applies a parameterized quantum circuit composed of problem-dependent
        and mixer Hamiltonians to approximate solutions of combinatorial optimization problems. By varying the circuit depth
        and gate parameters, QAOA bridges the gap between classical heuristics and quantum computing, with theoretical guarantees
        on approximation ratios for specific problem instances and empirical success on near-term quantum hardware.""",

        """Quantum simulation on analog quantum processors allows the study of complex quantum systems that are intractable
        for classical computers, such as strongly correlated materials and high-energy physics models. Cold atoms trapped in
        optical lattices can simulate the Hubbard model and other condensed matter systems, revealing emergent phenomena including
        superfluid-insulator transitions and exotic quantum phases.""",

        """Quantum machine learning leverages quantum algorithms to enhance classical machine learning tasks, including state
        preparation for efficient feature mapping, quantum kernel methods for classification, and variational quantum circuits
        for supervised and unsupervised learning. The advantage of quantum machine learning remains an active research question,
        as classical algorithms continue to improve and quantum hardware capabilities expand.""",

        """The dynamics of quantum information in chaotic systems exhibits universal properties characterized by out-of-time-order
        correlators and scrambling, where initial perturbations spread throughout the system becoming distributed across many degrees
        of freedom. Black hole physics suggests that information is ultimately preserved through quantum scrambling, leading to deep
        connections between quantum chaos, holography, and the nature of quantum gravity.""",

        """Topological quantum computing exploits non-abelian anyons and protected edge states to encode and manipulate quantum
        information in a way that is inherently robust against local perturbations and noise. The proposed realization of Majorana
        fermions in condensed matter systems provides a pathway toward topological qubits with inherent error protection, though
        experimental verification remains challenging.""",

        """Quantum metrology uses entanglement and squeezed states to enhance the precision of parameter estimation beyond the
        shot-noise limit achievable with classical resources. By preparing squeezed or N00N states and employing quantum-enhanced
        measurements, quantum sensors can achieve Heisenberg-limited scaling, enabling applications in atomic clocks, gravitational
        wave detection, and magnetometry.""",

        """Adiabatic quantum computing encodes optimization problems into the ground state of a problem Hamiltonian, then slowly
        evolves the system from a trivial initial state to the problem Hamiltonian while remaining in the ground state. The adiabatic
        theorem guarantees success if the evolution is sufficiently slow, though the required evolution time may scale exponentially
        with problem size.""",

        """Quantum phase transitions occur at zero temperature where the ground state of a system undergoes a qualitative change as
        a parameter is varied across a critical point. These transitions are driven by quantum fluctuations rather than thermal fluctuations,
        and are characterized by diverging length scales and power-law correlations that reflect underlying changes in the quantum state
        structure.""",

        """The Hubbard model is a fundamental model in condensed matter physics describing interacting electrons on a lattice with kinetic
        and interaction terms. Despite its simplicity, the model exhibits rich physics including Mott insulators, superconductivity, and
        strange metal behavior, making it central to understanding strongly correlated electron systems in real materials.""",

        """Quantum annealing provides a heuristic approach to optimization by slowly varying a magnetic field to transform an easily-prepared
        initial state into the ground state of a problem Hamiltonian. Although not guaranteed to find global optima, quantum annealing can
        escape local minima through tunneling effects and shows promise for certain optimization landscapes and problem structures.""",

        """Quantum supremacy or quantum advantage describes the achievement where quantum computers solve specific problems faster than any
        known classical algorithm and the best classical simulation. Demonstrated on random circuit sampling and optimization problems, quantum
        advantage motivates development toward practical applications where quantum speedups provide genuine computational benefits.""",

        """Symmetry-protected topological phases are quantum phases protected by global symmetries where the bulk is gapped but protected
        edge modes exist at boundaries. These phases go beyond topological order and can be classified by group cohomology, with applications
        to quantum information and robust edge transport in condensed matter systems.""",

        """The quantum Fourier transform is a key subroutine in quantum algorithms that efficiently computes amplitudes of superposition
        states, enabling period finding and order finding problems with exponential speedup. As a core component of Shor's factoring algorithm,
        the quantum Fourier transform demonstrates how quantum parallelism and interference can provide computational advantages.""",

        """Quantum state tomography aims to reconstruct the density matrix describing a quantum state from repeated measurements on identically
        prepared copies, though the number of measurements required grows exponentially with system size. Classical post-processing and shadow
        techniques provide scalable alternatives that estimate properties of quantum states without full tomography.""",

        """Variational quantum algorithms encompass a broad class of methods using parameterized quantum circuits and classical optimization to
        solve problems on near-term quantum devices. From chemistry simulations to machine learning, variational approaches adapt ansatz designs
        and loss functions to balance expressivity with trainability on noisy quantum processors.""",

        """Quantum entanglement entropy quantifies the degree of quantum correlation between subsystems and diverges at quantum critical points,
        reflecting the emergence of long-range correlations in quantum phases. Computing entanglement entropy through replica tricks and Monte Carlo
        methods provides insights into quantum phase transitions, black hole thermodynamics, and the structure of quantum information in many-body systems.""",
    ]

    return texts


def main():
    """Main training and generation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load pre-trained GPT-2 model and tokenizer
    logger.info("Loading pre-trained GPT-2 model...")
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set pad token to avoid warnings
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Prepare training data
    logger.info("Preparing training data...")
    texts = get_training_data()
    logger.info(f"Loaded {len(texts)} quantum physics text snippets")

    # Create dataset and dataloader
    dataset = QuantumPhysicsDataset(texts, tokenizer, max_length=256)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Training setup
    num_epochs = 3
    num_training_steps = len(dataloader) * num_epochs
    warmup_steps = int(0.1 * num_training_steps)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    logger.info(f"Training for {num_epochs} epochs with {num_training_steps} steps")
    logger.info(f"Warmup steps: {warmup_steps}")

    # Train the model
    model = train(
        model,
        dataloader,
        optimizer,
        scheduler,
        device,
        epochs=num_epochs,
        gradient_accumulation_steps=4
    )

    # Save fine-tuned model
    logger.info("Saving fine-tuned model...")
    model.save_pretrained("./quantum-gpt2")
    tokenizer.save_pretrained("./quantum-gpt2")
    logger.info("Model saved to ./quantum-gpt2")

    # Generate text examples
    logger.info("\n" + "="*80)
    logger.info("GENERATING TEXT EXAMPLES")
    logger.info("="*80 + "\n")

    model.eval()
    prompts = [
        "Quantum entanglement is",
        "The variational quantum eigensolver",
        "Many-body localization occurs when",
        "Quantum error correction protects",
        "Topological quantum computing uses",
    ]

    for prompt in prompts:
        logger.info(f"Prompt: '{prompt}'")
        generated = generate_text(
            model,
            tokenizer,
            prompt,
            max_length=150,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            device=device
        )
        logger.info(f"Generated: {generated[0]}\n")


if __name__ == "__main__":
    main()
