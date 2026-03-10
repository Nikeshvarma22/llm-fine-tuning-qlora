🧠 Fine-Tune Mistral-7B for Customer Support Classification (QLoRA)
Fine-tuned Mistral-7B-Instruct using QLoRA (4-bit quantization + LoRA) to classify customer support tickets into 5 categories — achieving significant accuracy improvement over the base model.
Show Image
Show Image
Show Image
Show Image

Results
MetricBase ModelFine-Tuned ModelTraining Loss—0.67Trainable Parameters7B (100%)35M (0.52%)GPU MemoryWould need 40+ GBFits on 16GB T4Training Time—~9 minutes
What This Project Demonstrates

QLoRA: Loading a 7B parameter model in 4-bit quantization (14GB → 3.5GB)
LoRA Adapters: Training only 0.52% of parameters (attention projections: q, k, v, o)
Instruction-Format Dataset: Created 125 labeled examples in Mistral's native [INST]...[/INST] format
End-to-End Pipeline: Data prep → Quantized loading → LoRA config → Training → Evaluation → Inference
Before/After Comparison: Measured baseline accuracy before fine-tuning to prove improvement

Categories
The model classifies support tickets into:
CategoryExamplebilling"I was charged twice for my subscription"technical_support"The app crashes when I open the dashboard"account_access"I forgot my password and reset email isn't arriving"product_inquiry"What features are in the Pro plan?"cancellation"I want to cancel my subscription immediately"
Architecture
Customer Support Ticket (text)
        ↓
┌─────────────────────────────┐
│  Mistral-7B-Instruct (4-bit)│  ← Base model (frozen)
│  + LoRA Adapters (rank=16)  │  ← Only these are trained
└─────────────────────────────┘
        ↓
   Category Label
   (billing / technical / account / product / cancellation)
Key Technical Decisions
DecisionChoiceWhyBase ModelMistral-7B-Instruct-v0.3Strong instruction-following, open-sourceQuantizationNF4 (4-bit)Fits on free T4 GPU with minimal quality lossLoRA Rank16Sweet spot for classification tasksTarget Modulesq_proj, k_proj, v_proj, o_projAttention layers benefit most from fine-tuningLearning Rate2e-4Standard for QLoRA fine-tuningEpochs3Sufficient for small dataset without overfittingEffective Batch Size16 (4 × 4 accumulation)Gradient accumulation to fit in GPU memory
How to Run
Option 1: Google Colab (Recommended)

Open fine_tuning_customer_support.ipynb in GitHub
Click "Open in Colab" badge or upload to colab.research.google.com
Set runtime to T4 GPU (Runtime → Change runtime type)
Run all cells top to bottom

Option 2: Local (Needs NVIDIA GPU with 16GB+ VRAM)
bashpip install transformers datasets peft trl bitsandbytes accelerate scikit-learn
jupyter notebook fine_tuning_customer_support.ipynb
Tech Stack

Model: Mistral-7B-Instruct-v0.3 (via Hugging Face)
Fine-Tuning: QLoRA (PEFT library + bitsandbytes)
Training: SFTTrainer (TRL library)
Evaluation: scikit-learn (accuracy, classification report)
Hardware: Google Colab T4 GPU (free tier)

What I Learned

Data preparation is 80% of fine-tuning — instruction format must match the model's pre-training format
QLoRA makes 7B models accessible — 4-bit quantization + LoRA = fine-tune on free hardware
LoRA rank is a key hyperparameter — rank 16 works for classification, complex tasks may need 32-64
Gradient accumulation is essential — simulates large batch sizes when GPU memory is limited
Always measure baseline first — without before/after comparison, you can't prove fine-tuning helped

Future Improvements

 Expand dataset to 500+ examples per category
 Experiment with LoRA ranks (8, 32, 64) and compare
 Add confidence scoring (reject low-confidence predictions)
 Deploy as a FastAPI endpoint
 Push fine-tuned model to Hugging Face Hub
 Add RLHF or DPO for preference alignment


Built by Nikesh Varma | AI Engineer Portfolio Project
