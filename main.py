import torch
import math
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from helper import vocab_size, midi_to_sequence, process_midi_sequences, sequence_to_midi
import os
import sys
from datetime import datetime
import gdown
from tqdm import tqdm

URL = 'https://drive.google.com/uc?id=1v_3nYiDdgLhWqcE9eHVImZrAHYR2Tjjp'

class Logger:
    def __init__(self, filename):
        self.terminal, self.log = sys.stdout, open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Logging setup
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
sys.stdout = Logger(log_file)
print(f"Logging initialized. Log file: {log_file}")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0)])

class Composer(nn.Module):
    def __init__(self, load_trained=False):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = 256
        self.n_head = 8
        self.dim_ff = 512
        self.num_layers = 6
        self.dropout = 0.1

        self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        if load_trained:
            self.load_model()

        print(f"Composer initialized with vocab_size={self.vocab_size}, d_model={self.d_model}, "
              f"n_head={self.n_head}, dim_ff={self.dim_ff}, num_layers={self.num_layers}")

    def _build_model(self):
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.n_head, self.dim_ff, self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.linear_out = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, src):
        src_mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0, 1))
        output = self.transformer_encoder(src, mask=src_mask)
        return self.linear_out(output.transpose(0, 1))

    def train_step(self, x):
        """
        Perform a single training step on a batch of data.

        Args:
            x (torch.Tensor): Input tensor for training.

        Returns:
            float: The loss value for this training step.
        """
        try:
            x = x.unsqueeze(0) if x.dim() == 1 else x
            self.optimizer.zero_grad()
            output = self(x)

            target = x[:, 1:].contiguous().view(-1)
            output = output[:, :-1].contiguous().view(-1, self.vocab_size)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            return loss.item()
        except Exception as e:
            print(f"Error in train_step method: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def load_model(self):
        print('Loading trained model...')
        model_path = 'abin_model.pt'
        if not os.path.exists(model_path):
            self._download_model(URL)

        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded pre-trained model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']}")
        except Exception as e:
            print(f"Error loading pre-trained model: {str(e)}")
            print("Initializing with random weights instead.")

    @staticmethod
    def _download_model(url):
        print("Downloading pre-trained model...")
        gdown.download(url, 'abin_model.pt', quiet=False)
        print("Pre-trained model downloaded successfully.")

    def compose(self, n, temperature=1.0):
        try:
            device = next(self.parameters()).device
            with torch.no_grad():
                current_seq = torch.tensor([[self.vocab_size-1]], dtype=torch.long).to(device)

                for i in range(n-1):
                    print(f"Generating token number {i+1}")
                    output = self(current_seq)
                    next_token_logits = output[:, -1, :] / temperature
                    next_token_probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(next_token_probs, num_samples=1)
                    current_seq = torch.cat([current_seq, next_token], dim=1)

                return current_seq.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error in compose method: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def train_model():
    print("Starting the training process")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading and processing MIDI data")
    piano_seq = torch.from_numpy(process_midi_sequences())
    print(f"MIDI data shape: {piano_seq.shape}")

    bsz, num_epochs = 32, 300
    print(f"Training parameters: batch_size={bsz}, epochs={num_epochs}")

    loader = DataLoader(TensorDataset(piano_seq), shuffle=True, batch_size=bsz, num_workers=4)
    print(f"DataLoader created with {len(loader)} batches")

    cps = Composer().to(device)
    print("Composer model initialized and moved to device")

    # Training loop with enhanced logging
    for epoch in range(num_epochs):
        cps.train()  # Set the model to training mode
        epoch_loss = 0.0
        batch_count = 0

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}")

        for batch_idx, (x,) in progress_bar:
            loss = cps.train_step(x.to(device).long())
            epoch_loss += loss
            batch_count += 1

            # Update progress bar
            progress_bar.set_postfix({"Batch Loss": f"{loss:.4f}", "Avg Loss": f"{epoch_loss/batch_count:.4f}"})

            # Log every 50 batches
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(loader)}: Loss = {loss:.4f}, Avg Loss = {epoch_loss/batch_count:.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': cps.state_dict(),
                'optimizer_state_dict': cps.optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f'composer_checkpoint_epoch_{epoch+1}.pt')
            print(f"Checkpoint saved at epoch {epoch+1}")

    print("Training completed")

    print("Generating composition...")
    cps.eval()  # Set the model to evaluation mode
    midi = cps.compose(200)
    print("Composition generation finished")
    midi_obj = sequence_to_midi(midi)
    midi_obj.write('piano1.midi')
    print("Composition saved as 'piano1.midi'")

def inference():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cps = Composer(load_trained=True).to(device)
    try:
        composition_length = 2000
        print(f"Generating composition of length {composition_length}...")
        midi_sequence = cps.compose(composition_length)
        print("Composition generation finished")
        midi_obj = sequence_to_midi(midi_sequence)
        midi_obj.write('long_piano_composition.midi')
        print(f"Composition of length {composition_length} saved as 'long_piano_composition.midi'")
    except Exception as e:
        print(f"Error during composition: {str(e)}")
        print("Unable to generate composition")

    print("Script execution completed.")

if __name__ == "__main__":
    # Uncomment the desired function
    train_model()
    # inference()