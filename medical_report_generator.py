import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class MedicalReportGenerator(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(MedicalReportGenerator, self).__init__()
        
        # Load pre-trained DenseNet121 for image feature extraction
        self.encoder = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        
        # Remove the classifier layer
        self.encoder.classifier = nn.Identity()
        
        # Image feature embedding
        self.embed = nn.Linear(1024, embed_size)
        
        # LSTM decoder for generating reports
        self.decoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        # Word embedding layer
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights for better training"""
        self.embed.weight.data.normal_(0.0, 0.02)
        self.embed.bias.data.fill_(0)
        
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                param.data.fill_(0)
                
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)
        
    def forward(self, images, reports=None, max_length=50):
        """
        Forward pass of the model
        Args:
            images: Input images (batch_size, 3, 224, 224)
            reports: Target reports for training (batch_size, max_length)
            max_length: Maximum length of generated report
        Returns:
            outputs: Generated report tokens
        """
        # Extract image features
        features = self.encoder(images)
        features = self.embed(features)
        
        # Initialize LSTM hidden state
        batch_size = features.size(0)
        h = torch.zeros(2, batch_size, self.decoder.hidden_size).to(features.device)
        c = torch.zeros(2, batch_size, self.decoder.hidden_size).to(features.device)
        
        # Prepare for decoding
        if self.training and reports is not None:
            # Training mode: use teacher forcing
            embeddings = self.word_embedding(reports)
            outputs, _ = self.decoder(embeddings, (h, c))
            outputs = self.fc(outputs)
            return outputs
        else:
            # Inference mode: generate report token by token
            generated = []
            input_token = torch.ones(batch_size, 1).long().to(features.device)  # Start token
            
            for _ in range(max_length):
                # Embed input token
                word_embeddings = self.word_embedding(input_token)
                
                # Concatenate with image features
                decoder_input = word_embeddings + features.unsqueeze(1)
                
                # LSTM step
                output, (h, c) = self.decoder(decoder_input, (h, c))
                output = self.fc(output)
                
                # Get predicted token
                predicted = output.argmax(dim=-1)
                generated.append(predicted)
                
                # Prepare for next step
                input_token = predicted
                
                # Stop if end token is predicted
                if (predicted == 2).any():  # 2 is the index of <end> token
                    break
            
            return torch.cat(generated, dim=1)
    
    def generate_report(self, image, vocab, max_length=50):
        """
        Generate a medical report for a single image
        Args:
            image: Input image tensor (1, 3, 224, 224)
            vocab: Vocabulary dictionary for converting indices to words
            max_length: Maximum length of generated report
        Returns:
            report: Generated medical report as text
        """
        self.eval()
        with torch.no_grad():
            # Generate tokens
            tokens = self.forward(image, max_length=max_length)
            
            # Convert tokens to words
            idx2word = {v: k for k, v in vocab.items()}
            words = [idx2word.get(token.item(), '<unk>') for token in tokens[0]]
            
            # Remove special tokens and join words
            words = [word for word in words if word not in ['<pad>', '<start>', '<end>']]
            report = ' '.join(words)
            
            return report
    
    def save_model(self, path):
        """Save model state"""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        """Load model state"""
        try:
            self.load_state_dict(torch.load(path))
            self.eval()
        except Exception as e:
            return {
                "error": "Please correct the following errors",
                "alerts": [
                    {
                        "field": "model",
                        "type": "error",
                        "title": "Model Load Error",
                        "message": str(e)
                    }
                ]
            }
        return None 