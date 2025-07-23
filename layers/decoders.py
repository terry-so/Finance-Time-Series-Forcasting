import torch 
import torch.nn as nn

class LSTMDecoder(nn.Module):
    """
    LSTM seq2seq decoder, uses a context vector to initialize its state then generates the output sequence step by step.
    
    """
    def __init__(self, input_size, hidden_size, output_size, pred_len):

        super(LSTMDecoder, self).__init__()
        self.pred_len = pred_len
        self.hidden_size = hidden_size

        self.lstm_cell = nn.LSTMCell(input_size = input_size, hidden_size = hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, context_vector):
        batch_size = context_vector.shape[0]
        h_t = context_vector
        c_t = torch.zeros(batch_size, self.hidden_size).to(context_vector.device)

        #input for first time step
        decoder_input = torch.zeros(batch_size, self.lstm_cell.input_size).to(context_vector.device)

        outputs = []

        for i in range(self.pred_len):
            h_t, c_t = self.lstm_cell(decoder_input, (h_t, c_t))
            output = self.output_layer(h_t)
            outputs.append(output)
            
            decoder_input = h_t

 
        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.permute(1, 0, 2)
        
        return outputs
