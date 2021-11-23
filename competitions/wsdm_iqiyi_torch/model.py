import torch
import torch.nn as nn


class AQYModel(nn.Module):
    def __init__(self):
        super(AQYModel, self).__init__()

        self.user_id_embedding = nn.Embedding(600000 + 1, 16)
        self.launch_type_embedding = nn.Embedding(2 + 1, 16)

        self.launch_seq_gru = nn.GRU(input_size=16,
                                     hidden_size=16,
                                     batch_first=True)

        self.fc = nn.Linear(32, 1)

    def forward(self, user_id, launch_seq):
        user_id_emb = self.user_id_embedding(user_id)

        # launch_seq = launch_seq.reshape((-1, 32, 1))
        launch_seq = self.launch_type_embedding(launch_seq)

        launch_seq, _ = self.launch_seq_gru(launch_seq)
        launch_seq = torch.mean(launch_seq, dim=1)

        fc_input = torch.cat([user_id_emb, launch_seq], 1)

        pred = self.fc(fc_input)

        return pred
