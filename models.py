import torch
import torch.nn as nn
from torch import cat
import torch.nn.functional as F
# from pointnet import *

class DenoisingDiffusionModelMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, T):
        super(DenoisingDiffusionModelMLP, self).__init__()

        # MLP for processing inputs (sample)
        self.input_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

        # embedding t
        self.t_embedding = nn.Embedding(T, input_dim)

        # change to broadcast t to size (1, sample_size)
        # input = sample + t
        # one NN

    def forward(self, t, inputs):
        t_emb = self.t_embedding(t.int())
        # Process inputs separately
        input_output = self.input_processor(inputs + 0.5 * t_emb)

        # Process t separately
        # t_output = self.t_processor(t.view(1, 1))  # Add view to handle scalar input

        # Combine the processed outputs
        combined_output = input_output

        return combined_output


class DenoisingDiffusionModelUNet(nn.Module):
    def __init__(self, T):
        super(DenoisingDiffusionModelUNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )
        self.t_embedding = nn.Embedding(T, 32 * 32)

    def forward(self, t, inputs):
        t_emb = self.t_embedding(t.int())
        x1 = self.encoder(inputs + t_emb.view(1, 32, 32))
        x2 = self.bottleneck(x1)
        x3 = self.decoder(x2)
        return x3


#
# class DenoisingDiffusionModelUNet1D(nn.Module):
#     def __init__(self, T):
#         super().__init__()
#         # Encoder
#         self.encoder_conv1 = nn.Conv1d(1, 10, kernel_size=3, padding=1)
#         self.encoder_conv2 = nn.Conv1d(10, 20, kernel_size=3, padding=1)
#         self.encoder_conv3 = nn.Conv1d(20, 50, kernel_size=3, padding=1)
#         self.encoder_activation = nn.LeakyReLU()
#         self.encoder_pool = nn.AvgPool1d(kernel_size=2)
#
#         # Decoder
#         self.decoder_conv1 = nn.ConvTranspose1d(50, 20, kernel_size=2, stride=2)
#         self.decoder_conv2 = nn.ConvTranspose1d(20, 10, kernel_size=2, stride=2)
#         self.decoder_conv3 = nn.Conv1d(10, 1, kernel_size=3, padding=1)
#         self.decoder_activation = nn.LeakyReLU()
#
#         self.linear_layer = nn.Linear(3300, 1000)
#         self.t_embedding = nn.Embedding(T, 100)
#
#     def forward(self, t, x):
#         t_emb = self.t_embedding(t.int())
#         t_emb = t_emb.unsqueeze(0)
#
#         # Encoder
#         x1 = self.encoder_conv1(cat((x, t_emb),dim=1))
#         x1_skip = self.encoder_activation(x1)
#         x = self.encoder_pool(x1_skip)
#
#         x2 = self.encoder_conv2(x)
#         x2_skip = self.encoder_activation(x2)
#         x = self.encoder_pool(x2_skip)
#
#         x = self.encoder_conv3(x)
#         x = self.encoder_activation(x)
#
#         # Decoder
#         x = self.decoder_conv1(x)
#         x = self.decoder_activation(x)
#         x = torch.cat((x, x2_skip), dim=1)  # Concatenate with skip connection
#
#         x = self.decoder_conv2(x)
#         x = self.decoder_activation(x)
#         x = torch.cat((x, x1_skip), dim=1)  # Concatenate with skip connection
#
#         x = self.decoder_conv3(x)
#
#         # Reshape the output without adaptive average pooling
#         print(x.size())
#         x_reshaped = x.view(x.size(0), -1)  # Reshape to [batch_size, 1024]
#
#         # Ensure the linear layer input size matches the reshaped output
#         x_reshaped = self.linear_layer(x_reshaped)
#
#         return x_reshaped

#
# class DenoisingDiffusionModelUNet1D(nn.Module):
#     def __init__(self, T):
#         super().__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=3, padding=1),
#             nn.Tanh(),
#             nn.BatchNorm1d(16),
#             nn.Conv1d(16, 32, kernel_size=3, padding=1),
#             nn.Tanh(),
#             nn.BatchNorm1d(32),
#             # nn.Conv1d(512, 256, kernel_size=3, padding=1),
#             # nn.LeakyReLU(),
#             # nn.Conv1d(256, 128, kernel_size=3, padding=1),
#             # nn.LeakyReLU(),
#             # nn.Conv1d(128, 64, kernel_size=3, padding=1),
#             # nn.LeakyReLU(),
#             # nn.Conv1d(64, 32, kernel_size=3, padding=1),
#             # nn.LeakyReLU(),
#             #nn.MaxPool1d(kernel_size=2, stride=2),
#         )
#
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Conv1d(32, 16, kernel_size=3, padding=1),
#             nn.Tanh(),
#             nn.BatchNorm1d(16),
#             nn.Conv1d(16, 1, kernel_size=3, padding=1),
#             #nn.LeakyReLU(),
#             # nn.Conv1d(64, 128, kernel_size=3, padding=1),
#             # nn.LeakyReLU(),
#             # nn.Conv1d(128, 256, kernel_size=3, padding=1),
#             # nn.LeakyReLU(),
#             # nn.Conv1d(256, 512, kernel_size=3, padding=1),
#             # nn.LeakyReLU(),
#             # nn.Conv1d(512, 1024, kernel_size=3, padding=1),
#         )
#
#         self.linear_layer = nn.Linear(1124, 1024)
#         self.t_embedding = nn.Embedding(T, 1024)  # REPLACE 1024 WITH SOMETHING SMALLER
#
#     def forward(self, t, x):
#         print(f"{x.size()=}")
#         # t_emb = torch.tensor([0])
#         # for t in range(0,len(t)):
#         #         t_i = t[0,i]
#         #         t_emb_i = self.t_embedding(t_i.int())
#         #         t_emb = torch.cat((t_emb,t_i))
#         t_emb = self.t_embedding(t.int())
#         #t_emb = torch.permute(t_emb, (1, 0, 2))
#         print(f"{t_emb.size()=}")
#         input = x + t_emb
#         # x2 = self.linear_layer(input)
#         # input = torch.cat((x, t_emb), dim=1)
#         #input = torch.unsqueeze(input,2)
#         # input = torch.permute(input, (0,2,1))
#         print(f"{input.size()=}")
#         x1 = self.encoder(input)  # CONCATENATE
#         print(f"{x1.size()=}")
#         x2 = self.decoder(x1)
#         # x2 = torch.squeeze(x2, 0)
#         print(f"{x2.size()=}")
#         # Reshape the output without adaptive average pooling
#         # x2_reshaped = x2.view(x2.size(0), -1)  # Reshape to [batch_size, 1024]
#         #f
#         # Ensure the lerear layer input size matches the reshaped output
#         # x2_reshaped = self.linear_layer(x2)
#
#         return x2
#

class DenoisingDiffusionModelUNet1D(nn.Module):
    def __init__(self, T):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 127, kernel_size=3, stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(127),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 1, kernel_size=3, stride=3),
        )

        self.linear_layer = nn.Linear(1074, 1024)
        self.t_embedding = nn.Embedding(T, 128)  # REPLACE 1024 WITH SOMETHING SMALLER

    def forward(self, t, x):
        #x = x.unsqueeze(2)
        print(f"{x.size()=}")
        t_emb = self.t_embedding(t.int())#.unsqueeze(0).unsqueeze(2)
        print(f"{t_emb.size()=}")
        input = x # + t_emb
        # x2 = self.linear_layer(input)
        # input = torch.cat((x, t_emb), dim=-1)
        #input = torch.unsqueeze(input,2)
        #input = torch.permute(input, (0,2,1))
        print(f"{input.size()=}")
        x1 = self.encoder(input)  # CONCATENATE
        print(f"{x1.size()=}")
        x1 = torch.cat((x1, t_emb), dim = 1)
        # x1 = self.bottleneck(x1)
        x2 = self.decoder(x1)
        #x2 = torch.squeeze(x2, 0)
        print(f"{x2.size()=}")
        # Reshape the output without adaptive average pooling
        # x2_reshaped = x2.view(x2.size(0), -1)  # Reshape to [batch_size, 1024]
        #
        # Ensure the linear layer input size matches the reshaped output
        #x2_reshaped = self.linear_layer(x2)
        #print(f"{x2_reshaped.size()=}")
        return x2


class DenoisingDiffusionModelUNet2D(nn.Module):
    def __init__(self, T):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(8, 15, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(15),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Conv1d(64, 127, kernel_size=2, stride=2),
            # # nn.Tanh(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            # nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.BatchNorm1d(8),
            nn.ConvTranspose1d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4),
            nn.ConvTranspose1d(4, 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4),
            nn.ConvTranspose1d(4, 2, kernel_size=4, stride=2, padding=1),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(2),
            # nn.ConvTranspose1d(8, 2, kernel_size=4, stride=2, padding=1),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(16),
            # nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(8),
            # nn.ConvTranspose1d(8, 2, kernel_size=4, stride=2, padding=1),
            # nn.ConvTranspose1d(32, 2, kernel_size=4, stride=2, padding=1),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(32),
            # nn.ConvTranspose1d(32, 2, kernel_size=4, stride=2, padding=1),
        )

        self.t_embedding = nn.Embedding(T, 64)

    def forward(self, t, x):
        print(f"{x.size()=}")
        t_emb = self.t_embedding(t.int())
        print(f"{t_emb.size()=}")
        input = x.float()
        print(f"{input.size()=}")
        x1 = self.encoder(input)
        print(f"{x1.size()=}")
        x1 = torch.cat((x1, t_emb), dim = 1)
        x2 = self.decoder(x1)
        print(f"{x2.size()=}")
        return x2




class DenoisingDiffusionModelUNetND(nn.Module):
    def __init__(self, T):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential( #layer norm?
            #check where actual large/small entries are (is it learning random positions)?
            nn.Conv2d(1, 8, kernel_size=3,stride=1,padding=1),
            nn.RReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.PReLU(init=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 15, kernel_size=4, stride=2, padding=1),
            nn.PReLU(init=1),
            # nn.RReLU(),
            nn.BatchNorm2d(15),
        )
        # Decoder
        self.decoder = nn.Sequential(
            # nn.RReLU(),
            nn.PReLU(init=1),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(init=1),
            # nn.RReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.PReLU(init=1),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.PReLU(init=1),
        )

        self.t_embedding = nn.Embedding(T, 512)
        #
        # self.bottleneck = nn.Sequential(
        #     nn.BatchNorm2d(8),
        #     nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,stride=1,padding=1),
        #     nn.LeakyReLU(),
        # )

        # self.activation = torch.nn.PReLU(num_parameters=2, init=1)

    def forward(self, t, x):
        print(f"{x.size()=}")
        t_emb = self.t_embedding(t.int())
        print(f"{t_emb.size()=}")
        # input = x.unsqueeze(1)
        input = x
        print(f"{input.size()=}")
        x1 = self.encoder(input)
        print(f"{x1.size()=}")
        x1 = torch.cat((x1, t_emb.unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, 4)), dim=1)
        # x1 = self.bottleneck(x1)
        print(f"{x1.size()=}")
        x2 = self.decoder(x1)
        print(f"{x2.size()=}")
        return x2#.permute(0,1,3,2)




class DenoisingDiffusionModelCNN(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1, padding=1)

        self.fc1 = nn.Linear(589824, 8*1024)

        self.t_embedding = nn.Embedding(T, 1024)

    # def forward(self, t, x):
    #     input = torch.cat((x, t_emb.unsqueeze(1)), dim = 1)
    #     print(f"{input.size()=}")
    #     output = self.network(input)
    #     return output.permute(0,2,1).unsqueeze(1)

    def forward(self, t, x):
        x = x.unsqueeze(1)
        print(f"{x.size()=}")
        t_emb = self.t_embedding(t.int())
        print(f"{t_emb.size()=}")
        input = torch.cat((x, t_emb.unsqueeze(1).unsqueeze(1)), dim=-2)
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        print(f"{x.size()=}")
        # x = self.fc1(x)
        return x.reshape(10,1, 8, 1024)


class DenoisingDiffusionModelLinear(nn.Module):
    def __init__(self, T):
        super().__init__()
        # Encoder
        self.linear = nn.Linear(1024,1024)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
        )

        self.linear_layer = nn.Linear(1024, 1024)
        self.t_embedding = nn.Embedding(T, 128)  # REPLACE 1024 WITH SOMETHING SMALLER

    def forward(self, t, x):
        x = x.unsqueeze(2)
        print(f"{x.size()=}")
        t_emb = self.t_embedding(t.int()).unsqueeze(0).unsqueeze(2)
        #input = x + t_emb
        torch.cat((x,t_emb), dim = 1)
        # x2 = self.linear_layer(input)
        input = torch.squeeze(input,2)
        print(f"{input.size()=}")
        x2 = self.linear(input)
        # x1 = self.encoder(input)  # CONCATENATE
        # print(f"{x1.size()=}")
        # x2 = self.decoder(x1)
        # x2 = torch.squeeze(x2, 0)
        print(f"{x2.size()=}")
        # Reshape the output without adaptive average pooling
        # x2_reshaped = x2.view(x2.size(0), -1)  # Reshape to [batch_size, 1024]
        #
        # Ensure the linear layer input size matches the reshaped output
        # x2_reshaped = self.linear_layer(x2)

        return x2


def reverse_max_pooling(pooled_output, indices, output_size):
    # Unpooling using indices
    unpool_output = torch.zeros_like(pooled_output)
    unpool_output.view(-1)[indices] = pooled_output.view(-1)

    # Upsampling to original size
    reversed_output = F.interpolate(unpool_output.unsqueeze(0).unsqueeze(0), size=output_size, mode='nearest')

    return reversed_output.squeeze()
#
# class ConvolutionalGAN(nn.Module):
#     def __init__(self, noise_dim, param_dims):
#         super().__init__()
#         # Encoder
#         self.model = nn.Sequential(
#             nn.Linear(noise_dim, 2*noise_dim),
#             nn.ReLU(),
#             # nn.Linear(2 * noise_dim, 2 * noise_dim),
#             # nn.ReLU(),
#             nn.Linear(2*noise_dim,int(param_dims))
#             # nn.Conv1d(1, 4, kernel_size=3,stride=1,padding=1),
#             # nn.ReLU(),
#             # nn.Conv1d(4, 4, kernel_size=3, stride=2, padding=1),
#             # nn.PReLU(init=1),
#             # nn.Conv1d(4, 4, kernel_size=3, stride=2, padding=1),
#             # nn.PReLU(init=1),
#             # nn.Conv1d(4, 8, kernel_size=3, stride=2, padding=1),
#             # nn.PReLU(init=1),
#             # nn.BatchNorm1d(4),
#             # nn.Conv1d(4, 2, kernel_size=4, stride=2, padding=1),
#             # nn.PReLU(init=1),
#             # nn.Conv1d(2, 2, kernel_size=4, stride=2, padding=1),
#             # nn.PReLU(init=1),
#             # nn.Conv1d(2, 1, kernel_size=4, stride=2, padding=1),
#             # nn.PReLU(init=1),
#             # nn.RReLU(),
#         )
#
#         self.linear = nn.Linear(8, int(param_dims))
#
#     def forward(self, x):
#         print(f"{x.size()=}")
#         # input = x
#         # print(f"{input.size()=}")
#         # x1 = self.model(input)
#         # print(f"{x1.size()=}")
#         x1 = x
#         x1 = x1.view(x1.size(0), -1)
#         x2 = self.model(x1)
#         x2_1, x2_2 = x2.chunk(2, dim = -1)
#         x2_2 = torch.abs(x2_2)
#         x2 = torch.cat((x2_1, x2_2), dim=-1)
#         return x2#.permute(0,1,3,2)
#
#
#
class ConvolutionalGAN(nn.Module):
    def __init__(self, noise_dim, param_dims):
        super().__init__()
        # Encoder
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 2*noise_dim),
            nn.ReLU(),
            # nn.Linear(2*noise_dim, 2 * noise_dim),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(3 * noise_dim, 3 * noise_dim),
            # nn.LeakyReLU(),
            # # nn.Dropout(p=0.5),
            # nn.Linear(3 * noise_dim, 3 * noise_dim),
            # nn.LeakyReLU(),
            # # nn.Dropout(p=0.5),
            # nn.Linear(3*noise_dim,int(param_dims))
            nn.Linear(2*noise_dim, int(param_dims))
        )

    def forward(self, x):
        # print(f"{x.size()=}")
        # input = x
        # print(f"{input.size()=}")
        # x1 = self.model(input)
        # print(f"{x1.size()=}")
        x1 = x
        # x1 = x1.view(x1.size(0), -1)
        x2 = self.model(x1)
        x2 = nn.LeakyReLU()(x2)
        # x2_1, x2_2 = x2.chunk(2, dim = -1)
        # x2_2 = torch.abs(x2_2)
        # x2 = torch.cat((x2_1, x2_2), dim=-1)
        return x2#.permute(0,1,3,2)



class DiscriminatorNetwork(nn.Module):
    def __init__(self, sample_size, param_dims):
        super().__init__()
        # Encoder
        self.model = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            # # nn.ReLU(),
            # nn.Linear(sample_size*param_dims, 10),
            # nn.Tanh(),
            # nn.Linear(sample_size, 1),
            # nn.Sigmoid()
        )
        self.linear = nn.Linear(int(sample_size * param_dims/2), 1)
        self.act = nn.Sigmoid()
    def forward(self, x):
        print(f"{x.size()=}")
        # input = x
        # print(f"{input.size()=}")
        # x1 = self.model(input)
        # print(f"{x1.size()=}")
        # x1 = x
        x1 = self.model(x.unsqueeze(1))
        x1 = x1.view(x.size(0), -1)
        x1 = self.linear(x1)
        x1 = self.act(x1)
        return x1#.permute(0,1,3,2)
#
# class DeepSet(nn.Module):
#     def __init__(self, input_dim=2, hidden_dim=8, output_dim=1):
#         super(DeepSet, self).__init__()
#         # Phi network
#         self.phi = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU()
#         )
#         # Rho network
#         self.rho = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         batch_size, input_size, input_dim = x.size()
#
#         # Reshape x to process all points in parallel
#         x = x.view(-1, input_dim)  # (batch_size * input_size, input_dim)
#         phi_x = self.phi(x)
#         # Reshape back to the original shape
#         phi_x = phi_x.view(batch_size, input_size, -1)
#         # phi_x = self.phi(x)  # shape (batch_size, set_size, hidden_dim)
#         # Aggregate using sum
#         aggregated = phi_x.mean(dim=1)  # shape (batch_size, hidden_dim)
#         # Apply rho to the aggregated result
#         output = self.rho(aggregated)  # shape (batch_size, output_dim)
#         return output

class DeepSet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8, output_dim=1):
        super(DeepSet, self).__init__()
        self.element_nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # # # nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.aggregation_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        batch_size, set_size, input_dim = x.size()

        x = x.view(-1, input_dim)  # Shape: (batch_size * set_size, input_dim)
        x = self.element_nn(x)  # Shape: (batch_size * set_size, hidden_dim)
        x = x.view(batch_size, set_size, -1)  # Shape: (batch_size, set_size, hidden_dim)

        x = x.sum(dim=1)  # Aggregation using sum (batch_size, hidden_dim)

        x = self.dropout(x)
        x = self.aggregation_nn(x)  # Shape: (batch_size, output_dim)

        return x

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Transpose for multihead attention (batch_first=False for nn.MultiheadAttention)
        x = x.transpose(0, 1)  # Shape: (set_size, batch_size, hidden_dim)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        return x.transpose(0, 1)  # Shape: (batch_size, set_size, hidden_dim)
class DeepSetsWithAttention(nn.Module):
    def __init__(self, batch_size=5,input_dim=2, hidden_dim=8, output_dim=1, num_heads=4):
        super(DeepSetsWithAttention, self).__init__()
        self.element_nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.self_attention = SelfAttention(hidden_dim, num_heads)
        self.aggregation_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        batch_size, set_size, input_dim = x.size()

        x = x.view(-1, input_dim)  # Shape: (batch_size * set_size, input_dim)
        x = self.element_nn(x)  # Shape: (batch_size * set_size, hidden_dim)
        x = x.view(batch_size, set_size, -1)  # Shape: (batch_size, set_size, hidden_dim)

        # x = self.self_attention(x)  # Shape: (batch_size, set_size, hidden_dim)

        x = x.sum(dim=1)  # Aggregation using sum (batch_size, hidden_dim)

        # x = self.dropout(x)
        x = self.aggregation_nn(x)  # Shape: (batch_size, output_dim)

        return x


# Discriminator Network
class PointNetDiscriminator(nn.Module):
    def __init__(self, log=False):
        self.log = log
        super(PointNetDiscriminator, self).__init__()
        self.pointnet = PointNetCls(k = 1)
        # self.fc1 = nn.Linear(128, 64)
        # self.fc2 = nn.Linear(64, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, pc):
        if self.log:
            pc = torch.log(pc)
        pc = pc.float()
        out, _, _ = self.pointnet(pc)
        # out = out.float()
        # nn.Dropout(p=0.5),
        # out = F.relu(self.fc1(features))
        # out = self.sigmoid(self.fc2(out))
        out = F.sigmoid(out)
        # out = out.data.max(1)
        # out = torch.argmax(out, dim=1)
        return out#.unsqueeze(-1)
class TNet(nn.Module):
    def __init__(self, k):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).view(-1))

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k).view(1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.tnet1 = TNet(k=2)  # Change k=2 for 2D point clouds
        self.conv1 = nn.Conv1d(2, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.tnet1(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        x = torch.sigmoid(x)  # Use sigmoid for binary classification
        return x
class SurrogatePhysicsModel(nn.Module):
    def __init__(self, input_dim):
        super(SurrogatePhysicsModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)        # Second fully connected layer
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()          # Sigmoid activation for binary classification

    def forward(self, x):
        x = x.float()
        x = torch.relu(self.fc1(x))          # Apply ReLU activation function
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))          # Apply ReLU activation function
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)                  # Apply Sigmoid activation function
        return x

class DeepSets(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=2):
        super(DeepSets, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Apply phi to each element in the set
        phi_x = self.phi(x)
        # Aggregate using max pooling for permutation invariance
        aggregated, _ = torch.max(phi_x, dim=1)
        # Apply rho to the aggregated features
        output = self.rho(aggregated)
        torch.sigmoid(output)
        return output


class PointDiscriminator(nn.Module):
    def __init__(self, loglog=False):
        super(PointDiscriminator, self).__init__()
        # Define your discriminator network architecture here
        self.fc1 = nn.Linear(in_features=2, out_features=8)  # Assuming 3D points
        self.fc2 = nn.Linear(in_features=8, out_features=1)  # Output a single scalar for each point

        self.sigmoid = nn.Sigmoid()
        self.loglog = False
    def forward(self, point_cloud):
        if self.loglog:
            point_cloud = torch.log(point_cloud+1e-10)
        # point_cloud shape: (batch_size, num_points, 3)
        batch_size, num_points, _ = point_cloud.size()

        # Flatten the point cloud to process each point individually
        x = point_cloud.view(batch_size * num_points, -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)

        # Reshape back to (batch_size, num_points, 1)
        x = x.view(batch_size, num_points, 1)

        # Aggregate classification across points (mean or sum)
        # Here, using mean to get a probability for the entire point cloud
        point_cloud_decision = torch.mean(x, dim=1)  # shape: (batch_size, 1)

        return point_cloud_decision

class MLP(nn.Module):
    def __init__(self, input_dim=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)  # First fully connected layer
        # self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 8)        # Second fully connected layer
        # self.bn2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()          # Sigmoid activation for binary classification

    def forward(self, x):
        # x_ratio = x[:, 0] * (1/ (x[:, 1] + 1e-8))  # Compute the ratio between x and y
        # x_ratio = x_ratio.unsqueeze(-1)  # Add an extra dimension for concatenation

        # x = torch.log(torch.cat([x, x_ratio], dim=-1))  # Concatenate the original data with the ratio
        x = x.float()
        # x = torch.log(x)
        x = torch.relu(self.fc1(x))
        # x = self.bn1(x)# Apply ReLU activation function
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))          # Apply ReLU activation function
        x = self.dropout(x)
        # x = self.bn2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)                  # Apply Sigmoid activation function
        return x