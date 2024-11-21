import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from utils import create_sequences_fun
from models import SimpleGRU, SimpleLSTM, SimpleTCN, mine_net

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class mine:
    def __init__(self, p_dis, q_dis, num_iterations, all = True, batch_size = 512,lr = 0.0001):
        self.lr = lr
        self.all = all
        '''
        ################
        Here we present our implementation of mine
        ################
        '''
        self.ma_window_size = 1#int(num_iterations/5)
        if p_dis.shape[0] < batch_size:
            self.batch_size = p_dis.shape[0]
        else:
            self.batch_size = batch_size

        if not isinstance(p_dis, torch.Tensor):
            self.obs = torch.tensor(p_dis, dtype=torch.float32)
        else:
            self.obs = p_dis.float()
        if not isinstance(q_dis, torch.Tensor):
            self.acs = torch.tensor(q_dis, dtype=torch.float32)
        else:
            self.acs = q_dis.float()
        
        if self.obs.dim() == 1:
            self.obs = self.obs.unsqueeze(1)
        if self.acs.dim() == 1:
            self.acs = self.acs.unsqueeze(1)
        
        self.num_iterations = num_iterations

        self.expts =3

    def kullback_liebler(self, dis_p, dis_q, kl_net):
        t = kl_net(dis_p)
        et = torch.exp(kl_net(dis_q))
        mi_lb = torch.mean(t) - torch.log(torch.mean(et)) 
        return mi_lb, t, et

    def learn_klne(self, batch, mine_net, mine_net_optim, ma_et, ma_rate=0.001):
        joint, marginal = batch
        joint = torch.autograd.Variable(torch.FloatTensor(joint)).to(device)
        marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).to(device)
        mi_lb, t, et = self.kullback_liebler(joint, marginal, mine_net)
        ma_et = (1-ma_rate) * ma_et + ma_rate * torch.mean(et)
        loss = -(torch.mean(t) - (1/ma_et.mean()).detach() * torch.mean(et))
        mine_net_optim.zero_grad()
        autograd.backward(loss)
        mine_net_optim.step()
        return mi_lb, ma_et
   
    def trip_sample_batch(self, sample_mode='joint'):
        index = np.random.choice(range(self.obs.shape[0]), size=self.batch_size, replace=False)
        if sample_mode == 'marginal':
            marginal_index = np.random.choice(range(self.obs.shape[0]), size=self.batch_size, replace=False)
            
            batch = np.concatenate((self.obs[index, :],np.array(self.acs[marginal_index, :])),axis=1)
        else:
            batch = np.concatenate((self.obs[index, :],np.array(self.acs[ index, :])),axis=1)
        
        return batch

    def trip_train(self, tripmine_net, tripmine_net_optim):
        ma_et = 1.
        result = list()
        for i in range(self.num_iterations):
            batch = self.trip_sample_batch(), self.trip_sample_batch(sample_mode='marginal') 
            mi_lb, ma_et = self.learn_klne(batch, tripmine_net, tripmine_net_optim, ma_et)
            result.append(mi_lb.detach().cpu().numpy())
        return result

    def ma(self, a):
        return [np.mean(a[i:i+self.ma_window_size]) for i in range(0, len(a)-self.ma_window_size)]

    def trip_initialiser(self):
        tripmine_net = mine_net(self.obs.shape[1]+self.acs.shape[1]).to(device)
        tripmine_net_optim = optim.Adam(tripmine_net.parameters(), lr=self.lr)
        trip_results = list()
        for expt in range(self.expts):
            trip_results.append(self.ma(self.trip_train( tripmine_net, tripmine_net_optim)))
        return np.array(trip_results)

    
    def run(self):
        results = self.trip_initialiser()
        return results

class mine_fa:
    def __init__(self, p_dis, q_dis, num_iterations, all=True, batch_size=512, lr=0.0001):
        self.lr = lr
        self.all = all
        '''
        ################
        Here we present an alternative method for estimating the mi
        which measures the reduction in uncertainty of a classifier (this is quicker)
        ################
        '''
        if not isinstance(p_dis, torch.Tensor):
            p_dis = torch.tensor(p_dis, dtype=torch.float32)
        if not isinstance(q_dis, torch.Tensor):
            q_dis = torch.tensor(q_dis, dtype=torch.float32)
        self.batch_size = min(batch_size, p_dis.shape[0])
        self.num_iterations = num_iterations
        self.p_dis= p_dis
        self.q_dis = q_dis
        if len(self.p_dis.shape) == 1:
            self.p_dis = self.p_dis.unsqueeze(1)
        if len(self.q_dis.shape) == 1:
            self.q_dis = self.q_dis.unsqueeze(1)
        self.expts = 3

    def create_regressor(self):
        return  nn.Sequential(
            nn.Linear(self.p_dis.shape[1], 10), nn.ReLU(),
            nn.Linear(10, 10), nn.ReLU(),
            nn.Linear(10, len(np.unique(self.q_dis.squeeze()))),
                nn.LogSoftmax(dim=1)
        ).to(device)

    def fit_mlp(self):
        '''
        Fit a regressor (this measures our maximum predictive power)
        '''
        p_dis = self.p_dis
        q_dis = self.q_dis.long()
        criterion = nn.NLLLoss()
        mlp_regressor = self.create_regressor().train()
        optimizer = optim.SGD(mlp_regressor.parameters(), lr=self.lr, momentum=0.9)
        losses_feats = []
        for epoch in range(self.num_iterations):
            
            
            permutation = torch.randperm(p_dis.size()[0])
            losses = []
            for i in range(0, p_dis.size()[0], self.batch_size):
                indices = permutation[i:i+self.batch_size]
                if len(indices) == self.batch_size:
                    batch_p_dis = p_dis[indices].to(device)
                    batch_q_dis = q_dis[indices].to(device)
                    predictions = mlp_regressor(batch_p_dis)
                    loss = criterion(predictions, batch_q_dis.squeeze() if len(batch_q_dis.shape) > 1 else batch_q_dis)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

            losses_feats.append(np.array(losses).mean())
            if (epoch+1)%25 == 0:
                print(f"Epoch {epoch+1}: Average Loss: {losses_feats[-1]}")
        '''
        Fit a null regressor (this measures our minimum predictive power)
        '''
        losses_no_feats = []
        mlp_regressor = self.create_regressor().train()
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(mlp_regressor.parameters(), lr=self.lr, momentum=0.9)

        for epoch in range(self.num_iterations):
            permutation = torch.randperm(p_dis.size()[0])
            losses = []
            for i in range(0, p_dis.size()[0], self.batch_size):
                indices = permutation[i:i+self.batch_size]
                if len(indices) == self.batch_size:
                    batch_q_dis = q_dis[indices].to(device)

                    optimizer.zero_grad()
                    predictions = mlp_regressor(torch.zeros_like(batch_p_dis))
                    loss = criterion(predictions, batch_q_dis.squeeze() if len(batch_q_dis.shape) > 1 else batch_q_dis)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

            losses_no_feats.append(np.array(losses).mean())
            if (epoch+1)%25 == 0:
                print(f"Epoch {epoch+1}: Average Loss with no features: {losses_no_feats[-1]}")

        return np.array(losses_no_feats) - np.array(losses_feats)
    
    def ma(self, a):
        return [np.mean(a[i:i+int(len(a)/10)]) for i in range(0, len(a)-int(len(a)/10))]

    def run(self):
        trip_results = list()
        for expt in range(self.expts):
            trip_results.append((self.fit_mlp()))
        return np.array(trip_results)
    
class mine_td:
    def __init__(self, p_dis, q_dis, num_iterations, model_type='LSTM', all=True, batch_size=50, lr=0.001, sequence_length=10):
        self.lr = lr
        self.all = all
        self.sequence_length = sequence_length
        self.model_type = model_type

        '''
        ################
        Integrated MINE with Temporal Dependence
        ################
        '''
        self.ma_window_size = 1#int(num_iterations / 5)
        self.batch_size = min(batch_size, p_dis.shape[0])
        self.num_iterations = num_iterations

        # Convert distributions to torch tensors and ensure they are 2D
        if not isinstance(p_dis, torch.Tensor):
            self.p_dis = torch.tensor(p_dis, dtype=torch.float32)
        else:
            self.p_dis = p_dis.float()
        if not isinstance(q_dis, torch.Tensor):
            self.q_dis = torch.tensor(q_dis, dtype=torch.float32)
        else:
            self.q_dis = q_dis.float()
        
        if self.p_dis.dim() == 1:
            self.p_dis = self.p_dis.unsqueeze(1)
        if self.q_dis.dim() == 1:
            self.q_dis = self.q_dis.unsqueeze(1)
        
        self.expts = 3  # Number of experiments

        # Initialize the MINE network with temporal model
        self.mine_net = self.create_model().to(device)
        self.mine_net_optim = optim.Adam(self.mine_net.parameters(), lr=self.lr)
    
    def create_model(self):
        # Mapping of model types to their corresponding classes
        model_classes = {
            'LSTM': SimpleLSTM,
            'GRU': SimpleGRU,
            'TCN': SimpleTCN
        }

        # Check if the provided model_type is supported
        if self.model_type not in model_classes:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Supported types are: {list(model_classes.keys())}")

        # Select the appropriate model class
        ModelClass = model_classes[self.model_type]

        # Determine the output size based on unique values in q_dis
        output_size = len(torch.unique(self.q_dis.squeeze()))

        # Instantiate and return the model moved to the appropriate device
        model = ModelClass(self.p_dis.shape[1]+ self.q_dis.shape[1], 250, output_size)
        return model.to(device)
    
    def create_sequences(self, X, y):
        return create_sequences_fun(X, y, self.sequence_length)
    
    def kullback_liebler(self, dis_p, dis_q, kl_net):
        t = kl_net(dis_p)
        et = torch.exp(kl_net(dis_q))
        mi_lb = torch.mean(t) - torch.log(torch.mean(et)) 
        return mi_lb, t, et

    def learn_klne(self, batch, ma_et, ma_rate=0.001):
        joint, marginal = batch
        joint = joint.to(device)
        marginal = marginal.to(device)
        mi_lb, t, et = self.kullback_liebler(joint, marginal, self.mine_net)
        ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et).detach()
        loss = -(torch.mean(t) - (1 / ma_et) * torch.mean(et))
        self.mine_net_optim.zero_grad()
        loss.backward()
        self.mine_net_optim.step()
        return mi_lb, ma_et

    def trip_sample_batch(self, sample_mode='joint'):
        if sample_mode == 'marginal':
            indices_p = np.random.choice(range(self.p_dis.shape[0]), size=self.batch_size, replace=False)
            indices_q = np.random.choice(range(self.q_dis.shape[0]), size=self.batch_size, replace=False)
        else:
            indices_p = np.random.choice(range(self.p_dis.shape[0]), size=self.batch_size, replace=False)
            indices_q = indices_p

        batch_p = self.p_dis[indices_p]
        batch_q = self.q_dis[indices_q]

        # Create sequences
        batch_p_seq, _ = self.create_sequences(batch_p, batch_p)
        batch_q_seq, _ = self.create_sequences(batch_q, batch_q)

        return batch_p_seq, batch_q_seq
    
    def trip_train(self, tripmine_net, ma_et):
        result = list()
        for i in range(self.num_iterations):
            batch = self.trip_sample_batch(sample_mode='joint'), self.trip_sample_batch(sample_mode='marginal') 
            # Flatten the batch tuples
            joint_batch_p, joint_batch_q = batch[0]
            marginal_batch_p, marginal_batch_q = batch[1]
            # Concatenate along the batch dimension
            joint_batch = torch.cat((joint_batch_p, joint_batch_q), dim=2)  # Assuming feature dimension is last
            marginal_batch = torch.cat((marginal_batch_p, marginal_batch_q), dim=2)
            mi_lb, ma_et = self.learn_klne((joint_batch, marginal_batch), ma_et)
            result.append(mi_lb.detach().cpu().numpy())
            
        return result

    def ma_func(self, a):
        return [np.mean(a[i:i+self.ma_window_size]) for i in range(0, len(a)-self.ma_window_size)]
    
    def trip_initialiser(self):
        trip_results = list()
        for expt in range(self.expts):
            ma_et = 1.0
            trip_results.append(self.ma_func(self.trip_train(self.mine_net, ma_et)))
        return np.array(trip_results)

    def run(self):
        results = self.trip_initialiser()
        return results

    
class mine_fa_td:
    def __init__(self, p_dis, q_dis, num_iterations, model_type, all=True, batch_size=100, lr=0.0001):
        self.lr = lr
        self.all = all

        # Convert p_dis and q_dis to torch tensors if they aren't already
        if not isinstance(p_dis, torch.Tensor):
            p_dis = torch.tensor(p_dis, dtype=torch.float32)
        if not isinstance(q_dis, torch.Tensor):
            q_dis = torch.tensor(q_dis, dtype=torch.float32)
        
        # Set batch size to the smaller of the provided batch_size or the size of p_dis
        self.batch_size = min(batch_size, p_dis.shape[0])
        self.num_iterations = num_iterations
        self.p_dis = p_dis
        self.q_dis = q_dis
        self.model_type = model_type

        # Ensure p_dis and q_dis are at least 2D
        if self.p_dis.dim() == 1:
            self.p_dis = self.p_dis.unsqueeze(1)
        if self.q_dis.dim() == 1:
            self.q_dis = self.q_dis.unsqueeze(1)
        
        self.expts = 3  # Assuming this is used elsewhere in your class

    def create_model(self):
        # Mapping of model types to their corresponding classes
        model_classes = {
            'LSTM': SimpleLSTM,
            'GRU': SimpleGRU,
            'TCN': SimpleTCN
        }

        # Check if the provided model_type is supported
        if self.model_type not in model_classes:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Supported types are: {list(model_classes.keys())}")

        # Select the appropriate model class
        ModelClass = model_classes[self.model_type]

        # Determine the output size based on unique values in q_dis
        # It's more efficient and consistent to use torch.unique instead of numpy
        output_size = len(torch.unique(self.q_dis.squeeze()))

        # Instantiate and return the model moved to the appropriate device
        model = ModelClass(self.p_dis.shape[1], 250, output_size)
        return model.to(device)
    
    def create_sequences(self, X, y, sequence_length):
        return create_sequences_fun(X, y, sequence_length)

    def fit_lstm(self):
        indexes = np.random.choice(self.q_dis.shape[0], int(self.q_dis.shape[0]*0.1), replace=False).tolist()
        p_dis = self.p_dis[indexes, :]
        q_dis = self.q_dis[indexes, :]
        self.model = self.create_model().to(device)
        losses_feats = []
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        for epoch in range(self.num_iterations):
            
            
            losses = []
            for i in range(0, p_dis.size()[0], self.batch_size):
                    indices = list(range(i, min(i + self.batch_size, p_dis.size()[0])))
                    if len(indices) < self.batch_size:
                        continue  # Skip batch if it's not full size (optional based on model requirements)
                    batch_p_dis = p_dis[indices]
                    batch_q_dis = q_dis[indices]

                    batch_p_dis, batch_q_dis = self.create_sequences(batch_p_dis, batch_q_dis, 10)
                    optimizer.zero_grad()
                    predictions = self.model(batch_p_dis.to(device))
                    loss = criterion(predictions, batch_q_dis.squeeze().to(device))
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())
                    break
            losses_feats.append(np.array(losses).mean())
        losses_no_feats = []
        self.model = self.create_model().to(device)
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)

        for epoch in range(self.num_iterations):
            
            
            losses = []
            for i in range(0, p_dis.size()[0], self.batch_size):
                    indices = list(range(i, min(i + self.batch_size, p_dis.size()[0])))
                    if len(indices) < self.batch_size:
                        continue  # Skip batch if it's not full size (optional based on model requirements)
                    batch_p_dis = p_dis[indices]
                    batch_q_dis = q_dis[indices]
                    batch_p_dis, batch_q_dis = self.create_sequences(batch_p_dis, batch_q_dis, 10)
                    optimizer.zero_grad()
                    predictions = self.model(batch_p_dis.to(device))
                    loss = criterion(predictions, batch_q_dis.squeeze().to(device))
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())
                    break
            losses_no_feats.append(np.array(losses).mean())

        return np.array(losses_no_feats) - np.array(losses_feats)

    
    def ma(self, a):
        return [np.mean(a[i:i+int(len(a)/10)]) for i in range(0, len(a)-int(len(a)/10))]

    def run(self):
        trip_results = list()
        for expt in range(self.expts):
            trip_results.append((self.fit_lstm()))
        return np.array(trip_results)