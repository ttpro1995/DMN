import torch
import torch.nn as nn
from torch.autograd import Variable as Var
import torch.nn.functional as F

class EpisodicMemoryModule(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim):
        super(EpisodicMemoryModule, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.z_dim = mem_dim*9
        self.n_episode = 2

        self.Wb = nn.Linear(mem_dim, mem_dim, bias=False)
        self.W1 = nn.Linear(self.z_dim, mem_dim)
        self.W2 = nn.Linear(mem_dim, mem_dim)
        self.memory_rnn = nn.GRUCell(in_dim, mem_dim)
        self.attention_rnn = nn.GRUCell(in_dim, mem_dim)

    def forward(self, c, q):
        m_prev = q
        for i in range(self.n_episode):
            e = self.episode_forward(c, m_prev, q)
            m = self.memory_rnn(e, m_prev)
            m_prev = m
        return m

    def episode_forward(self, c, m, q):
        h_prev = torch.zeros(self.mem_dim)
        for t in range(c.size(0)):
            c_t = c[t]
            z = self.make_z(c_t, m, q)
            g1 = F.tanh(self.W1(z))
            g2 = F.sigmoid(self.W2(g1))
            g_t = g2
            h_t = g_t* self.attention_rnn(c_t, h_prev) + (1-g_t)*h_prev
            h_prev = h_t
        e = h_t
        return e


    def make_z(self, c, m, q):
        cq = c.t() * self.Wb(q)
        cm = c.t() * self.Wb(m)
        l = [c, m, q, c * q, c * m, torch.abs(c - q), torch.abs(c - m), cq, cm]
        z = torch.cat(l, 1)
        return z


class InputModule(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim):
        super(InputModule, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.rnn = nn.GRU(in_dim, mem_dim)

    def forward(self, in_emb):
        c, _ = self.rnn(in_emb)
        return c


class QuestionModule(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim):
        super(InputModule, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.rnn = nn.GRU(in_dim, mem_dim)

    def forward(self, in_emb):
        _, q = self.rnn(in_emb)
        return q





