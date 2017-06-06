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

        self.Wb = nn.Linear(mem_dim, 1, bias=False)
        self.W1 = nn.Linear(self.z_dim, mem_dim)
        self.W2 = nn.Linear(mem_dim, 1)
        self.memory_rnn = nn.GRUCell(mem_dim, mem_dim)
        self.attention_rnn = nn.GRUCell(mem_dim, mem_dim)

    def forward(self, c, q):
        m_prev = q
        for i in range(self.n_episode):
            e = self.episode_forward(c, m_prev, q)
            m = self.memory_rnn(e, m_prev)
            m_prev = m
        return m

    def episode_forward(self, c, m, q):
        h_prev = Var(torch.zeros(1, self.mem_dim))
        for t in range(c.size(0)):
            c_t = c[t]
            z = self.make_z(c_t, m, q)
            g1 = F.tanh(self.W1(z))
            g2 = F.sigmoid(self.W2(g1))
            g_t = g2
            h_t = g_t.mm(self.attention_rnn(c_t, h_prev)) + (1-g_t).mm(h_prev)
            h_prev = h_t
        e = h_t
        return e


    def make_z(self, c, m, q):
        # cq = c.t().mm(self.Wb(q))
        # cm = c.t().mm(self.Wb(m))
        cq = self.Wb(q).mm(c)
        cm = self.Wb(m).mm(c)
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
        super(QuestionModule, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.rnn = nn.GRU(in_dim, mem_dim)

    def forward(self, in_emb):
        _, q = self.rnn(in_emb)
        q = q.squeeze(0)
        return q


class AnswerModule(nn.Module):
    def __init__(self, cuda, mem_dim, out_dim):
        super(AnswerModule, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.out_dim = out_dim

        self.rnn = nn.GRUCell(out_dim, mem_dim)
        self.Wa = nn.Linear(mem_dim, out_dim)

    def forward(self, m):
        a_prev = m
        y_t = F.softmax(self.Wa(a_prev))
        return y_t




class DMN(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim, out_dim):
        super(DMN, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.input_module = InputModule(cuda, in_dim, mem_dim)
        self.question_module = QuestionModule(cuda, in_dim, mem_dim)
        self.memory_module = EpisodicMemoryModule(cuda, in_dim, mem_dim)
        self.answer_module = AnswerModule(cuda, mem_dim, out_dim)

    def forward(self, input_emb, question_emb):
        c = self.input_module(input_emb)
        q = self.question_module(question_emb)
        m = self.memory_module(c, q)
        y = self.answer_module(m)
        return y


def run_it():
    in_dim = 300
    mem_dim = 150
    out_dim = 3
    model = DMN(True, in_dim, mem_dim, out_dim)

    input_emb = Var(torch.rand(12, 1, 300))
    question_emb = Var(torch.rand(2, 1, 300))
    y = model(input_emb, question_emb)

if __name__ == '__main__':
    run_it()





