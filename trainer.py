from tqdm import tqdm
import torch
from torch.autograd import Variable as Var
from utils import map_label_to_target, map_label_to_target_sentiment
import torch.nn.functional as F
from metrics import SubtreeMetric

class SentimentTrainer(object):
    """
    For Sentiment module
    """
    def __init__(self, args, model, embedding_model ,criterion, optimizer):
        super(SentimentTrainer, self).__init__()
        self.args       = args
        self.model      = model
        self.embedding_model = embedding_model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0
        self.emb_params_init = None
        self.question = None

    def set_initial_emb(self, emb):
        self.emb_params_init = emb

    def set_question(self, indices):
        self.question = indices

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.embedding_model.train()
        self.embedding_model.zero_grad()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        # torch.manual_seed(789)
        indices = torch.randperm(len(dataset))
        for idx in tqdm(xrange(len(dataset)),desc='Training epoch '+str(self.epoch+1)+''):
            tree, sent, label = dataset[indices[idx]]
            input = Var(sent)
            question = Var(self.question.long())
            target = Var(map_label_to_target_sentiment(label,self.args.num_classes, fine_grain=self.args.fine_grain))
            if self.args.cuda:
                input = input.cuda()
                question = question.cuda()
                target = target.cuda()
            emb = F.torch.unsqueeze(self.embedding_model(input), 1)
            question_emb = F.torch.unsqueeze(self.embedding_model(question), 1)


            output, err, n_subtrees = self.model(tree, emb, question_emb, training=True)
            batch_size = self.args.batchsize * n_subtrees


            if self.args.reg > 0 or self.args.embreg > 0:
                params = self.model.getParameters()
                params_norm = params.norm()
                l2_model = 0.5*self.args.reg*params_norm*params_norm
                emb_params = list(self.embedding_model.parameters())[0]
                emb_init = Var(self.emb_params_init, requires_grad = False)
                emb_params_norm = (emb_params - emb_init).norm()
                l2_emb_params = 0.5 * self.args.embreg* emb_params_norm * emb_params_norm
                if l2_emb_params.data[0] > 0:
                    err = (err + l2_model + l2_emb_params) / batch_size
                else:
                    err = (err + l2_model ) / batch_size
            else:
                err = err / batch_size

            # err = err / self.args.batchsize
            #params = self.model.childsumtreelstm.getParameters()
            # params_norm = params.norm()


            loss += err.data[0] #
            err.backward()
            k += 1
            if k==self.args.batchsize:
                if self.args.manually_emb == 1:
                    if self.args.embwd == 0: # save time on calculate 0 function
                        for f in self.embedding_model.parameters():
                            f.data.sub_(f.grad.data * self.args.emblr)
                    else:
                        for f in self.embedding_model.parameters():
                            f.data.sub_(f.grad.data * self.args.emblr + self.args.emblr*self.args.embwd*f.data)

                    # https://stats.stackexchange.com/questions/29130/difference-between-neural-net-weight-decay-and-learning-rate
                self.optimizer.step()
                self.embedding_model.zero_grad()
                self.optimizer.zero_grad()
                k = 0
        self.epoch += 1
        return loss/len(dataset)

    # helper function for testing
    def test(self, dataset):
        subtree_metric = SubtreeMetric()
        self.model.eval()
        self.embedding_model.eval()
        loss = 0
        predictions = torch.zeros(len(dataset))
        predictions = predictions
        for idx in tqdm(xrange(len(dataset)),desc='Testing epoch  '+str(self.epoch)+''):
            tree, sent, label = dataset[idx]
            input = Var(sent, volatile=True)
            question = Var(self.question.long(), volatile=True)
            target = Var(map_label_to_target_sentiment(label,self.args.num_classes, fine_grain=self.args.fine_grain), volatile=True)
            if self.args.cuda:
                input = input.cuda()
                target = target.cuda()
                question = question.cuda()
            emb = F.torch.unsqueeze(self.embedding_model(input),1)
            question_emb = F.torch.unsqueeze(self.embedding_model(question), 1)
            output, _, _= self.model(tree, emb, question_emb, training = False) # size(1,5)
            err = self.criterion(output, target)
            loss += err.data[0]
            if self.args.num_classes == 3:
                output[:,1] = -9999 # no need middle (neutral) value
            val, pred = torch.max(output, 1)
            pred_cpu = pred.data.cpu()[0][0]
            predictions[idx] = pred_cpu
            correct = pred_cpu == tree.gold_label
            subtree_metric.current_idx = idx
            subtree_metric.count_depth(correct, 0, tree.idx, pred_cpu)
            # predictions[idx] = torch.dot(indices,torch.exp(output.data.cpu()))
        return loss/len(dataset), predictions, subtree_metric


class SimilarityTrainer(object):
    def __init__(self, args, model, embedding_model, criterion, optimizer):
        super(SimilarityTrainer, self).__init__()
        self.args       = args
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0
        self.embedding_model = embedding_model

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.embedding_model.train()
        self.embedding_model.zero_grad()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        indices = torch.randperm(len(dataset))
        for idx in tqdm(xrange(len(dataset)),desc='Training epoch '+str(self.epoch+1)+''):
            ltree,lsent,rtree,rsent,label = dataset[indices[idx]]
            linput, rinput = Var(lsent), Var(rsent)
            target = Var(map_label_to_target(label,dataset.num_classes))
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            lemb = torch.unsqueeze(self.embedding_model(linput), 1)
            remb = torch.unsqueeze(self.embedding_model(rinput), 1)
            output = self.model(ltree, lemb, rtree, remb)
            err = self.criterion(output, target)
            loss += err.data[0]
            err.backward()
            k += 1
            if k==self.args.batchsize:
                for f in self.embedding_model.parameters():
                    f.data.sub_(f.grad.data * self.args.emblr)
                self.optimizer.step()
                self.embedding_model.zero_grad()
                self.optimizer.zero_grad()
                k = 0
        self.epoch += 1
        return loss/len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        loss = 0
        predictions = torch.zeros(len(dataset))
        indices = torch.range(1,dataset.num_classes)
        for idx in tqdm(xrange(len(dataset)),desc='Testing epoch  '+str(self.epoch)+''):
            ltree,lsent,rtree,rsent,label = dataset[idx]
            linput, rinput = Var(lsent, volatile=True), Var(rsent, volatile=True)
            target = Var(map_label_to_target(label,dataset.num_classes), volatile=True)
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            lemb = torch.unsqueeze(self.embedding_model(linput), 1)
            remb = torch.unsqueeze(self.embedding_model(rinput), 1)
            output = self.model(ltree,lemb,rtree,remb)
            err = self.criterion(output, target)
            loss += err.data[0]
            predictions[idx] = torch.dot(indices,torch.exp(output.data.cpu()))
        return loss/len(dataset), predictions
