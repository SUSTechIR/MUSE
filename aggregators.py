import torch
import torch.nn as nn
from abc import abstractmethod
import torch.nn.functional as F

class Aggregator(nn.Module):
    def __init__(self, batch_size, input_dim, output_dim, act, self_included, isAttension):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.self_included = self_included
        self.isAttension = isAttension
        self.w1 = nn.Linear(self.input_dim, self.input_dim)
        nn.init.xavier_uniform_(self.w1.weight)
        self.w2 = nn.Linear(self.input_dim, self.input_dim)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, self_vectors, neighbor_vectors, masks):
        # self_vectors: [batch_size, -1, input_dim]
        # neighbor_vectors: [batch_size, -1, 2, n_neighbor, input_dim]
        # masks: [batch_size, -1, 2, n_neighbor, 1]
        # 将边聚合到点上
        if(self.isAttension):
            entity_em = self_vectors[1]
            entity_em = entity_em.view((self.batch_size, -1, 2, 1, self.input_dim))
            attension_weight = F.leaky_relu(torch.sum(entity_em * neighbor_vectors ,dim=-1))
            attension_weight = F.softmax(attension_weight+(-1e9 * (1.0 - masks).squeeze(-1)), dim=-1)
            attension_weight = attension_weight.unsqueeze(-1)
            entity_vectors= torch.mean(neighbor_vectors * masks * attension_weight, dim=-2)
        else:
            # attension_weight = F.leaky_relu(torch.sum(self.w1(neighbor_vectors * masks) * self.w2(neighbor_vectors* masks), dim=-1))
            # attension_weight = F.softmax(attension_weight, dim=-1)
            # attension_weight = attension_weight.unsqueeze(-1)
            # entity_vectors = torch.mean(neighbor_vectors * masks * attension_weight, dim=-2)  # [batch_size, -1, 2, input_dim]
            entity_vectors = torch.mean(neighbor_vectors * masks, dim=-2)  # [batch_size, -1, 2, input_dim]
        outputs = self._call(self_vectors[0], entity_vectors)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]
        pass


class MeanAggregator(Aggregator):
    def __init__(self, batch_size, input_dim, output_dim, act=lambda x: x, self_included=True, isAttension=False):
        super(MeanAggregator, self).__init__(batch_size, input_dim, output_dim, act, self_included, isAttension)

        self.layer = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)

    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]

        output = torch.mean(entity_vectors, dim=-2)  # [batch_size, -1, input_dim]
        if self.self_included:
            self_vectors = self_vectors.view(self.batch_size,-1,self.input_dim)
            output += self_vectors
        output = output.view([-1, self.input_dim])  # [-1, input_dim]
        output = self.layer(output)  # [-1, output_dim]
        output = output.view([self.batch_size, -1, self.output_dim])  # [batch_size, -1, output_dim]

        return self.act(output)


class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, input_dim, output_dim, act=lambda x: x, self_included=True, isAttension = False):
        super(ConcatAggregator, self).__init__(batch_size, input_dim, output_dim, act, self_included, isAttension)

        multiplier = 3 if self_included else 2

        self.layer = nn.Linear(self.input_dim * multiplier, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)

    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]

        output = entity_vectors.view([-1, self.input_dim * 2])  # [-1, input_dim * 2]
        if self.self_included:
            self_vectors = self_vectors.view([-1, self.input_dim])  # [-1, input_dim]
            output = torch.cat([self_vectors, output], dim=-1)  # [-1, input_dim * 3]
        output = self.layer(output)  # [-1, output_dim]
        output = output.view([self.batch_size, -1, self.output_dim])  # [batch_size, -1, output_dim]

        return self.act(output)


class CrossAggregator(Aggregator):
    def __init__(self, batch_size, input_dim, output_dim, act=lambda x: x, self_included=True, isAttension = False):
        super(CrossAggregator, self).__init__(batch_size, input_dim, output_dim, act, self_included, isAttension)

        addition = self.input_dim if self.self_included else 0

        self.layer = nn.Linear(self.input_dim * self.input_dim + addition, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)

    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]

        # [batch_size, -1, 1, input_dim] 把两个entity_vector提取出来
        entity_vectors_a, entity_vectors_b = torch.chunk(entity_vectors, 2, dim=-2)
        entity_vectors_a = entity_vectors_a.view([-1, self.input_dim, 1])
        entity_vectors_b = entity_vectors_b.view([-1, 1, self.input_dim])
        output = torch.matmul(entity_vectors_a, entity_vectors_b)  # [-1, input_dim, input_dim]
        output = output.view([-1, self.input_dim * self.input_dim])  # [-1, input_dim * input_dim]
        if self.self_included:
            self_vectors = self_vectors.view([-1, self.input_dim])  # [-1, input_dim]
            output = torch.cat([self_vectors, output], dim=-1)  # [-1, input_dim * input_dim + input_dim]
        output = self.layer(output)  # [-1, output_dim]
        output = output.view([self.batch_size, -1, self.output_dim])  # [batch_size, -1, output_dim]

        return self.act(output)
