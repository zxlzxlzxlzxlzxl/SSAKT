import torch
from torch.utils.data import Dataset

class KTData(Dataset):
    def __init__(self, data, train, num_skills, max_seqlen, question=False,sepchar=','):
        self.data = 'data/{}/{}_{}.csv'.format(data,data,train)
        self.num_skills = num_skills
        self.max_seqlen = max_seqlen
        self.sepchar = sepchar
        self.question = question
        self.q_seq, self.qa_seq, self.target_seq, self.p_seq = self.load_data()

    def __getitem__(self, i):
        return self.q_seq[i], self.qa_seq[i], self.target_seq[i], self.p_seq[i]

    def __len__(self):
        return len(self.qa_seq)
        
    def load_data(self):
        q_seq = []
        qa_seq = []
        target_seq = []
        p_seq = []
        with open(self.data, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not self.question:
                    if line_num % 3 == 1:
                        qa = [int(i) for i in line.split(self.sepchar) if i!=""]
                        q = qa.copy()
                    elif line_num % 3 == 2:
                        target = [int(i) for i in line.split(self.sepchar) if i!=""]
                        start = 0
                        len_seq = len(qa)
                        if len_seq < 40:
                            continue
                        end = min(self.max_seqlen, len_seq)
                        while start < len_seq:
                            for ind in range(start, end):
                                qa[ind] += self.num_skills * target[ind]
                            if end - start < self.max_seqlen:
                                qa_seq.append(qa[start:end] + [0] * (self.max_seqlen + start - end))
                                target_seq.append(target[start:end] + [-1] * (self.max_seqlen + start - end))
                                q_seq.append(q[start:end] + [0] * (self.max_seqlen + start - end))
                            else:
                                qa_seq.append(qa[start:end] )
                                target_seq.append(target[start:end])
                                q_seq.append(q[start:end])
                            p_seq.append([0] * self.max_seqlen)
                            start = end
                            end = min(end + self.max_seqlen, len_seq)
                else:
                    if line_num % 4 == 1:
                        p = [int(i) for i in line.split(self.sepchar) if i!=""]
                    elif line_num % 4 == 2:
                        qa = [int(i) for i in line.split(self.sepchar) if i!=""]
                        q = qa.copy()
                    elif line_num % 4 == 3:
                        target = [int(i) for i in line.split(self.sepchar) if i!=""]
                        start = 0
                        len_seq = len(qa)
                        if len_seq < 40:
                            continue
                        end = min(self.max_seqlen, len_seq)
                        while start < len_seq:
                            for ind in range(start, end):
                                qa[ind] += self.num_skills * target[ind]
                            if end - start < self.max_seqlen:
                                qa_seq.append(qa[start:end] + [0] * (self.max_seqlen + start - end))
                                target_seq.append(target[start:end] + [-1] * (self.max_seqlen + start - end))
                                q_seq.append(q[start:end] + [0] * (self.max_seqlen + start - end))
                                p_seq.append(p[start:end] + [0] * (self.max_seqlen + start - end))
                            else:
                                qa_seq.append(qa[start:end] )
                                target_seq.append(target[start:end])
                                q_seq.append(q[start:end])
                                p_seq.append(p[start:end])
                            start = end
                            end = min(end + self.max_seqlen, len_seq)
        return torch.Tensor(q_seq).long(), torch.Tensor(qa_seq).long(), torch.Tensor(target_seq), torch.Tensor(p_seq).long()
