import torch
from sklearn import metrics

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk        = max(topk)
    batch_size  = target.size(0)
    _, pred     = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred        = pred.t()
    correct     = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

class Preds():
    def __init__(self) -> None:
        self.count = 0
        self.logit_list = []
        self.pred_list = []
        self.gt_list = []
    
    def _cal_acc(self):
        return metrics.accuracy_score(self.gt_list, self.pred_list)

    def _cal_precision(self):
        return metrics.precision_score(self.gt_list, self.pred_list, average='macro')
    
    def _cal_recall(self):
        return metrics.recall_score(self.gt_list, self.pred_list, average='macro')
    
    def _cal_f1(self):
        return metrics.f1_score(self.gt_list, self.pred_list, average='macro')
    
    def _logit2num(self, logits):
        _, pred = logits.topk(1, dim=1, largest=True, sorted=True)
        return pred.view(-1).tolist()
    
    def _onehot2num(self, labels):
        _, gt = labels.topk(1, dim=1, largest=True, sorted=True)
        return gt.view(-1).tolist()
    
    def got_ll(self, logits, labels, istorch=True):
        if not istorch:
            pass
        self.pred_list.extend(self._logit2num(logits))
        self.gt_list.extend(self._onehot2num(labels))

    def cal_metrics(self):
        res_dict = {
            "acc" : self._cal_acc(),
            "preicision" : self._cal_precision(),
            "recall" : self._cal_recall(),
            "f1" : self._cal_f1()
        }
        return res_dict

    def clear_list(self):
        self.pred_list = []
        self.gt_list = []

if __name__ == '__main__':
    inputlogit = torch.rand(12, 5)
    targetlogit = torch.tensor([
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
    ])

    pr = Preds()
    pr.got_ll(inputlogit, targetlogit)
    print(pr.cal_metrics())