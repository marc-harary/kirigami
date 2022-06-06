import torch
from torchmetrics import Metric



class GroundMCC(Metric):


    _pairs = {"AU", "UA", "CG", "GC", "GU", "UG"}


    def __init__(self):
        super().__init__()
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_prd", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_grd", default=torch.tensor(0), dist_reduce_fx="sum")


    # def update(self, prd_pairs: set, grd_pairs: set, seq_len: int) -> dict:
    def update(self, prd: torch.Tensor, grd: torch.Tensor):
        seq = self._fasta2str(grd)
        seq_len = len(seq)
        prd_pairs = self._prd2set(prd)
        grd_pairs = self._grd2set(grd)

        total = seq_len * (seq_len-1) / 2
        tp = float(len(prd_pairs.intersection(grd_pairs)))
        fp = len(prd_pairs) - tp
        fn = len(grd_pairs) - tp
        tn = total - tp - fp - fn

        self.n_prd += len(prd_pairs)
        self.n_grd += len(grd_pairs)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn


    def compute(self):
        mcc = f1 = 0. 
        if self.n_prd > 0 and self.n_grd > 0:
            sn = self.tp / (self.tp+self.fn)
            pr = self.tp / (self.tp+self.fp)
            if self.tp > 0:
                f1 = 2*sn*pr / (pr+sn)
            if (self.tp+self.fp) * (self.tp+self.fn) * (self.tn+self.fp) * (self.tn+self.fn) > 0:
                mcc = ((self.tp*self.tn-self.fp*self.fn) /
                      ((self.tp+self.fp)*(self.tp+self.fn)*(self.tn+self.fp)*(self.tn+self.fn))**.5)
            return mcc


    def _grd2set(self, ipt: torch.Tensor):
        """Converts ground truth contact `Tensor` to set of ordered pairs"""
        if isinstance(ipt, Tuple):
            ipt_ = ipt[0]
        else:
            ipt_ = ipt 
        ipt_ = ipt_.squeeze()
        idxs = torch.where(ipt_) 
        idxs = torch.vstack(idxs).T
        out = set()
        for i, j in idxs:
            if i < j:
                out.add((i.item(), j.item()))
        return out


    def _fasta2str(self, ipt: torch.Tensor) -> str:
        """Converts `Tensor` to FASTA string"""
        if isinstance(ipt, Tuple):
            ipt_ = ipt[0]
        else:
            ipt_ = ipt 
        ipt_ = ipt.squeeze()[:4,:,0].char()
        total_length = ipt_.shape[1]
        fasta_length = int(ipt_.sum().item())
        _, js = torch.max(ipt_, 0)
        return "".join("ACGU"[j] for j in js)


    def _prd2set(self, ipt: torch.Tensor, thres_pairs: int, seq: str):
        """Converts predicted contact `Tensor` to set of ordered pairs"""
        if isinstance(ipt, Tuple):
            ipt_ = ipt[0]
        else:
            ipt_ = ipt 
        ipt_ = ipt_.squeeze() 
        # get indices sorted in descending order by probability
        ipt_ = ipt_.flatten() # PyTorch makes sorting 2D tensors impractical
        idxs = ipt_.argsort(descending=True)
        # convert 1D index to 2D index using pointer arithmetic
        ii = idxs % len(seq)
        jj = torch.div(idxs, len(seq), rounding_mode="floor")
        kept = torch.zeros(len(seq), dtype=bool) # records indices already used
        out_set = set()
        num_pairs = 0
        for (i, j) in zip(ii, jj):
            if num_pairs == thres_pairs:
                break
            if (seq[i]+seq[j] in self.PAIRS and # canonical base pairs
                not kept[i] and not kept[j] and # not already
                i < j and # get upper triangular matrix
                j - i >= min_dist): # ensure i and j are at least min_dist apart
                    out_set.add((i.item(), j.item()))
                    kept[i] = kept[j] = True
                    num_pairs += 1
        return out_set
