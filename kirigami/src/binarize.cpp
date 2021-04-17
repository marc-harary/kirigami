#include <torch/extension.h>
#include <iostream>
#include <algorithm>
#include <vector>

#define BASE_TENSORS torch::eye(4)
#define BASES "AUGC"
#define PAIRS std::set<std::string>({"AU", "UA", "CG", "GC", "GA", "AG"})
#define CANONICAL(i,j) PAIRS.find({i,j}) != PAIRS.end()
#define COMP_PAIR(i,j) i.first < j.first

using namespace torch::indexing;
using PairProb = std::pair<double,std::pair<int,int>>;


bool compPair(PairProb i, PairProb j) {
    return i.first > j.first;
}


char tensor2char(torch::Tensor a) {
    auto a_ = a.squeeze();
    a_ = a_.to(BASE_TENSORS[0].device());
    if (torch::equal(a_, BASE_TENSORS[0])) {
        return 'A';
    } else if (torch::equal(a_, BASE_TENSORS[1])) {
        return 'U';
    } else if (torch::equal(a_, BASE_TENSORS[2])) {
        return 'C';
    } else if (torch::equal(a_, BASE_TENSORS[3])) {
        return 'G';
    } else {
       throw std::invalid_argument("Base not valid");
    } 
}


std::string tensor2string(torch::Tensor sequence) {
    auto seq = sequence.squeeze();
    seq = seq.index({Slice(0,4,1), Slice(), 0});
    seq = seq.transpose(0, 1);
    int L = seq.size(0);
    std::string out(L, ' ');
    for (int i = 0; i < L; i++) {
        out[i] = tensor2char(seq[i]);
    }
    return out;
}
        

torch::Tensor binarize(torch::Tensor lab,
                       torch::Tensor seq,
                       double thres = 0.5,
                       int minDist = 4,
                       bool symmetrize = true,
                       bool canonicalize = true) {
    auto lab_ = lab.squeeze();
    if (symmetrize) {
        lab_ += lab_.transpose(0, 1);
        lab_ /= 2;
    }
    std::string seqStr = tensor2string(seq);

    int L = seqStr.size();
    PairProb pairProbs[(L-minDist) * (L-minDist+1) / 2];

    float prob;
    int numPairs = 0;
    for (int i = 0; i < L; i++) {
        for (int j = i + minDist; j < L; j++) {
            prob = lab_[i][j].item<double>();
            if (prob >= thres && CANONICAL(seqStr[i], seqStr[j])) {
                pairProbs[numPairs++] = {prob, {i,j}};
            }
        }
    }
    std::sort(pairProbs, pairProbs+numPairs, [](PairProb i, PairProb j) {
        return i.first > j.first;
    });

    auto out = torch::zeros_like(lab_, lab.device());
    std::string dotBracket(L, '.');


    std::pair<int,int> idxs;
    int j, k;
    PairProb curPair;
    for (int i = 0; i < numPairs; i++) {
        curPair = pairProbs[i];
        prob = curPair.first;
        idxs = curPair.second;
        j = idxs.first;
        k = idxs.second;
        if (dotBracket[j] != '.' || dotBracket[k] != '.') {
            continue;
        }
        out[j][k] = out[k][j] = 1.;
        dotBracket[j] = '(';
        dotBracket[k] = ')';
    }

    while (out.dim() <  lab.dim()) {
        out.unsqueeze_(0);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("binarize",
          &binarize,
          py::arg("label"),
          py::arg("sequence"),
          py::arg("thres") = 0.5,
          py::arg("min_dist") = 4,
          py::arg("symmetrize") = true,
          py::arg("canonicalize") = true);
}
