#include <torch/extension.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <limits.h>

#define BASES "AUGC"
#define PAIRS std::set<std::string>({"AU", "UA", "CG", "GC", "GA", "AG"})

using namespace torch::indexing;
using PairProb = std::pair<double,std::pair<int,int>>;


std::string dense2sequence(torch::Tensor seq) {
    auto seq_ = seq.squeeze();
    int totalLength = seq_.size(1);
    int seqLength = seq_.sum().item<int>();
    int beg = (totalLength - seqLength) / 2;
    int end = beg + seqLength;
    auto outTuple = at::max(seq_.index({"...", Slice(beg,end,1)}), 0);
    std::string out(seqLength, ' ');
    for (int i = 0; i < seqLength; i++) {
        out[i] = BASES[i];
    }
    return out;
}
        

torch::Tensor binarize(torch::Tensor lab,
                       std::string seq,
                       int minDist = 4,
                       int thresPairs = INT_MAX,
                       double thresProb = 0.0,
                       bool symmetrize = true,
                       bool canonicalize = true) {
    auto lab_ = lab.squeeze();
    if (symmetrize) {
        lab_ += lab_.transpose(0, 1).clone();
        lab_ /= 2;
    }

    int seqLength = seq.size();
    int maxSize = lab_.size(0);
    int beg = (maxSize - seqLength) / 2;
    int end = beg + seqLength;

    double prob;
    int numPairs = 0;
    PairProb pairProbs[(seqLength-minDist) * (seqLength-minDist+1) / 2];
    for (int i = beg; i < end; i++) {
        for (int j = i + minDist; j < end; j++) {
            prob = lab_[i][j].item<double>();
            if (prob >= thresProb && (!canonicalize || PAIRS.find({seq[i-beg],seq[j-beg]}) != PAIRS.end())) {
                pairProbs[numPairs++] = {prob, {i,j}};
            }
        }
    }

    std::sort(pairProbs, pairProbs+numPairs, [](PairProb i, PairProb j) {
        return i.first > j.first;
    });

    auto out = torch::zeros({maxSize, maxSize}, lab.device());
    std::string dotBracket(seqLength, '.');

    std::pair<int,int> idxs;
    int j, k;
    PairProb curPair;
    int maxIters = std::min(numPairs, thresPairs);
    for (int i = 0; i < maxIters; i++) {
        curPair = pairProbs[i];
        prob = curPair.first;
        idxs = curPair.second;
        j = idxs.first;
        k = idxs.second;
        if (dotBracket[j-beg] != '.' || dotBracket[k-beg] != '.') {
            continue;
        }
        out[j][k] = out[k][j] = 1.;
        dotBracket[j-beg] = '(';
        dotBracket[k-beg] = ')';
    }

    while (out.dim() <  lab.dim()) {
        out.unsqueeze_(0);
    }

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("binarize",
          &binarize,
          py::arg("lab"),
          py::arg("seq"),
          py::arg("min_dist") = 4,
          py::arg("thres_pairs") = INT_MAX,
          py::arg("thres_prob") = 0.0,
          py::arg("symmetrize") = true,
          py::arg("canonicalize") = true);
}
