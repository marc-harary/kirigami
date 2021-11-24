#include <algorithm>
#include <vector>
#include <limits.h>
#include <string>
#include <set>
#include <pair>

enum Base { A = 'A', U = 'U', C = 'C', G = 'G' };
const std::set<Pair<Base,Base>> PAIRS = {{Base::A, Base::U},
                                         {Base::U, Base::A},
                                         {Base::C, Base::G},
                                         {Base::G, Base::C},
                                         {Base::G, Base::U},
                                         {Base::U, Base::G}};
using Sequence = std::vector<Base>;

class Zuker {
    const Sequence _S;
    static short minDist;
public:
    Zuker(Sequence);
    std::string getS() const;
    float W(int,int);
    float V(int,int);
};

static short Zuker::minDist = 4;

Zuker::Zuker(Sequence S) : _S(S) {
    ;
}

Sequence Zuker::getS() const { return _S; }

float Zuker::W(int i, int j) {
    if (j - i == minDist) return 0; 
    Base Si = S[i];
    Base Sj = S[j];
    
