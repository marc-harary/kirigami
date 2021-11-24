#define NO_CONTACT -1 

using PairMap = std::vector<int>;

PairMap parseDotBracket(std::string line) {
    int j, length;
    length = line.size();
    std::deque<int> paren;
    std::deque<int> brack;
    std::vector<int> out;
    out.reserve(length);
    for (int i = 0; i < length; i++) {
        switch (line[i]) {
            case '.':
                out.push_back(NO_CONTACT);
                break;
            case '(': 
                out.push_back(NO_CONTACT);
                paren.push_back(i);
                break;
            case ')':
                j = paren.back();
                paren.pop_back();
                out.push_back(j);
                out[j] = i;
                break;
            case '[':
                out.push_back(NO_CONTACT);
                out[i] = NO_CONTACT;
                brack.push_back(i);
                break;
            case ']':
                j = brack.back();
                brack.pop_back();
                out.push_back(j);
                out[j] = i;
                break;
        }
    }
    return out;
}
         

std::tuple<std::string, PairMap>
st2pairmap(std::string file_name) {
    std::fstream file;
    file.open(file_name, std::ios::in);
    std::string line, sequence;
    // skip over header (which starts with #'s)
    do {
        getline(file, line);
    } while (line[0] == '#');
    // get sequence
    sequence = line;
    // get dot-bracket file  
    getline(file, line); 
    PairMap out = parseDotBracket(line); 
    return {sequence, out};
}



torch::Tensor pairMap2Tensor(PairMap pair_map, int out_dim = 3) {
    int length = pair_map.size();
    torch::Tensor out = torch::zeros({length,length});    
    for (int i = 0; i < length; i++) {
        int j = pair_map[i];
        if (j != NO_CONTACT) 
            out.index_put_({i,j}, torch::ones({1}));
    }
    while (out_dim > out.dim())
        out.unsqueeze_(0); 
    return out;
}
