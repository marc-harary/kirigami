std::string tensor2sequence(torch::Tensor ipt) { 
    auto chars_embed = ipt.squeeze();
    chars_embed = chars_embed[:N_BASES, :, 0].T
    std::string out;
    for (auto row: chars_embed) {
        _, idx = torch.max(row, 0);
        out += BASES[idx];
    }
    return out;
}
