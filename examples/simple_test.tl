main {
    tensor mask: float16[2, 2] = causal_mask(2)
    print(mask)
}
