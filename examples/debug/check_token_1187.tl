// Check what token 1187 is

main {
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let empty = tokenize(tokenizer, "", false)
    let with_1187 = append(empty, 1187)
    let decoded = detokenize(tokenizer, with_1187, false)

    print("Token 1187:", decoded)
}
