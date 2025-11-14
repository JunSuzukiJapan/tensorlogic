// Check what token 6898 is

main {
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let empty = tokenize(tokenizer, "", false)
    let with_6898 = append(empty, 6898)
    let decoded = detokenize(tokenizer, with_6898, false)

    print("Token 6898:", decoded)
}
