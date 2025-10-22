// TensorLogic Local LLM Chat Example
//
// このサンプルは、ローカルでGGUF形式のLLMを使ってChatGPTライクな
// 対話型チャットシステムを実装します。
//
// 使い方:
//   1. モデルのダウンロード:
//      cargo run --bin download_model -- --model tinyllama
//      または
//      cargo run --bin download_model -- --model phi2
//
//   2. チャットの起動:
//      tl run examples/local_llm_chat.tl
//
// モデルは ~/.tensorlogic/models/ に保存されます

// === 設定 ===

// 使用するモデル（~/.tensorlogic/models/以下のファイル名）
let model_name = "tinyllama-1.1b-chat-q4_0.gguf"
// または
// let model_name = "phi-2-q4_0.gguf"

// モデルのパスを構築
let home_dir = env("HOME")
let model_path = home_dir + "/.tensorlogic/models/" + model_name

// === モデルのロード ===

print("Loading model: " + model_path)
let model = load_model(model_path)
print("Model loaded successfully!")

// モデルメタデータを表示
print("Format: " + model.metadata.format)
print("Quantization: " + model.metadata.quantization)

// === トークナイザー設定 ===

// シンプルなトークナイザー（実際のプロダクションでは専用のトークナイザーが必要）
// ここでは文字ベースの簡易実装
function tokenize(text: string) -> int[] {
    // 簡易的な実装：文字を整数に変換
    // 実際には sentencepiece や tiktoken を使用
    let tokens = []
    for i in 0..text.length {
        let char_code = text.char_at(i).to_int()
        tokens.push(char_code)
    }
    return tokens
}

function detokenize(tokens: int[]) -> string {
    // トークンを文字列に戻す
    let text = ""
    for token in tokens {
        text += char_from_code(token)
    }
    return text
}

// === LLM推論関数 ===

function generate_response(
    model: Model,
    prompt: string,
    max_tokens: int = 100,
    temperature: float16 = 0.7
) -> string {
    // プロンプトをトークン化
    let input_tokens = tokenize(prompt)

    // 埋め込みテーブルを取得
    let embeddings = model.get_tensor("token_embd.weight")

    // 入力をテンソルに変換
    let input_ids = tensor<int>(input_tokens, device: gpu)

    // トークン埋め込みを取得
    let embedded = embeddings[input_ids, :]

    // 簡易的な生成ループ（実際のTransformer推論は複雑）
    // ここでは埋め込みの次元削減で次トークンを予測
    let generated_tokens = input_tokens

    for step in 0..max_tokens {
        // 最後の埋め込みから次のトークンを予測
        let last_embed = embedded[-1, :]

        // 出力層で語彙全体のロジットを計算（簡易版）
        let output_weight = model.get_tensor("output.weight")
        let logits = last_embed @ output_weight.transpose()

        // Temperature scaling
        let scaled_logits = logits / temperature

        // Softmax適用
        let probs = softmax(scaled_logits)

        // 確率が最も高いトークンを選択（greedy sampling）
        let next_token = argmax(probs)

        // EOSトークンチェック（トークンID 2は一般的にEOS）
        if next_token == 2 {
            break
        }

        generated_tokens.push(next_token)

        // 次の埋め込みを取得
        let next_embed = embeddings[next_token, :]
        embedded = concat([embedded, next_embed.unsqueeze(0)], dim: 0)
    }

    // 生成されたトークンを文字列に変換
    return detokenize(generated_tokens)
}

// === チャットシステム ===

print("\n=== TensorLogic Local LLM Chat ===")
print("Type 'exit' or 'quit' to end the chat")
print("Type 'clear' to clear chat history")
print("Type 'help' for available commands\n")

let chat_history = []
let system_prompt = "You are a helpful AI assistant."

// チャットループ
loop {
    // ユーザー入力
    print("You: ", end: "")
    let user_input = input()

    // コマンド処理
    if user_input == "exit" || user_input == "quit" {
        print("Goodbye!")
        break
    }

    if user_input == "clear" {
        chat_history = []
        print("Chat history cleared.")
        continue
    }

    if user_input == "help" {
        print("\nAvailable commands:")
        print("  exit, quit - Exit the chat")
        print("  clear - Clear chat history")
        print("  help - Show this help message")
        print("  /temp <value> - Set temperature (0.0-2.0)")
        print("  /tokens <value> - Set max tokens (1-500)")
        continue
    }

    // 温度設定
    if user_input.starts_with("/temp ") {
        let temp_str = user_input.substring(6)
        temperature = parse_float(temp_str)
        print("Temperature set to: " + temperature)
        continue
    }

    // 最大トークン数設定
    if user_input.starts_with("/tokens ") {
        let tokens_str = user_input.substring(8)
        max_tokens = parse_int(tokens_str)
        print("Max tokens set to: " + max_tokens)
        continue
    }

    // チャット履歴に追加
    chat_history.push({
        "role": "user",
        "content": user_input
    })

    // プロンプトを構築（チャット履歴を含む）
    let full_prompt = system_prompt + "\n\n"
    for message in chat_history {
        full_prompt += message.role + ": " + message.content + "\n"
    }
    full_prompt += "assistant: "

    // LLMで応答生成
    print("Assistant: ", end: "", flush: true)
    let response = generate_response(
        model,
        full_prompt,
        max_tokens: 100,
        temperature: 0.7
    )

    // 応答を表示（ストリーミング風に1文字ずつ）
    for char in response {
        print(char, end: "", flush: true)
        sleep(0.01)  // 10msの遅延でストリーミング効果
    }
    print("\n")

    // チャット履歴に追加
    chat_history.push({
        "role": "assistant",
        "content": response
    })
}

// === 統計情報表示 ===

print("\n=== Chat Statistics ===")
print("Total messages: " + chat_history.length)
print("Model: " + model_name)
print("Quantization: " + model.metadata.quantization)

// === 使用可能なモデル ===
//
// TinyLlama-1.1B-Chat (Q4_0: ~600MB)
//   - 高速、低メモリ
//   - 簡単な会話、質問応答
//   - ダウンロード: cargo run --bin download_model -- --model tinyllama
//
// Phi-2 (Q4_0: ~1.6GB)
//   - 高品質な応答
//   - コーディング支援、複雑な推論
//   - ダウンロード: cargo run --bin download_model -- --model phi2
//
// Mistral-7B (Q4_0: ~3.8GB)
//   - 最高品質
//   - プロフェッショナルな会話
//   - ダウンロード: cargo run --bin download_model -- --model mistral
//
// === カスタムモデルの使用 ===
//
// 手動でモデルをダウンロード:
//   1. HuggingFaceからGGUFモデルをダウンロード
//      https://huggingface.co/TheBloke
//   2. ~/.tensorlogic/models/ にコピー
//   3. model_name変数を更新

// === 高度な機能（将来の実装） ===
//
// - RAG（Retrieval-Augmented Generation）
// - Function Calling
// - マルチターンコンテキスト管理
// - トークンカウンター
// - コスト計算
// - チャット履歴の保存/ロード
// - ストリーミング応答
// - 複数モデルの切り替え
