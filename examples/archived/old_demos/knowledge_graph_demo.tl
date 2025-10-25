// 🎯 TensorLogic Knowledge Graph Demo
// Integrates all implemented features: Phase 4-10

// ============================================================================
// Phase 4: 型付きEntity宣言
// ============================================================================
entity Person
entity City
entity Item

// ============================================================================
// Phase 4: 型付き関係宣言
// ============================================================================
relation lives_in(person: Person, city: City)
relation owns(person: Person, item: Item)
relation friend_of(person1: Person, person2: Person)

// ============================================================================
// Phase 7: Entity Embedding（学習可能な密ベクトル表現）
// ============================================================================
embedding PersonEmbed {
    entities: {alice, bob, charlie, david}
    dimension: 16
    init: xavier
}

embedding CityEmbed {
    entities: {tokyo, osaka, kyoto, fukuoka}
    dimension: 8
    init: he
}

embedding ItemEmbed {
    entities: {book, laptop, phone, camera}
    dimension: 8
    init: random
}

main {
    print("╔═══════════════════════════════════════════════════╗")
    print("║  TensorLogic Knowledge Graph Demo                ║")
    print("║  統合機能: Phase 4-10                             ║")
    print("╚═══════════════════════════════════════════════════╝")
    print("")

    // ========================================================================
    // Phase 10: シンボルベースのファクト追加（クォート不要！）
    // ========================================================================
    print("📝 Adding facts (symbol-based syntax)...")

    lives_in(alice, tokyo)
    lives_in(bob, osaka)
    lives_in(charlie, kyoto)
    lives_in(david, fukuoka)

    owns(alice, book)
    owns(alice, laptop)
    owns(bob, phone)
    owns(charlie, camera)

    friend_of(alice, bob)
    friend_of(bob, charlie)
    friend_of(charlie, david)

    print("   ✓ 11 facts added")
    print("")

    // ========================================================================
    // Phase 5: エンティティ表示（with-block）
    // ========================================================================
    print("👥 Entity Collections:")
    print("")

    with Person {
        print("People in knowledge graph:")
    }
    print("")

    with City {
        print("Cities in knowledge graph:")
    }
    print("")

    // ========================================================================
    // Phase 8: メタタイプ（型を値として扱う）
    // ========================================================================
    print("🔢 Type Dimensions (meta-type system):")
    let person_dim = entity_dim(Person)
    let city_dim = entity_dim(City)
    let item_dim = entity_dim(Item)

    print("   Person dimension:", person_dim)
    print("   City dimension:", city_dim)
    print("   Item dimension:", item_dim)
    print("")

    // ========================================================================
    // Phase 6: One-hot変換（離散表現）
    // ========================================================================
    print("🎲 One-hot Encodings (discrete representation):")
    let alice_onehot = entity_onehot(Person, alice)
    print("   alice one-hot:", alice_onehot)

    let tokyo_onehot = entity_onehot(City, tokyo)
    print("   tokyo one-hot:", tokyo_onehot)
    print("")

    // ========================================================================
    // Phase 9: Embedding Lookup（密な学習可能表現）
    // ========================================================================
    print("🧠 Entity Embeddings (dense learned representations):")
    let alice_emb = PersonEmbed[alice]
    print("   alice embedding (16-dim):", alice_emb)

    let bob_emb = PersonEmbed[bob]
    print("   bob embedding (16-dim):", bob_emb)

    let tokyo_emb = CityEmbed[tokyo]
    print("   tokyo embedding (8-dim):", tokyo_emb)

    let book_emb = ItemEmbed[book]
    print("   book embedding (8-dim):", book_emb)
    print("")

    // ========================================================================
    // 統合: すべての機能を使ったクエリ例
    // ========================================================================
    print("🔍 Integrated Query Example:")
    print("   Who lives in tokyo?")
    with Person {
        lives_in(alice, tokyo)?
    }
    print("")

    print("   What does alice own?")
    with Item {
        owns(alice, book)?
        owns(alice, laptop)?
    }
    print("")

    // ========================================================================
    // まとめ
    // ========================================================================
    print("╔═══════════════════════════════════════════════════╗")
    print("║  ✅ Demo Completed!                              ║")
    print("╚═══════════════════════════════════════════════════╝")
    print("")
    print("実装された機能:")
    print("  ✓ Phase 4: 型付きEntity宣言")
    print("  ✓ Phase 5: エンティティ表示（with-block）")
    print("  ✓ Phase 6: Entity-Tensor変換（entity_onehot）")
    print("  ✓ Phase 7: Entity Embedding（学習可能な埋め込み）")
    print("  ✓ Phase 8: メタタイプシステム（型を値として）")
    print("  ✓ Phase 9: Embedding Lookup（密表現の取得）")
    print("  ✓ Phase 10: シンボルベース参照（統一構文）")
    print("")
    print("次のステップ:")
    print("  • 知識グラフ埋め込みの学習（TransE, DistMult等）")
    print("  • リレーション予測")
    print("  • グラフニューラルネットワークとの統合")
}
