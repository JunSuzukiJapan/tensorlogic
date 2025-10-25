// Simplified GNN + KG Integration Demo
// Phase 14: GNN統合（簡略版）

main {
    print("======================================================================")
    print("   GNN + KNOWLEDGE GRAPH EMBEDDING (Simplified)")
    print("======================================================================")
    print("")

    print("🎯 Demonstrates GNN concepts for knowledge graphs")
    print("")

    // ========================================================================
    // Setup
    // ========================================================================
    print("--- Setup: Entity and Relation Embeddings ---")
    print("")

    let alice = zeros([8])
    let bob = zeros([8])
    let tokyo = zeros([8])
    let lives_in = zeros([8])
    let friend_of = zeros([8])

    print("Initialized 8-dimensional embeddings:")
    print("  Entities: alice, bob, tokyo")
    print("  Relations: lives_in, friend_of")
    print("")

    // ========================================================================
    // Part 1: Relational Message Passing (R-GCN)
    // ========================================================================
    print("======================================================================")
    print("PART 1: RELATIONAL MESSAGE PASSING (R-GCN)")
    print("======================================================================")
    print("")

    print("Concept: Different relations have different transformations")
    print("Formula: message = relation_emb * neighbor_emb")
    print("")

    print("Example: Alice receives messages from neighbors")
    print("")

    print("Message 1: From bob via friend_of relation")
    let msg_friend = relational_aggregate(alice, friend_of, bob)
    print("  Computed relational message")
    print("")

    print("Message 2: From tokyo via lives_in relation")
    let msg_location = relational_aggregate(alice, lives_in, tokyo)
    print("  Computed relational message")
    print("")

    print("Aggregate messages:")
    let combined_msg = msg_friend + msg_location
    print("  Combined:", combined_msg)
    print("")

    print("Apply activation (ReLU):")
    let alice_updated = relu(combined_msg)
    print("  Final embedding:", alice_updated)
    print("")

    // ========================================================================
    // Part 2: Feature Normalization
    // ========================================================================
    print("======================================================================")
    print("PART 2: FEATURE NORMALIZATION")
    print("======================================================================")
    print("")

    print("Why normalize?")
    print("  - Prevents gradient explosion/vanishing")
    print("  - Makes training more stable")
    print("  - Improves model generalization")
    print("")

    let alice_norm = normalize_features(alice, "l2")
    print("L2 normalized alice embedding:")
    print("  Result:", alice_norm)
    print("")

    let bob_norm = normalize_features(bob, "l2")
    print("L2 normalized bob embedding:")
    print("  Result:", bob_norm)
    print("")

    // ========================================================================
    // Part 3: Multi-Layer GNN
    // ========================================================================
    print("======================================================================")
    print("PART 3: MULTI-LAYER GNN CONCEPT")
    print("======================================================================")
    print("")

    print("Layer 1: Capture 1-hop neighborhood")
    let layer1_msg = relational_aggregate(alice, friend_of, bob)
    let layer1_out = relu(layer1_msg)
    print("  Layer 1 output computed")
    print("")

    print("Layer 2: Capture 2-hop neighborhood")
    print("  Uses layer1_out as input")
    let layer2_msg = relational_aggregate(layer1_out, lives_in, tokyo)
    let layer2_out = relu(layer2_msg)
    print("  Layer 2 output computed")
    print("")

    print("Interpretation:")
    print("  - Layer 1: Direct neighbors (1-hop)")
    print("  - Layer 2: Neighbors of neighbors (2-hop)")
    print("  - More layers → Larger receptive field")
    print("")

    // ========================================================================
    // Part 4: Applications
    // ========================================================================
    print("======================================================================")
    print("PART 4: GNN + KG APPLICATIONS")
    print("======================================================================")
    print("")

    print("1. ENHANCED LINK PREDICTION")
    print("   Problem: Predict (alice, ?, tokyo)")
    print("   Solution:")
    print("     a) Use GNN to refine alice & tokyo embeddings")
    print("     b) Consider graph structure (not just triples)")
    print("     c) Score with TransE/DistMult")
    print("   Benefit: Better accuracy by using neighborhood info")
    print("")

    print("2. NODE CLASSIFICATION")
    print("   Problem: Is this entity a person or place?")
    print("   Solution:")
    print("     a) GNN aggregates neighborhood features")
    print("     b) Classification based on refined embeddings")
    print("   Benefit: Leverage graph structure for classification")
    print("")

    print("3. MULTI-HOP REASONING")
    print("   Problem: Who are alice's colleagues?")
    print("   Path: alice -friend_of-> bob -works_at-> company")
    print("   Solution:")
    print("     a) Multi-layer GNN captures multi-hop paths")
    print("     b) Attention weights learn important paths")
    print("   Benefit: Answer complex queries")
    print("")

    print("4. KNOWLEDGE GRAPH COMPLETION")
    print("   Problem: Fill missing facts in KG")
    print("   Solution:")
    print("     a) GNN refines all entity embeddings")
    print("     b) Link prediction on refined embeddings")
    print("   Benefit: More complete knowledge graph")
    print("")

    // ========================================================================
    // Part 5: Workflow
    // ========================================================================
    print("======================================================================")
    print("PART 5: COMPLETE WORKFLOW")
    print("======================================================================")
    print("")

    print("Step 1: INITIALIZE")
    print("  - Entity embeddings from KG training")
    print("  - Relation embeddings from TransE/DistMult")
    print("")

    print("Step 2: GNN LAYERS")
    print("  for each layer:")
    print("    for each node:")
    print("      # Aggregate neighbor messages")
    print("      messages = []")
    print("      for neighbor, relation in neighbors:")
    print("        msg = relational_aggregate(node, relation, neighbor)")
    print("        messages.append(msg)")
    print("")
    print("      # Update node embedding")
    print("      aggregated = sum(messages)")
    print("      node_emb = relu(aggregated)")
    print("      node_emb = normalize(node_emb)")
    print("")

    print("Step 3: TASK-SPECIFIC HEAD")
    print("  - Link prediction: Use refined embeddings")
    print("  - Node classification: Add classifier layer")
    print("  - Graph classification: Global pooling")
    print("")

    print("Step 4: TRAINING")
    print("  - Loss: Task-specific (margin ranking, cross-entropy, etc.)")
    print("  - Optimizer: SGD, Adam")
    print("  - Backprop: Through all GNN layers")
    print("")

    // ========================================================================
    // Summary
    // ========================================================================
    print("======================================================================")
    print("SUMMARY")
    print("======================================================================")
    print("")

    print("✅ Demonstrated Concepts:")
    print("  ✓ Relational message passing (R-GCN)")
    print("  ✓ Feature normalization")
    print("  ✓ Multi-layer GNN")
    print("  ✓ Applications: Link prediction, classification, reasoning")
    print("")

    print("📊 Key Insights:")
    print("  - GNN captures graph structure")
    print("  - KG embeddings capture local patterns")
    print("  - Together: Best of both worlds!")
    print("")

    print("🎯 Implementation Functions:")
    print("  ✓ relational_aggregate() - R-GCN message passing")
    print("  ✓ normalize_features() - L2 normalization")
    print("  ✓ Combined with TransE/DistMult scoring")
    print("")

    print("✅ GNN + KG integration demo completed!")
}
