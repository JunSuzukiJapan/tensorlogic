// GNN and Knowledge Graph Embedding Integration
// Phase 14.1: GNNメッセージパッシング基礎

main {
    print("======================================================================")
    print("   GNN + KNOWLEDGE GRAPH EMBEDDING INTEGRATION")
    print("======================================================================")
    print("")

    print("🎯 Goal: Combine GNN message passing with KG embeddings")
    print("")

    // ========================================================================
    // Setup: Knowledge Graph Structure
    // ========================================================================
    print("--- Knowledge Graph Structure ---")
    print("")
    print("Entities:")
    print("  - alice, bob, charlie")
    print("  - tokyo, osaka")
    print("  - company_a, company_b")
    print("")
    print("Relations:")
    print("  - lives_in: (alice, tokyo), (bob, osaka)")
    print("  - works_at: (alice, company_a), (charlie, company_b)")
    print("  - friend_of: (alice, bob), (bob, charlie)")
    print("")

    // Create node embeddings (entities)
    let alice = zeros([8])
    let bob = zeros([8])
    let charlie = zeros([8])
    let tokyo = zeros([8])
    let osaka = zeros([8])
    let company_a = zeros([8])

    // Create relation embeddings
    let lives_in = zeros([8])
    let works_at = zeros([8])
    let friend_of = zeros([8])

    print("Initialized 8-dimensional embeddings")
    print("")

    // ========================================================================
    // Part 1: Simple Message Passing
    // ========================================================================
    print("======================================================================")
    print("PART 1: SIMPLE MESSAGE PASSING (GCN-style)")
    print("======================================================================")
    print("")

    print("Task: Update alice's embedding using neighbor information")
    print("")

    print("Alice's neighbors:")
    print("  - bob (friend_of)")
    print("  - tokyo (lives_in)")
    print("  - company_a (works_at)")
    print("")

    // Aggregate neighbor features
    print("Step 1: Aggregate neighbor features")
    let aggregated = aggregate_neighbors(bob, 3, "mean")
    print("  Aggregated features from 3 neighbors")
    print("  Aggregation type: mean")
    print("")

    print("Step 2: Combine with alice's features")
    let alice_updated = alice + aggregated
    print("  Updated embedding: alice_old + aggregated_neighbors")
    print("")

    print("Step 3: Apply activation (ReLU)")
    let alice_new = relu(alice_updated)
    print("  Final embedding: ReLU(alice + neighbors)")
    print("  Result shape:", alice_new)
    print("")

    // ========================================================================
    // Part 2: Relational Message Passing (R-GCN)
    // ========================================================================
    print("======================================================================")
    print("PART 2: RELATIONAL MESSAGE PASSING (R-GCN)")
    print("======================================================================")
    print("")

    print("R-GCN: Considers relation types in message passing")
    print("Formula: h_i' = σ(Σ_r Σ_{j∈N_r(i)} W_r * h_j)")
    print("")

    print("Example: Update alice's embedding")
    print("")

    print("Relation 1: friend_of → bob")
    let msg_from_bob = relational_aggregate(alice, friend_of, bob)
    print("  Message from bob via friend_of relation")
    print("")

    print("Relation 2: lives_in → tokyo")
    let msg_from_tokyo = relational_aggregate(alice, lives_in, tokyo)
    print("  Message from tokyo via lives_in relation")
    print("")

    print("Relation 3: works_at → company_a")
    let msg_from_company = relational_aggregate(alice, works_at, company_a)
    print("  Message from company_a via works_at relation")
    print("")

    print("Aggregate all messages:")
    let temp_sum = msg_from_bob + msg_from_tokyo
    let all_messages = temp_sum + msg_from_company
    print("  Combined messages from all relations")
    print("")

    print("Apply activation:")
    let alice_rgcn = relu(all_messages)
    print("  Final R-GCN embedding:", alice_rgcn)
    print("")

    // ========================================================================
    // Part 3: Graph Attention (GAT-style)
    // ========================================================================
    print("======================================================================")
    print("PART 3: GRAPH ATTENTION MECHANISM (GAT)")
    print("======================================================================")
    print("")

    print("GAT: Learn importance weights for different neighbors")
    print("Formula: α_ij = softmax(LeakyReLU(a^T [W*h_i || W*h_j]))")
    print("")

    print("Example: Attend to bob's features")
    print("")

    print("Step 1: Compute attention weight")
    let attention_value = graph_attention(alice, bob, bob)
    print("  Query: alice's embedding")
    print("  Key: bob's embedding")
    print("  Value: bob's features")
    print("  Attention-weighted value:", attention_value)
    print("")

    print("Interpretation:")
    print("  - High attention → bob's features are important for alice")
    print("  - Low attention → bob's features less relevant")
    print("")

    // ========================================================================
    // Part 4: Feature Normalization
    // ========================================================================
    print("======================================================================")
    print("PART 4: FEATURE NORMALIZATION")
    print("======================================================================")
    print("")

    print("Normalize embeddings for stability")
    print("")

    let alice_normalized = normalize_features(alice, "l2")
    print("L2 normalized alice:", alice_normalized)
    print("")

    print("Benefits of normalization:")
    print("  - Prevents gradient explosion")
    print("  - Makes training more stable")
    print("  - Improves generalization")
    print("")

    // ========================================================================
    // Part 5: Complete GNN Layer
    // ========================================================================
    print("======================================================================")
    print("PART 5: COMPLETE GNN LAYER")
    print("======================================================================")
    print("")

    print("Combining all components into one GNN layer:")
    print("")

    print("def GNN_layer(node, neighbors, relations):")
    print("  # 1. Message passing from neighbors")
    print("  messages = []")
    print("  for neighbor, relation in zip(neighbors, relations):")
    print("    msg = relational_aggregate(node, relation, neighbor)")
    print("    messages.append(msg)")
    print("")
    print("  # 2. Aggregate messages")
    print("  aggregated = sum(messages)")
    print("")
    print("  # 3. Apply transformation and activation")
    print("  output = ReLU(aggregated)")
    print("")
    print("  # 4. Normalize")
    print("  output = normalize_features(output, 'l2')")
    print("")
    print("  return output")
    print("")

    // ========================================================================
    // Part 6: Applications
    // ========================================================================
    print("======================================================================")
    print("PART 6: APPLICATIONS OF GNN + KG")
    print("======================================================================")
    print("")

    print("1. ENHANCED LINK PREDICTION")
    print("   - Use GNN to refine entity embeddings")
    print("   - Capture multi-hop neighborhood information")
    print("   - Better predictions by considering graph structure")
    print("")

    print("2. NODE CLASSIFICATION")
    print("   - Classify entities based on their neighborhood")
    print("   - Example: Is this entity a person or place?")
    print("")

    print("3. KNOWLEDGE GRAPH COMPLETION")
    print("   - Combine local (TransE/DistMult) and global (GNN) info")
    print("   - More accurate than embeddings alone")
    print("")

    print("4. MULTI-HOP REASONING")
    print("   - Answer complex queries requiring multiple hops")
    print("   - Example: Who are friends of alice's colleagues?")
    print("")

    print("5. TEMPORAL KNOWLEDGE GRAPHS")
    print("   - Track how relationships evolve over time")
    print("   - Predict future connections")
    print("")

    // ========================================================================
    // Summary
    // ========================================================================
    print("======================================================================")
    print("SUMMARY")
    print("======================================================================")
    print("")

    print("✅ Implemented GNN Components:")
    print("  ✓ aggregate_neighbors() - Basic aggregation")
    print("  ✓ relational_aggregate() - R-GCN style")
    print("  ✓ graph_attention() - GAT mechanism")
    print("  ✓ normalize_features() - L2 normalization")
    print("")

    print("📊 Key Concepts:")
    print("  - Message passing: Nodes exchange information")
    print("  - Relational GCN: Relation-specific transformations")
    print("  - Attention: Learn neighbor importance")
    print("  - Normalization: Training stability")
    print("")

    print("🔗 Integration Benefits:")
    print("  - KG embeddings: Capture local triple patterns")
    print("  - GNN: Capture global graph structure")
    print("  - Combined: Best of both worlds!")
    print("")

    print("✅ GNN + KG integration demo completed!")
}
