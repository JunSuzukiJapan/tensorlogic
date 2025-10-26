use crate::autograd::{GradNode, NodeId, TensorVariant};
use crate::error::TensorResult;
use crate::tensor::Tensor;
use half::f16;
use std::collections::{HashMap, HashSet};

/// 計算グラフ（動的に構築される）
pub struct ComputationGraph {
    pub(crate) nodes: HashMap<NodeId, GradNode>,
    pub(crate) next_id: NodeId,
    pub(crate) enabled: bool,
}

impl ComputationGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
            enabled: true,
        }
    }

    /// 新しいノードを追加
    pub fn add_node(&mut self, node: GradNode) -> NodeId {
        let id = node.id;
        self.nodes.insert(id, node);
        id
    }

    /// 次のノードIDを取得して進める
    pub fn allocate_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// 逆伝播を実行 (f16専用、後方互換性のため)
    ///
    /// # Arguments
    /// * `node_id` - 開始ノード（通常は損失テンソルのノード）
    /// * `grad` - 初期勾配（通常は1.0のスカラー）
    /// * `enabled` - 勾配計算が有効かどうか（外部から渡される）
    ///
    /// # Returns
    /// 各ノードIDに対する勾配のマップ
    pub fn backward(
        &self,
        node_id: NodeId,
        grad: Tensor<f16>,
        _enabled: bool,
    ) -> TensorResult<HashMap<NodeId, Tensor<f16>>> {
        // トポロジカルソートで逆順にノードを処理
        let topo_order = self.topological_sort(node_id)?;
        let mut gradients: HashMap<NodeId, Tensor<f16>> = HashMap::new();

        // 初期勾配を設定
        gradients.insert(node_id, grad);

        // トポロジカル順の逆順で処理（葉ノードから根ノードへ）
        for &current_id in topo_order.iter().rev() {
            if !gradients.contains_key(&current_id) {
                // このノードの勾配がまだ計算されていない場合はスキップ
                continue;
            }

            let grad_output = gradients[&current_id].clone();

            if let Some(node) = self.nodes.get(&current_id) {
                if let Some(ref grad_fn) = node.grad_fn {
                    // 勾配関数を使って入力テンソルの勾配を計算
                    // Note: ここでは入力テンソルの実体が必要だが、現在の設計では
                    // GradientFunction内に保存されている前提とする
                    let input_grads = grad_fn.backward(&grad_output, &[])?;

                    // 各入力ノードに勾配を累積
                    for (input_id, input_grad) in node.inputs.iter().zip(input_grads.iter()) {
                        gradients
                            .entry(*input_id)
                            .and_modify(|existing_grad| {
                                // 勾配を累積（複数パスからの勾配を合算）
                                *existing_grad = existing_grad.add(input_grad).unwrap();
                            })
                            .or_insert_with(|| input_grad.clone());
                    }
                }
            }
        }

        Ok(gradients)
    }

    /// 逆伝播を実行 (TensorVariant版)
    ///
    /// # Arguments
    /// * `node_id` - 開始ノード（通常は損失テンソルのノード）
    /// * `grad` - 初期勾配（通常は1.0のスカラー）
    /// * `enabled` - 勾配計算が有効かどうか（外部から渡される）
    ///
    /// # Returns
    /// 各ノードIDに対する勾配のマップ
    pub fn backward_variant(
        &self,
        node_id: NodeId,
        grad: TensorVariant,
        _enabled: bool,
    ) -> TensorResult<HashMap<NodeId, TensorVariant>> {
        // TensorVariant版では、現時点では簡易的にf16としてbackwardを呼び出す
        // TODO: 将来的にはGradNodeにもVariant対応を追加
        match grad {
            TensorVariant::F16(tensor_f16) => {
                let result = self.backward(node_id, tensor_f16, _enabled)?;
                Ok(result.into_iter().map(|(k, v)| (k, TensorVariant::F16(v))).collect())
            }
            TensorVariant::F32(_) => {
                // f32のbackwardはまだ未実装
                // 暫定的にエラーを返す
                Err(crate::error::TensorError::InvalidOperation(
                    "f32 backward is not yet implemented".to_string()
                ))
            }
        }
    }

    /// トポロジカルソート（DFS）
    fn topological_sort(&self, start: NodeId) -> TensorResult<Vec<NodeId>> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();

        self.dfs(start, &mut visited, &mut order)?;

        Ok(order)
    }

    /// 深さ優先探索でトポロジカル順序を構築
    fn dfs(
        &self,
        node_id: NodeId,
        visited: &mut HashSet<NodeId>,
        order: &mut Vec<NodeId>,
    ) -> TensorResult<()> {
        if visited.contains(&node_id) {
            return Ok(());
        }

        visited.insert(node_id);

        // 入力ノードを再帰的に訪問
        if let Some(node) = self.nodes.get(&node_id) {
            for &input_id in &node.inputs {
                self.dfs(input_id, visited, order)?;
            }
        }

        // 後順（post-order）で追加
        order.push(node_id);
        Ok(())
    }

    /// 計算グラフをクリア
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.next_id = 0;
    }

    /// 勾配計算の有効/無効を切り替え
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = ComputationGraph::new();
        assert_eq!(graph.next_id, 0);
        assert!(graph.enabled);
        assert!(graph.nodes.is_empty());
    }

    #[test]
    fn test_allocate_id() {
        let mut graph = ComputationGraph::new();
        let id1 = graph.allocate_id();
        let id2 = graph.allocate_id();
        let id3 = graph.allocate_id();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);
        assert_eq!(graph.next_id, 3);
    }
}
