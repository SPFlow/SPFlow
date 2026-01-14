import torch
import os
from spflow.modules.sums import Sum
from spflow.modules.products import Product, OuterProduct
from spflow.modules.leaves import Normal, Categorical
from spflow.modules.ops import SplitConsecutive
from spflow.meta.data.scope import Scope
from spflow.utils.visualization import visualize

def generate_docs_visualizations():
    output_dir = "docs/source/_static"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating node-based structure visualization...")
    x11 = Normal(scope=0, out_channels=1)
    x12 = Normal(scope=0, out_channels=1)
    x21 = Normal(scope=1, out_channels=1)
    x22 = Normal(scope=1, out_channels=1)

    prod1 = Product([x11, x21])
    prod2 = Product([x12, x21])
    prod3 = Product([x11, x22])
    prod4 = Product([x12, x22])

    pc = Sum([prod1, prod2, prod3, prod4], weights=[0.3, 0.1, 0.2, 0.4])
    visualize(pc, output_path=os.path.join(output_dir, "node-based-structure"), format="svg")
    print(f"Saved to {os.path.join(output_dir, 'node-based-structure.svg')}")

    print("Generating layered structure visualization...")
    x_layered = Normal(scope=[0, 1], out_channels=2)
    prod_layered = OuterProduct(SplitConsecutive(x_layered), num_splits=2)
    pc_layered = Sum(prod_layered, weights=[0.3, 0.1, 0.2, 0.4])
    visualize(pc_layered, output_path=os.path.join(output_dir, "layered-structure"), format="svg")
    print(f"Saved to {os.path.join(output_dir, 'layered-structure.svg')}")

def generate_readme_visualization():
    output_dir = "res"
    os.makedirs(output_dir, exist_ok=True)

    X_idx, Z1_idx, Z2_idx = 0, 1, 2

    leaf_Z1_left = Categorical(scope=Scope([Z1_idx]), out_channels=2, K=3)
    leaf_X_1 = Normal(scope=Scope([X_idx]), out_channels=2)
    leaf_Z2_1 = Normal(scope=Scope([Z2_idx]), out_channels=2)
    leaf_X_2 = Normal(scope=Scope([X_idx]), out_channels=2)

    leaf_Z2_right = Normal(scope=Scope([Z2_idx]), out_channels=2)
    leaf_Z1_1 = Categorical(scope=Scope([Z1_idx]), out_channels=2, K=3)
    leaf_X_3 = Normal(scope=Scope([X_idx]), out_channels=2)
    leaf_Z1_2 = Categorical(scope=Scope([Z1_idx]), out_channels=2, K=3)

    prod_x_z2 = Product(inputs=[leaf_X_1, leaf_Z2_1])
    prod_z2_x = Product(inputs=[leaf_Z2_1, leaf_X_2])
    sum_x_z2 = Sum(inputs=[prod_x_z2, prod_z2_x], out_channels=2)
    prod_z1_sum_xz2 = Product(inputs=[leaf_Z1_left, sum_x_z2])

    prod_z1_x_1 = Product(inputs=[leaf_Z1_1, leaf_X_3])
    prod_z1_x_2 = Product(inputs=[leaf_Z1_2, leaf_X_3])
    sum_z1_x = Sum(inputs=[prod_z1_x_1, prod_z1_x_2], out_channels=2)
    prod_z2_sum_z1x = Product(inputs=[leaf_Z2_right, sum_z1_x])

    root = Sum(inputs=[prod_z1_sum_xz2, prod_z2_sum_z1x], out_channels=1)

    print("Generating README visualization...")
    visualize(root, output_path="res/structure", show_scope=True, show_shape=True, show_params=True, format="svg")
    print(f"Saved to {os.path.join(output_dir, 'structure.svg')}")

if __name__ == "__main__":
    generate_docs_visualizations()
    generate_readme_visualization()
