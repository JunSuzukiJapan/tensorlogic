#!/bin/bash
# Add f32 versions of all f16 kernels

for shader in shaders/*.metal; do
  echo "Processing $shader..."
  
  # Extract all f16 kernel definitions
  grep -n "^kernel void.*_f16(" "$shader" | while IFS=: read -r linenum kernel_decl; do
    # Extract kernel name
    kernel_name=$(echo "$kernel_decl" | sed 's/^kernel void \([a-z_0-9]*\)_f16.*/\1/')
    
    # Read the entire kernel function
    start_line=$linenum
    
    # Find the end of the function (closing brace at start of line)
    end_line=$(awk "NR>$start_line && /^}/ {print NR; exit}" "$shader")
    
    if [ -z "$end_line" ]; then
      continue
    fi
    
    # Extract the kernel
    kernel_text=$(sed -n "${start_line},${end_line}p" "$shader")
    
    # Convert to f32 version
    f32_kernel=$(echo "$kernel_text" | sed '
      s/_f16/_f32/g
      s/device const half\*/device const float*/g
      s/device half\*/device float*/g  
      s/constant half\*/constant float*/g
      s/threadgroup half /threadgroup float /g
      s/threadgroup half\[/threadgroup float\[/g
      s/ half / float /g
      s/(half)/(float)/g
      s/0\.0h/0.0f/g
      s/1\.0h/1.0f/g
      s/half(/float(/g
    ')
    
    # Check if f32 version already exists
    if ! grep -q "${kernel_name}_f32" "$shader"; then
      echo "Adding ${kernel_name}_f32 to $shader"
      echo "" >> "$shader"
      echo "// F32 version" >> "$shader"
      echo "$f32_kernel" >> "$shader"
    fi
  done
done
