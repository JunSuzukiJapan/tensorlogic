// Buffer Pool Statistics Test
// Tests automatic buffer recycling and displays statistics

fn test_buffer_recycling() {
    let device = get_metal_device()

    // Create some tensors - these will use pooled buffers
    let t1 = zeros(device, [100, 100])
    let t2 = zeros(device, [100, 100])
    let t3 = zeros(device, [100, 100])

    // Perform operations - intermediate buffers will be recycled
    let r1 = add(t1, t2)
    let r2 = add(r1, t3)
    let r3 = mul(r2, t1)

    // More operations to trigger buffer reuse
    let r4 = add(r3, r2)
    let r5 = mul(r4, r1)

    print("Buffer recycling test completed!")
    print("Final result shape: ", shape(r5))
}

test_buffer_recycling()
