// Simple buffer test
let device = get_metal_device()
let a = zeros(device, [10, 10])
let b = zeros(device, [10, 10])
let c = add(a, b)
print("Test completed")
print(shape(c))
