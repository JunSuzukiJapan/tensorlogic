// Main file that imports the library

import "lib.tl"

tensor radius: float16[1] = [2.0]

main {
    // Use imported constant pi
    area = pi * radius * radius

    // Use imported constant e
    result = e * area

    print("Pi:", pi)
    print("E:", e)
    print("Radius:", radius)
    print("Area:", area)
    print("Result:", result)
}
