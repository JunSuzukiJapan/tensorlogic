// File B imports A (creates circular dependency)
import "circular_a.tl"

tensor b: float16[1] = [2.0]
