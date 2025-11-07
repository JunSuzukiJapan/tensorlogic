// Simple struct test
struct Point {
    x: float32,
    y: float32,
}

struct Vec<T> {
    data: T,
    len: int32,
}

impl Point {
    fn new(x: float32, y: float32) -> Point {
        return Point { x: x, y: y }
    }

    fn distance(self, other: Point) -> float32 {
        let dx = self.x - other.x
        let dy = self.y - other.y
        result := sqrt(dx * dx + dy * dy)
    }
}

main {
    print("Struct test")
    let p1 = Point { x: 1.0, y: 2.0 }
    let p2 = Point::new(3.0, 4.0)
    print("Done")
}
