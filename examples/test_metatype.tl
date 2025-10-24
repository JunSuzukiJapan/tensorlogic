// Test metatype system - using types as values

entity Person
entity City

relation lives_in(person: Person, city: City)

main {
    // Add some entities
    lives_in(alice, tokyo)
    lives_in(bob, osaka)
    lives_in(charlie, kyoto)

    // Test metatype: Person is a type, can be used as a value
    print("Testing metatype system:")
    print("")

    // Old way (string):
    print("Old way (string):")
    let dim1 = entity_dim("Person")
    print("  entity_dim(\"Person\") =", dim1)
    let vec1 = entity_onehot("Person", "alice")
    print("  entity_onehot(\"Person\", \"alice\") =", vec1)
    print("")

    // New way (type as value):
    print("New way (type as value):")
    let dim2 = entity_dim(Person)
    print("  entity_dim(Person) =", dim2)
    let vec2 = entity_onehot(Person, "alice")
    print("  entity_onehot(Person, \"alice\") =", vec2)
    print("")

    // Test with City type
    print("City type:")
    let city_dim = entity_dim(City)
    print("  entity_dim(City) =", city_dim)
    let tokyo_vec = entity_onehot(City, "tokyo")
    print("  entity_onehot(City, \"tokyo\") =", tokyo_vec)
    print("")

    // Print the type itself
    print("Type as value:")
    print("  Person =", Person)
    print("  City =", City)

    print("")
    print("✅ Metatype system test completed!")
}
